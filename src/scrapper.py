# src/unified_scraper.py
import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Importing global settings so our pipeline stays modular and configurable.
from config.settings import BASE_URL, PROCESSED_DATA_PATH 

# A hard cap to prevent the crawler from running infinitely. 
# 100 pages is a solid dataset for a telecom support bot without overwhelming their servers.
MAX_PAGES_TO_SCRAPE = 100 

def is_internal_url(url: str) -> bool:
    """
    Security/Scope Check: We only want to scrape WE's domain. 
    If a link points to Facebook or a third-party payment gateway, we drop it.
    """
    parsed = urlparse(url)
    return parsed.netloc == "" or "te.eg" in parsed.netloc

def normalize_url(url: str) -> str:
    """
    Normalizes URLs to prevent scraping the exact same page twice.
    Removes fragments (e.g., #section1) and trailing slashes.
    """
    url = url.split("#")[0].strip()
    if url.endswith("/") and len(url) > len("https://te.eg/"):
        url = url.rstrip("/")
    return url

# --- 1. Text Cleaning Pipeline ---
def clean_text(raw_text: str) -> str:
    """
    LLMs hate messy whitespace; it eats up valuable context tokens.
    This function flattens the text and replaces chaotic line breaks with a standard delimiter ('|').
    """
    if not raw_text: return ""
    cleaned = raw_text.replace('\n', ' | ')
    cleaned = re.sub(r'\s+', ' ', cleaned) # Collapse multiple spaces into one
    return cleaned.strip(' | ')

# --- 2. Language Detection ---
def detect_language(text: str) -> str:
    """
    A lightweight heuristic to detect if the chunk is Arabic or English.
    We inject this into the FAISS metadata later so we can potentially route 
    English queries to English chunks and Arabic queries to Arabic chunks.
    """
    if not text: return "AR"
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    return "EN" if english_chars > arabic_chars else "AR"

# --- 3. The Extraction Engine (RAG-Optimized) ---
def extract_page_content(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    
    # CRITICAL FOR RAG: We aggressively delete noise.
    # Headers, footers, and navbars repeat on every single page. If we embed them, 
    # our Vector DB will be polluted with redundant data, ruining semantic search.
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else "WE Page"
    
    content_blocks = []
    # Targeted Scraping: Telecom websites use tables heavily for internet packages and pricing.
    # By specifically targeting <th>, <td> (table cells), and <span> (often used for dynamic prices),
    # we ensure the LLM actually learns the pricing tiers.
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "th", "td", "span"]):
        text = clean_text(tag.get_text(" ", strip=True))
        # Drop micro-strings (less than 3 chars) to keep the embedding space clean.
        if text and len(text) >= 3: 
            content_blocks.append(text)

    # Deduplication trick: dict.fromkeys() removes duplicates but maintains the chronological 
    # reading order of the webpage, which is vital for the LLM to understand context.
    unique_blocks = list(dict.fromkeys(content_blocks))
    full_content = " | ".join(unique_blocks)

    # Extract internal links to feed the crawler's BFS (Breadth-First Search) queue.
    links = []
    for a in soup.find_all("a", href=True):
        full_url = normalize_url(urljoin(url, a["href"]))
        if full_url.startswith("http") and is_internal_url(full_url) and ".js" not in full_url:
            links.append(full_url)

    return {
        "url": url,
        "title": title,
        "content": full_content,
        "language": detect_language(full_content), 
        "internal_links": list(dict.fromkeys(links))
    }

# --- 4. The Async Crawler Pipeline ---
async def run_pipeline():
    # Ensure our output directory exists before we start scraping.
    Path(PROCESSED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    visited_urls = set() # O(1) lookup to prevent infinite loops.
    final_data = []
    to_visit = [BASE_URL] # The queue for our BFS crawl.

    print("Starting comprehensive data scraping and cleaning process...")

    # Why Playwright? Modern websites (like WE's) use React/Angular. 
    # The HTML delivered by the server is often empty until JavaScript executes.
    # Playwright acts as a real browser, executing the JS so we can scrape the actual rendered data (like prices).
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True) # Run invisibly in the background.
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        page = await context.new_page()

        while to_visit and len(final_data) < MAX_PAGES_TO_SCRAPE:
            current_url = normalize_url(to_visit.pop(0))
            if current_url in visited_urls: continue

            print(f"Processing [{len(final_data)+1}/{MAX_PAGES_TO_SCRAPE}]: {current_url}")
            try:
                # "networkidle" is the magic word here. 
                # It tells the crawler: "Wait until the page has stopped making API calls for at least 500ms."
                # This guarantees that all dynamic pricing and package details have loaded onto the screen.
                await page.goto(current_url, wait_until="networkidle", timeout=60000)
                
                html = await page.content()
                page_data = extract_page_content(html, current_url)

                # Only save pages that actually have substantial content.
                if len(page_data["content"]) > 50:
                    final_data.append({
                        "url": page_data["url"],
                        "title": page_data["title"],
                        "content": page_data["content"],
                        "language": page_data["language"]
                    })
                    visited_urls.add(current_url)
                    
                    # Add newly discovered links to the queue.
                    for link in page_data["internal_links"]:
                        if link not in visited_urls and link not in to_visit:
                            to_visit.append(link)
                else:
                    visited_urls.add(current_url)

            except Exception as e:
                # Graceful degradation. If a single page timeouts or crashes, log it and keep scraping the rest.
                visited_urls.add(current_url)
                print(f"  Scrape failed: {current_url} | Reason: {str(e)}")

        await browser.close()

    # Flush the curated dataset to disk. This JSON file is what the `chunker.py` will ingest.
    with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nTask completed successfully! Extracted and cleaned {len(final_data)} pages.")
    print(f"Final file is ready at: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    # Standard entry point for asyncio scripts.
    asyncio.run(run_pipeline())