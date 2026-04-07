import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import PROCESSED_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def create_chunks():
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    
    # Wrapping the file reading process in a try-except block. 
    # In a real-world data pipeline, you can never trust the scraped JSON file to always be perfect.
    try:
        with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    # Sanity check: If the web scraper failed and returned an empty list, 
    # we halt the pipeline right here. Building an empty Vector DB would crash the system later.
    if not data:
        print("File is empty!")
        return []

    print(f"🔍 Keys found in first entry: {list(data[0].keys())}")
    
    # Setting up the LangChain text splitter.
    # We use RecursiveCharacterTextSplitter because it's "semantically aware".
    # It tries to split on double newlines (paragraphs) first, then single newlines, then periods.
    # This prevents the chunker from brutally slicing a sentence in half, which would destroy the context for the LLM.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP, # The overlap acts as "glue" so we don't lose context between two adjacent chunks.
        separators=["\n\n", "\n", " | ", ".", " ", ""]
    )
    
    all_chunks = []
    
    # Iterating over every scraped webpage/document in our dataset.
    for entry in data:
        # 1. Grab the raw text payload. We use .strip() to remove trailing/leading whitespace noise.
        raw_content = entry.get('content', "").strip()
        
        # Data Cleaning: Filtering out noisy or empty pages. 
        # If a page has fewer than 10 characters, it's likely a scraper artifact (like "404 Not Found" or an empty div).
        # We drop these so they don't pollute our vector space and ruin search accuracy.
        if not raw_content or len(raw_content) < 10:
            continue 
            
        # 2. Slice the raw document into bite-sized chunks based on our CHUNK_SIZE limit.
        chunks = text_splitter.split_text(raw_content)
        
        # 3. Wrapping and Metadata Injection
        # A chunk without metadata is practically useless because the LLM won't be able to cite its sources later.
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk, # The actual text payload that will be converted into embeddings (vectors).
                "metadata": {
                    # We inject the source URL and Title so the frontend can display them as clickable citations.
                    "source": entry.get('url', ''),
                    "title": entry.get('title', 'WE Service'),
                    "language": entry.get('language', 'AR'), # Fallback default value just in case the key is missing.
                    "chunk_id": i # Super useful for debugging if we need to trace a chunk back to its exact position in the original doc.
                }
            })
            
    print(f" Created {len(all_chunks)} chunks from {len(data)} pages.")
    
    # Returning the fully processed, metadata-enriched chunks ready to be embedded into FAISS.
    return all_chunks