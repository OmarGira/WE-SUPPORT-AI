# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# --- 1. Environment Variables ---
# We use python-dotenv to load secrets from a local .env file.
# This is a strict security practice to ensure API keys (like Google GenAI) 
# are never hardcoded or accidentally committed to version control systems like GitHub.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- 2. Crawler/Scraping Configuration ---
# The seed URL for our async Playwright crawler. 
# The pipeline will start here and branch out to internal links.
BASE_URL = "https://te.eg/" 

# --- 3. Path Management ---
# Using pathlib instead of standard string concatenation ensures our paths 
# resolve correctly regardless of whether this runs on a Windows dev machine or a Linux Docker container.
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"

# Define exact locations for our data lake (raw vs. processed datasets).
RAW_DATA_PATH = DATA_DIR / "te_eg_raw.json"
PROCESSED_DATA_PATH = DATA_DIR / "te_data_with_language.json" 

# Define where the vector DB components will be persisted.
# FAISS requires a string path in its C++ backend, so we explicitly cast the Path objects to strings.
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index.bin")
BM25_INDEX_PATH = str(DATA_DIR / "bm25_index.pkl")

# --- 4. Text Splitting/Chunking Tuning ---
# These hyperparameters control the granularity of our vector search.
# Note: Since you upgraded to context-enriched semantic chunking earlier, you might want 
# to bump the CHUNK_SIZE up to 1000-1200 later to capture larger tables and full paragraphs. 
# The overlap ensures sentences on the boundary between two chunks are not lost.
CHUNK_SIZE = 400      
CHUNK_OVERLAP = 100   

# --- 5. RAG & Hybrid Search Hyperparameters ---
# The embedding model translates human text into dense vectors.
# We are using 'paraphrase-multilingual-MiniLM-L12-v2' because it's highly optimized 
# for Arabic/English mixed contexts, lightweight enough to run on a CPU, and fast on a GPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Alpha Blending Weights for Hybrid Search:
# We assign 60% weight to Semantic Search (Dense) to handle synonyms and natural language phrasing.
# We retain 40% weight for Keyword Search (Sparse BM25) to ensure exact matches for 
# specific error codes (e.g., "Error 404"), router models, or numeric package IDs.
DENSE_WEIGHT = 0.60  
BM25_WEIGHT = 0.40   

# How many chunks to retrieve from the DB and feed into the LLM context window.
# 12 is a solid sweet spot: it gives the LLM enough context to cross-reference facts 
# without overwhelming the prompt token limit or causing "lost in the middle" hallucination issues.
TOP_K = 12