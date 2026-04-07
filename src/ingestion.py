# src/ingestion.py
import torch
from src.chunker import create_chunks
from src.database import HybridRetriever
from config.settings import FAISS_INDEX_PATH, BM25_INDEX_PATH

def run_ingestion():
    # Hardware Acceleration Check
    # Generating embeddings for thousands of text chunks using a Transformer model is highly compute-intensive.
    # Doing this on a CPU would take forever. We explicitly check for a GPU ('cuda') so the embedding 
    # model inside the HybridRetriever can process batches in parallel. It gracefully falls back to 'cpu' if needed.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Initializing ingestion pipeline on device: {device.upper()}")
    
    # 1. Fetching the Semantic Chunks
    # We call our chunker module here. Remember, these aren't just raw strings; they are 
    # dictionaries containing the context-enriched text along with metadata (URLs, titles) needed for citations.
    chunks = create_chunks() 
    
    # Safety Net (Fail-Fast Mechanism)
    # If the web scraper failed or the chunker returned an empty dataset, we need to catch it immediately.
    # Passing an empty list to FAISS or BM25 will trigger deep, obscure indexing errors. 
    # It's standard engineering practice to validate data flow and halt the pipeline gracefully here.
    if not chunks or len(chunks) == 0:
        print(" Critical: Chunker returned zero chunks. Check te_data_with_language.json")
        return
    
    print(f"Successfully received {len(chunks)} chunks. Starting Hybrid Indexing...")
    
    # 2. Building the Hybrid Retriever
    # We initialize the retriever and pass the device so the underlying embedding models know where to run.
    retriever = HybridRetriever(device=device)
    
    # The build_indices method executes two massive operations under the hood:
    # A) FAISS (Dense Search): It passes every chunk through an embedding model (e.g., sentence-transformers) 
    #    to map the text into a high-dimensional vector space. This allows the bot to understand "meaning" (Semantic Search).
    # B) BM25 (Sparse Search): It tokenizes the chunks and builds a statistical index based on term frequencies.
    #    This ensures we don't miss exact keyword matches (like specific error codes or package names).
    retriever.build_indices(chunks)
    
    # 3. Persisting the Indices to Disk
    # Generating embeddings takes time and compute. We don't want to do this every time we start the FastAPI server.
    # By saving the FAISS index (.bin/.pkl) and the BM25 state to disk, our backend can boot up instantly 
    # in production by just loading these pre-computed files into RAM.
    retriever.save(FAISS_INDEX_PATH, BM25_INDEX_PATH)
    print(" Ingestion complete! Knowledge base is fully indexed and saved to disk.")

# Entry point for the ingestion script. 
# In a real-world pipeline, this script is usually triggered by a Cron job or an Airflow DAG 
# whenever the underlying dataset (the scraped JSON files) gets updated with new information.
if __name__ == "__main__":
    run_ingestion()