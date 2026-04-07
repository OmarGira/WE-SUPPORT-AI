# src/database.py
import numpy as np
import faiss
import pickle
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL_NAME, BM25_WEIGHT, DENSE_WEIGHT, TOP_K

class HybridRetriever:
    def __init__(self, device='cuda'):
        # Hardware Check: We default to CUDA if available because encoding thousands of chunks 
        # on a CPU is painfully slow. We want to leverage the GPU for heavy matrix multiplications.
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"⚙️ Initializing Embedding Model on: {self.device}")
        
        # Load the dense embedding model (e.g., multilingual-E5 or similar).
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        
        # Placeholders for our two search engines and data stores.
        self.dense_index = None  # Will hold the FAISS vector database
        self.bm25 = None         # Will hold the sparse BM25 keyword index
        self.documents = []      # Raw text mapping
        self.metadatas = []      # Metadata mapping (URLs, titles)

    def build_indices(self, chunks):
        """Builds both the Dense (FAISS) and Sparse (BM25) indices from scratch."""
        if not chunks:
            print(" No chunks to index.")
            return

        # Separate the raw text and metadata into parallel lists.
        self.documents = [c['text'] for c in chunks]
        self.metadatas = [c['metadata'] for c in chunks]
        
        # 1. FAISS Dense Indexing
        # Notice the "passage: " prefix? Many modern embedding models (like the E5 family) are 
        # instruction-tuned. They need to know if the text they are embedding is a document ("passage:") 
        # or a user question ("query:"). We also inject the title to enrich the semantic context.
        texts_for_embedding = [f"passage: {c['metadata'].get('title', '')} {c['text']}" for c in chunks]
        
        print(f" Encoding {len(texts_for_embedding)} chunks...")
        embeddings = self.model.encode(texts_for_embedding, show_progress_bar=True, convert_to_numpy=True)
        
        # Edge case handling: If there's only one chunk, it might return a 1D array. FAISS expects 2D.
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        dimension = embeddings.shape[1]
        # We use IndexFlatIP (Inner Product). 
        self.dense_index = faiss.IndexFlatIP(dimension)
        # CRITICAL: We normalize the vectors using L2 normalization before adding them. 
        # Inner Product on L2-normalized vectors is mathematically equivalent to Cosine Similarity.
        faiss.normalize_L2(embeddings)
        self.dense_index.add(embeddings.astype('float32'))

        # 2. BM25 Sparse Indexing
        # We tokenize the corpus by splitting on spaces.
        # Engineering trick: We multiply the title by 2 (e.g., "title title content"). 
        # This artificially boosts the keyword weight of the title in the BM25 algorithm!
        tokenized_corpus = [
            (f"{c['metadata'].get('title', '')} " * 2 + c['text']).lower().split() 
            for c in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Hybrid indices built!")

    def save(self, faiss_path, bm25_path):
        """Persists the built indices to disk to avoid re-computing embeddings on every startup."""
        if self.dense_index is not None:
            # FAISS has its own optimized binary format for saving indexes.
            faiss.write_index(self.dense_index, faiss_path)
            # We dump the rest (BM25 object, text arrays, metadatas) into a standard Python pickle file.
            with open(bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25, 
                    'docs': self.documents, 
                    'meta': self.metadatas
                }, f)
            print(f"Indices saved to {faiss_path}")

    def load(self, faiss_path, bm25_path):
        """Loads pre-computed indices into RAM. This makes our backend boot up in milliseconds."""
        print(f"Loading indices from {faiss_path}...")
        try:
            self.dense_index = faiss.read_index(faiss_path)
            
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['docs']
                self.metadatas = data['meta']
            print("Indices loaded successfully!")
        except Exception as e:
            print(f" Error loading indices: {e}")

    def search(self, query, k=TOP_K):
        """The core Hybrid Search logic: Combines semantic search with exact keyword matching."""
        if self.dense_index is None or self.bm25 is None:
            print(" Indices not loaded!")
            return []

        # 1. Dense Search (Semantic understanding)
        # We prefix the user's input with "query:" to match the instruction-tuned model's expected format.
        query_embedding = self.model.encode([f"query: {query}"], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        # Search FAISS across ALL documents to get the raw semantic scores.
        dense_scores, dense_indices = self.dense_index.search(query_embedding.astype('float32'), len(self.documents))
        
        # 2. Sparse Search (Keyword matching)
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # 3. Score Normalization
        # Why do we do this? Dense (Cosine) scores are bounded between [-1, 1] or [0, 1].
        # BM25 scores are unbounded and can easily shoot up to 10 or 20. 
        # If we don't normalize BM25 to a [0, 1] scale, it will completely overpower the semantic dense scores.
        if bm25_scores.max() != bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())

        # 4. Hybrid Reranking (Alpha Blending)
        # We map everything to a flat array and combine the scores using the weights defined in our config.
        final_scores = np.zeros(len(self.documents))
        for score, idx in zip(dense_scores[0], dense_indices[0]):
            final_scores[idx] += score * DENSE_WEIGHT
            
        final_scores += bm25_scores * BM25_WEIGHT

        # Sort the array in descending order to get the highest scoring documents, and slice the top K.
        top_indices = np.argsort(final_scores)[::-1][:k]
        
        # Construct the final result payload
        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": final_scores[idx]
            })
        return results