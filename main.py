from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.rag_pipeline import WEAssistant
import uvicorn

# ================= App Initialization =================
# We use FastAPI because it is incredibly fast (built on Starlette) and provides 
# out-of-the-box asynchronous support and automatic OpenAPI (Swagger) documentation.
app = FastAPI(
    title="WE Support AI API",
    description="Backend API for WE Telecom Egypt Assistant. Handles RAG and Vision tasks.",
    version="2.0.0"
)

# ================= AI Core Instantiation (Singleton-like) =================
print("⚙️ Initializing WE Assistant Engine...")
# CRITICAL ARCHITECTURAL DECISION: 
# We instantiate the `WEAssistant` outside of any route functions. 
# This means the heavy FAISS vector database, BM25 indices, and API clients 
# are loaded into RAM *exactly once* when the server boots up. 
# If we put this inside the `/ask` route, the server would reload gigabytes of data 
# on every single user message, which would crash the system instantly.
bot = WEAssistant()

# ================= Data Validation Schema =================
# We use Pydantic to define the exact shape of the JSON payload we expect from Streamlit.
# This acts as a strict firewall: if the frontend sends malformed data (like an integer 
# instead of a string for the query), FastAPI will automatically reject it with a 422 error 
# before it ever touches our AI logic.
class QueryRequest(BaseModel):
    query: str
    file_data: Optional[str] = None       # Optional: Used if the user uploads a PDF/Docx/HTML
    image_base64: Optional[str] = None    # Optional: Used if the user uploads an image
    #use_local_vision: bool = False        # Toggle for the BLIP local vision model

# ================= Main Inference Endpoint =================
@app.post("/ask")
def ask_assistant(request: QueryRequest):
    """
    The primary endpoint that receives user queries, processes them through the RAG pipeline, 
    and returns the LLM's response along with citations.
    """
    # Basic sanitization: Prevent empty queries from wasting API calls to Google.
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً")
    
    try:
        # Pass the validated payload directly into our AI engine.
        result = bot.ask(
            query=request.query, 
            file_data=request.file_data,
            image_base64=request.image_base64,
            #use_local_vision=request.use_local_vision
        )
        return result # Automatically serialized back into JSON by FastAPI
    except Exception as e:
        # Error Isolation: If the GenAI API fails or FAISS crashes, we catch it here.
        # We print it to the server console for debugging, but we return a clean 500 error 
        # to the frontend so the Streamlit app can display a graceful error message 
        # instead of just freezing.
        print(f"❌ Backend Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================= Health Check Endpoint =================
@app.get("/")
def read_root():
    """
    Standard DevOps practice. 
    If you deploy this API via Docker, Kubernetes, or AWS, the load balancer needs 
    a lightweight endpoint to ping to ensure the server is alive and ready to receive traffic.
    """
    return {"status": "Active", "message": "WE Support API is running smoothly!"}

# ================= ASGI Server Runner =================
if __name__ == "__main__":
    # We use Uvicorn as our ASGI web server. 
    # Binding to 0.0.0.0 ensures the server is accessible from outside the localhost 
    # (crucial when you eventually deploy this to a cloud VM or Docker container).
    uvicorn.run("main:app", host="0.0.0.0", port=8000)