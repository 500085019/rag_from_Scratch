from fastapi import FastAPI
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from routes.query import router as query_router

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "handbook")

@asynccontextmanager
async def life_span(app:FastAPI):
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION):
        raise RuntimeError(f"Collection '{COLLECTION}' not found. Run ingest.py first.")
    print(client)
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    app.state.vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embedding_model,
    )
    print("RAG is ready")
    yield
    client.close()
    print("Closing RAG")

app = FastAPI(title="RAG", description="This is for understanding RAG",lifespan=life_span)
app.include_router(query_router)



