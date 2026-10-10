from fastapi import FastAPI

app = FastAPI(
    title="RAG",
    description="this is will be understanding for rag"
)

@app.get("/")
def root():
    return {"this is the rag file"}