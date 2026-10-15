from fastapi import APIRouter, Request , HTTPException
from pydantic import BaseModel
import ollama

router = APIRouter(prefix="/query",tags=["query"])

class QueryRequest(BaseModel):
    question : str

@router.post("/")
def get_query(body: QueryRequest, request: Request):
    store = request.app.state.vector_store

    try:
        results = store.similarity_search(body.question, k=4)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector DB error: {e}")

    context = "\n\n---\n\n".join(r.page_content for r in results)

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "Answer using only the context. If the answer is not there, say you don't know."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.question}"},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")

    return {
        "question": body.question,
        "answer": response["message"]["content"],
        "pages": [r.metadata["page"] + 1 for r in results],
    }