from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_engine import generate_answer

app = FastAPI()
# Define the data model for the question input
class Question(BaseModel):
    question: str
    image: str = None  # base64-encoded image (optional)

@app.post("/api/")
async def answer_question(q: Question):
    answer, links = await generate_answer(q.question, q.image)
    return {"answer": answer, "links": links}
