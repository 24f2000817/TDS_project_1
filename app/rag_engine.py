import openai
from app.utils import get_relevant_chunks
from app.ocr_utils import extract_text_from_image

async def generate_answer(question, image_b64=None):
    context = ""
    
    if image_b64:
        context += extract_text_from_image(image_b64)

    retrieved_chunks = get_relevant_chunks(question)
    prompt = (
        "You are a helpful TA. Use the course material and student questions below to answer.\n\n"
        + "\n\n".join(retrieved_chunks) + "\n\n"
        + "Question: " + question + "\nAnswer:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response['choices'][0]['message']['content'].strip()
    links = [{"url": chunk["url"], "text": chunk["title"]} for chunk in retrieved_chunks[:2]]
    return answer, links
