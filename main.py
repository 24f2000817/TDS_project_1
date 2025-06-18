from flask import Flask, request, jsonify
import base64, io, pickle, faiss
from PIL import Image
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Load FAISS index and embedding metadata ===
index = faiss.read_index("tds_index.faiss")
with open("embedding_data.pkl", "rb") as f:
    embedding_data = pickle.load(f)

# === Load models (use smaller ones for speed) ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# === Initialize Flask ===
app = Flask(__name__)

def extract_text_from_image(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return pytesseract.image_to_string(image)
    except:
        return ""

def retrieve(query, top_k=3):
    q_emb = embed_model.encode(query, convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb)
    D, I = index.search(np.array([q_emb.astype("float32")]), top_k)
    return [embedding_data[i] for i in I[0]]

def generate_answer(query, contexts):
    context = "\n\n".join([ctx["combined_text"] for ctx in contexts])
    prompt = f"Answer the question based on the following discussion excerpts:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    outputs = gen_model.generate(**inputs, max_length=256, num_beams=5)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/api/", methods=["POST"])
def answer_api():
    data = request.get_json()
    question = data.get("question", "")
    image_text = extract_text_from_image(data["image"]) if "image" in data else ""
    question += " " + image_text

    results = retrieve(question)
    answer = generate_answer(question, results)

    links = [{
        "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{r['topic_id']}/{r['post_numbers'][0]}",
        "text": r["topic_title"]
    } for r in results]

    return jsonify({"answer": answer, "links": links})

if __name__ == "__main__":
    app.run(debug=True)
