from sentence_transformers import SentenceTransformer
import faiss
import json

# Load and index posts
def get_relevant_chunks(query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query])

    with open("app/data/discourse_posts.jsonl") as f:
        docs = [json.loads(line) for line in f]

    # assume we have built FAISS index in setup
    index = faiss.read_index("app/data/vector_store.faiss")
    D, I = index.search(query_vec, top_k)

    return [docs[i] for i in I[0]]
