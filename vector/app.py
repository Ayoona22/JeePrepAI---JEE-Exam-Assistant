from flask import Flask, request, jsonify
from chromadb import PersistentClient
# from chromadb.utils.embedding_functions import embedding_function_factory
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Constants
CHROMA_PATH = os.getenv("CHROMA_PATH", "./rag_store")
TOP_K = int(os.getenv("TOP_K", "5"))
DISTANCE_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

# Initialize Chroma client
chroma_client = PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="rag_documents")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    embedding = data.get("embedding")

    if not embedding:
        return jsonify({"error": "Missing embedding"}), 400

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=TOP_K,
            include=["documents", "distances"]
        )

        chunks = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        filtered = [
            chunk for chunk, dist in zip(chunks, distances)
            if dist < DISTANCE_THRESHOLD
        ]

        # fallback: return top 2 even if none pass threshold
        if not filtered and chunks:
            filtered = chunks[:2]

        return jsonify({"chunks": filtered})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"📚 Vector Service running with Chroma at: {CHROMA_PATH}")
    app.run(host="0.0.0.0", port=5002, debug=True)
