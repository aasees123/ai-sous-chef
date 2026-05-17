from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

app = Flask(__name__)

# Embeddings model (same as before)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vector store
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)


# Health check route (NEW - for browser testing)
@app.route("/", methods=["GET"])
def home():
    return "Cooking Copilot API is running"


# Main RAG endpoint (UNCHANGED logic, safer input handling)
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"]

    # Retrieve relevant documents
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Prompt for LLM
    prompt = f"""
You are a cooking assistant.

Use the context below to answer the user.

Context:
{context}

User question:
{question}
"""

    # Call Ollama
    response = ollama.chat(
        model="llama3", messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
