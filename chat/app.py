from flask import Flask, request, Response, stream_with_context
# from flask_cors import CORS
from datetime import datetime
import requests, time, os, re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from models import (
    save_user_question, save_chat_message, save_session, session_exists,
    get_last_n_messages, get_total_chat_messages,
    get_chat_summary, save_chat_summary, clear_database
)
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import Optional

load_dotenv()

# ENV service URLs
VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://vector:5002/query")
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai:5003/generate")
SUMMARY_URL = os.getenv("AI_SUMMARY_URL", f"{AI_SERVICE_URL}/summarize")

# Flask & embeddings
app = Flask(__name__)
# CORS(app)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Chat state type
class ChatState(TypedDict):
    session_id: str
    user_input: str
    input_type: str
    file_data: Optional[dict]
    embedding: Optional[list]
    chat_history: str
    chat_summary: str
    retrieved_chunks: list
    final_answer: str
    error_message: Optional[str]

# Utility
def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s\.\,\?\!\-\(\)]', ' ', text)
    return text

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# Nodes
def check_input(state: ChatState) -> ChatState:
    file = state.get("file_data")
    if file and file.get("type", "").startswith("image/"):
        state["input_type"] = "image"
    else:
        state["input_type"] = "text"
    return state

def route_after_input_check(state: ChatState) -> str:
    print("\nRouting based on input type...")
    input_type = state.get("input_type", "text")
    if input_type == "image":
        return "generate_answer"
    else:
        return "embed_text"

def embed_text(state: ChatState) -> ChatState:
    try:
        input_text = preprocess_text(state.get("user_input", ""))
        state["embedding"] = embedder.encode(input_text).tolist()
    except Exception as e:
        state["error_message"] = f"Embedding error: {e}"
    return state

def query_vector_service(state: ChatState) -> ChatState:
    try:
        response = requests.post(VECTOR_SERVICE_URL, json={
            "embedding": state["embedding"]
        })
        response.raise_for_status()
        state["retrieved_chunks"] = response.json().get("chunks", [])
    except Exception as e:
        state["retrieved_chunks"] = []
        state["error_message"] = f"Vector service error: {e}"
    return state

def generate_answer(state: ChatState) -> ChatState:
    try:
        session_id = state["session_id"]
        if not session_exists(session_id):
            save_session(session_id, datetime.utcnow())

        save_chat_message(session_id, "user", state["user_input"])

        payload = {
            "session_id": session_id,
            "user_input": state["user_input"],
            "chat_summary": state["chat_summary"],
            "chat_history": state["chat_history"],
            "study_material": state["retrieved_chunks"],
            "embedding": state["embedding"],
            "input_type": state["input_type"]
        }

        if state["input_type"] == "image":
            file = state["file_data"]
            payload["file_data"] = {
                "name": file["name"],
                "type": file["type"],
                "bytes": file["stream"].hex()
            }

        res = requests.post(AI_SERVICE_URL, json=payload, stream=True)
        res.raise_for_status()

        answer = ""
        for chunk in res.iter_content(chunk_size=1):
            answer += chunk.decode()

        state["final_answer"] = answer

        save_user_question(session_id, state["user_input"], answer, state["embedding"])
        save_chat_message(session_id, "assistant", answer)

        if get_total_chat_messages(session_id) % 6 == 0:
            last_6 = get_last_n_messages(session_id, 6)
            last_6_str = "\n".join([f"User: {q}\nBot: {a}" for q, a in last_6])
            prev_summary = get_chat_summary(session_id)
            summary_res = requests.post(SUMMARY_URL, json={
                "previous_summary": prev_summary,
                "new_dialogue": last_6_str
            })
            if summary_res.status_code == 200:
                new_summary = summary_res.json().get("summary", "")
                save_chat_summary(session_id, new_summary)

    except Exception as e:
        state["final_answer"] = "Sorry, internal error occurred."
        state["error_message"] = f"AI service error: {e}"

    return state

# LangGraph
def create_workflow():
    graph = StateGraph(ChatState)
    graph.add_node("check_input", check_input)
    graph.add_node("embed_text", embed_text)
    graph.add_node("query_vector", query_vector_service)
    graph.add_node("generate_answer", generate_answer)
    graph.set_entry_point("check_input")
    graph.add_conditional_edges(
        "check_input",
        route_after_input_check,
        {
            "generate_answer": "generate_answer",
            "embed_text": "embed_text"
        }
    )
    graph.add_edge("embed_text", "query_vector")
    graph.add_edge("query_vector", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()

chat_flow = create_workflow()

@app.route("/chat", methods=["POST", "OPTIONS"])  # Add OPTIONS method
def chat_route():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    # Your existing chat_route code here...
    try:
        session_id = request.form.get("session_id")
        message = request.form.get("message", "")
        file = request.files.get("file")
        file_type = request.form.get("file_type")

        init_state = {
            "session_id": session_id,
            "user_input": message,
            "input_type": "text",
            "file_data": None,
            "embedding": None,
            "chat_history": "",
            "chat_summary": "",
            "retrieved_chunks": [],
            "final_answer": "",
            "error_message": None,
        }

        if file and file_type:
            init_state["file_data"] = {
                "name": file.filename,
                "type": file_type,
                "stream": file.read()
            }

        summary = get_chat_summary(session_id)
        chat_history = get_last_n_messages(session_id, 6)
        chat_history_str = "\n".join([f"User: {q}\nBot: {a}" for q, a in chat_history])
        init_state["chat_summary"] = summary
        init_state["chat_history"] = chat_history_str

        @stream_with_context
        def stream_response():
            result = chat_flow.invoke(init_state)
            for char in result["final_answer"]:
                yield char
                time.sleep(0.002)

        return Response(stream_response(), content_type="text/plain")
    except Exception as e:
        print(f"❌ Chat route error: {e}")
        return Response("Sorry, there was an error processing your request.", 
                       content_type="text/plain", status=500)
    
def health_check():
    return {"status": "healthy", "service": "chat"}

if __name__ == "__main__":
    # Only clear database if explicitly needed and we have permissions
    db_path = "data/chat_history.db"
    
    # Check if we need to clear database (optional - only if you want fresh start)
    if os.getenv("CLEAR_DB_ON_START", "false").lower() == "true":
        try:
            from models import clear_database
            clear_database()
            print("🧹 Database cleared on startup")
        except Exception as e:
            print(f"⚠️ Could not clear database on startup: {e}")
    
    print("✅ Chat Service running at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)