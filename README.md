# JEE Exam Assistant

A multi-service AI-powered assistant designed to help students prepare for the **Joint Entrance Examination (JEE)**. This application provides a conversational interface that answers questions related to **Physics**, **Chemistry**, and **Mathematics**, using a Retrieval-Augmented Generation (RAG) system.

---

## Features

- **Interactive Chat Interface**: A clean, user-friendly web interface for seamless interaction.
- **Microservices Architecture**: Built with four independent services: `frontend`, `chat`, `vector`, and `ai`, managed via Docker Compose.
- **RAG System**: Retrieves study material from ChromaDB to enable contextually accurate answers.
- **AI-Powered Responses**: Uses the **Google Gemini API** to generate detailed, step-by-step explanations.
- **Contextual Awareness**: Maintains chat history and summaries for relevant and coherent conversations.
- **Database Integration**: Uses SQLite for persisting chat data and vector store.
- **Healthcheck Ready**: Includes a startup healthcheck to ensure service readiness, especially for chat and frontend.

---

## Technical Stack

| Layer      | Technology                            |
|------------|----------------------------------------|
| Frontend   | Flask, HTML, CSS, JavaScript           |
| Chat       | Flask, LangGraph, Sentence-Transformers, SQLAlchemy |
| Vector DB  | ChromaDB (Persistent Client)           |
| AI Model   | Google Gemini via `google-generativeai` |
| Containerization | Docker, Docker Compose           |

---

## Architecture

The application is divided into four services:

### 1. `frontend` (Port: `5000`)
- User-facing web interface for chat interaction.

### 2. `chat` (Port: `5001`)
- Core orchestrator.
- Handles chat history, context management, and communication with vector and ai services.

### 3. `vector` (Port: `5002`)
- Hosts ChromaDB.
- Stores embeddings and returns relevant document chunks.

### 4. `ai` (Port: `5003`)
- Connects to Google Gemini API.
- Generates responses using chat context and RAG results.

---

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- A valid **Google Gemini API Key**

---

### Setup Instructions

1. **Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>
```

2. **Configure Environment Variables**

- Navigate to the `./ai` directory and open the `.env` file.
- Replace:

```env
GEMINI_API_KEY=your_gemini_api_key
```

> No changes are typically needed for `./chat/.env` and `./vector/.env`.

3. **Build and Run the Application**

From the root of the project, execute:

```bash
docker-compose up --build
```

4. **Access the App**

Open your browser and go to:

```
http://localhost:5000
```

---

## Directory Structure

```
project-root/
├── ai/               # Gemini API service
│   └── .env
├── chat/             # LangGraph-based chat orchestrator
│   └── .env
├── vector/           # ChromaDB service
│   └── .env
├── frontend/         # Flask-based web interface
├── docker-compose.yml
└── README.md
```

---

## Notes

- Make sure all services are up and running before testing the application.
- If you encounter issues with `chat` service startup, ensure its SQLite database file has the correct permissions and paths.

---

## Future Improvements

- Add user authentication
- Expand knowledge base dynamically
- Enable PDF uploads for personal notes
- Add progress tracking for users

---

## Author

AYOONA MARIA JOHN
