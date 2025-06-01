# AskMyNotes - Offline AI-Powered Notes Chatbot 📚

An offline-first document chat application that allows you to have conversations with your documents using local LLMs.

## Features 🌟

- Upload and process `.txt` and `.pdf` files
- Document chunking and local embeddings generation using sentence-transformers
- Local vector storage using ChromaDB
- Offline LLM integration with Mistral-7B
- Modern React-based chat interface
- 100% offline operation - no API costs!

## Project Structure 📁

```
askmynotes/
├── frontend/           # React + JSX + Vite frontend
└── backend/           # Python FastAPI backend
```

## Prerequisites 🛠️

- Python 3.9+
- Node.js 16+
- Git LFS (for downloading model files)

## Setup Instructions 🚀

### Backend

1. Create a Python virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn main:app --reload
```

### Frontend

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

## Tech Stack 💻

### Frontend
- React + Javascript
- Vite
- TailwindCSS
- shadcn/ui components

### Backend
- FastAPI
- sentence-transformers
- ChromaDB
- PyPDF2
- Mistral-7B (via ctransformers)

## License 📄

MIT 
