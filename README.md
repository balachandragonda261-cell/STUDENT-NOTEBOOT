# 📚 NoteBot — AI RAG Tutor

**NoteBot** is an intelligent, AI-powered study assistant built with [Streamlit](https://streamlit.io/). It uses a **Retrieval-Augmented Generation (RAG)** pipeline to let you upload any PDF document and ask natural language questions — getting precise, context-aware answers generated from your own notes.

---

## ✨ Features

- 📄 **PDF Ingestion** — Upload any PDF; text is extracted and indexed automatically.
- 🔍 **Semantic Search** — Uses vector embeddings to find the most relevant passages from your document.
- 🤖 **AI Answer Generation** — Sends retrieved context to Google Gemini or OpenAI to generate a single, clear, complete answer.
- 🆓 **Free Local Embeddings** — Uses HuggingFace `all-MiniLM-L6-v2` locally on CPU — no embedding API key needed.
- 🔒 **Secure** — API keys are entered only in the sidebar and are never stored or logged.
- 🎨 **Premium UI** — Glassmorphic dark-mode design with smooth animations.

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| Frontend / App | Streamlit |
| PDF Parsing | PyPDF2 |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings (Free) | HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) |
| Embeddings (Paid) | OpenAI `text-embedding-ada-002` |
| Vector Store | FAISS (local, in-memory) |
| LLM (Default) | Google Gemini (`gemini-3.5-flash`) |
| LLM (Alternative) | OpenAI GPT-3.5-Turbo |
| LangChain | LangChain Classic + Core + Community + Google GenAI |

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/balachandragonda261-cell/STUDENT-NOTEBOOT.git
cd STUDENT-NOTEBOOT
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## ⚙️ Environment Variables (Optional)

You can optionally set API keys as environment variables instead of entering them in the sidebar each time:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key (from [Google AI Studio](https://aistudio.google.com/)) |
| `OPENAI_API_KEY` | Your OpenAI API key (from [OpenAI Platform](https://platform.openai.com/)) |

> ⚠️ **Never commit** a `.env` file containing real keys to this repository.

---

## 📖 How to Use

1. Open the app at [http://localhost:8501](http://localhost:8501).
2. In the **sidebar**, select:
   - **LLM Provider**: `Google Gemini` (recommended) or `OpenAI`.
   - **Gemini Model**: `gemini-3.5-flash` (default, free-tier supported).
   - **Embeddings Provider**: `HuggingFace (Free & Local)` (no API key needed).
3. Enter your **Gemini API Key** (get one free at [aistudio.google.com](https://aistudio.google.com/)).
4. **Upload a PDF** file (e.g., your lecture notes or textbook chapter).
5. Wait for the ✅ **"Notes indexed successfully!"** message.
6. Type your question in the text box and get a single, clear answer based on your document!

---

## 🔐 Security Notes

- API keys are **never stored** in source code or committed to Git.
- `.env` files, virtual environments, FAISS indexes, and uploaded PDFs are excluded via `.gitignore`.
- Error messages are sanitized to prevent accidental key leaks in the UI.

---

## 📁 Project Structure

```
NEWPROJECT/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from Git
└── README.md           # This file
```

---

## 📦 Requirements

See [requirements.txt](requirements.txt) for the full list of dependencies.

---

## 📄 License

This project is open-source and available for educational purposes.
