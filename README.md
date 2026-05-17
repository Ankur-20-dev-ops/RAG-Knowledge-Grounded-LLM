# ⬡ NeuralDoc — RAG Knowledge Assistant

> A production-grade, knowledge-grounded RAG (Retrieval-Augmented Generation) chatbot that answers questions from your own PDF documents — with hallucination detection, streaming responses, model comparison, and PDF export.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-local-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-free_API-purple?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Uploader** | Drag & drop PDFs directly in the browser — auto chunked, embedded & indexed |
| ⚡ **Streaming Responses** | Token-by-token output like ChatGPT |
| 🧠 **Conversation Memory** | Remembers last 3 exchanges for follow-up questions |
| 🔍 **Re-Ranker** | Retrieves 2× chunks, re-ranks by relevance before LLM call |
| ✦ **Hallucination Detection** | Every answer verified against source context |
| 📊 **Confidence Score** | Visual meter showing how well the answer is grounded |
| ⚖ **Model Comparison** | Same question across 4 Groq models side by side |
| 📂 **Document Selector** | Filter which PDFs to search per query |
| 🕓 **Query History** | Click any past question to re-run it |
| 📥 **Export as PDF** | Download full chat as a styled PDF report |

---

## 🗂 Project Structure

```
neuraldoc/
├── app.py          # Streamlit UI — all tabs, chat, sidebar
├── retriever.py    # RAG pipeline — retrieval, reranker, streaming, hallucination check
├── ingest.py       # PDF ingestion — chunking, embedding, ChromaDB storage
├── exporter.py     # Chat → PDF export
├── .env            # Your Groq API key (never commit this)
├── .gitignore      # Excludes .env, chroma_db/, docs/
├── requirements.txt
└── docs/           # Put your PDFs here (git-ignored)
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/neuraldoc.git
cd neuraldoc
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

### 4. (Optional) Pre-ingest PDFs via CLI
```bash
mkdir docs
# copy your PDFs into docs/
python ingest.py
```

### 5. Run the app
```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Groq API (Llama 3.3 70B, Llama 3.1 8B, Gemma 2, Mixtral) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB (local) |
| Orchestration | LangChain |
| PDF Parsing | PyMuPDF |
| PDF Export | fpdf2 |

---

## 🧠 How It Works

```
User Query
    │
    ▼
Embed query with all-MiniLM-L6-v2
    │
    ▼
Search ChromaDB → Top 2K chunks retrieved
    │
    ▼
Re-Ranker (LLM pass) → Top K most relevant chunks
    │
    ▼
Build prompt: context + conversation history + question
    │
    ▼
Groq LLM → Streamed answer
    │
    ▼
Hallucination check → Grounded / Not Grounded + reason
    │
    ▼
Confidence score computed from similarity distances
    │
    ▼
Rendered in UI with sources, badge, meter
```

---

## ⚙ Configuration

All settings are available in the sidebar at runtime:

- **Model** — switch between 4 Groq models
- **Chunks to retrieve** — slider (2–8)
- **Re-ranker** — toggle on/off
- **Streaming** — toggle on/off
- **Document filter** — select which PDFs to search

---

## 📸 Screenshots

> Add screenshots of your running app here.
> `![Chat UI](screenshots/chat.png)`

---

## 🔒 Security Notes

- Never commit your `.env` file — it's in `.gitignore`
- The `chroma_db/` folder and `docs/` folder are also git-ignored
- All embeddings run locally — your documents never leave your machine

---

## 🗺 Roadmap

- [ ] Chunk highlighter — highlight exact sentence used in answer
- [ ] Evaluation dashboard — benchmark accuracy over preset questions
- [ ] Dark / Light mode toggle
- [ ] Multi-user session support
- [ ] Support for `.docx`, `.txt`, `.md` files

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

Built with [LangChain](https://langchain.com), [Groq](https://groq.com), [ChromaDB](https://www.trychroma.com), and [Streamlit](https://streamlit.io).
