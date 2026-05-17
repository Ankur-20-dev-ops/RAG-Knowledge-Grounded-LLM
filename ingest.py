import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER   = "docs"
CHROMA_DB_PATH = "chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Core ingestion ─────────────────────────────────────────────────────────────
def ingest_file(filepath: str, progress_cb=None) -> dict:
    """Ingest a single PDF file into ChromaDB. Returns status dict."""
    try:
        if progress_cb:
            progress_cb(f"Loading {os.path.basename(filepath)}…")

        loader = PyMuPDFLoader(filepath)
        documents = loader.load()

        if progress_cb:
            progress_cb(f"Chunking {len(documents)} pages…")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(documents)

        if progress_cb:
            progress_cb(f"Embedding {len(chunks)} chunks…")

        # Add to existing vectorstore or create new
        if os.path.exists(CHROMA_DB_PATH):
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings
            )
            vectorstore.add_documents(chunks)
        else:
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH
            )

        return {
            "success": True,
            "filename": os.path.basename(filepath),
            "pages": len(documents),
            "chunks": len(chunks)
        }

    except Exception as e:
        return {
            "success": False,
            "filename": os.path.basename(filepath),
            "error": str(e)
        }


def ingest_folder(folder: str = DOCS_FOLDER, progress_cb=None) -> list:
    """Ingest all PDFs in a folder."""
    results = []
    pdfs = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    for pdf in pdfs:
        path = os.path.join(folder, pdf)
        result = ingest_file(path, progress_cb)
        results.append(result)
    return results


def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit uploaded file to /docs and return its path."""
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    dest = os.path.join(DOCS_FOLDER, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def delete_doc(filename: str) -> bool:
    """Remove a PDF from docs/ folder (does not remove from ChromaDB)."""
    path = os.path.join(DOCS_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def reset_vectorstore():
    """Wipe ChromaDB and start fresh."""
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Starting Ingestion ===")
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    pdfs = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
    if not pdfs:
        print("No PDFs found in /docs. Add some and retry.")
    else:
        results = ingest_folder(progress_cb=print)
        for r in results:
            if r["success"]:
                print(f"✓ {r['filename']} — {r['pages']} pages, {r['chunks']} chunks")
            else:
                print(f"✗ {r['filename']} — {r['error']}")
    print("=== Done ===")
