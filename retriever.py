import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = "chroma_db"

# ── Embeddings ─────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Vector store ───────────────────────────────────────────────────────────────
def load_vectorstore():
    if not os.path.exists(CHROMA_DB_PATH):
        return None
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

# ── Available models ───────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 · 70B",
    "llama-3.1-8b-instant":    "Llama 3.1 · 8B (fast)",
    "gemma2-9b-it":            "Gemma 2 · 9B",
    "mixtral-8x7b-32768":      "Mixtral · 8x7B",
}

def get_llm(model_name: str, temperature: float = 0.2):
    return ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature
    )

# ── Prompts ────────────────────────────────────────────────────────────────────
ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question", "history"],
    template="""You are a precise, helpful research assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I don't have enough information in the provided documents to answer this."

Conversation so far:
{history}

Context from documents:
{context}

Question: {question}

Answer (be concise and accurate):"""
)

HALLUCINATION_PROMPT = PromptTemplate(
    input_variables=["context", "answer"],
    template="""Is the following answer fully supported by the context provided?
Reply with ONLY one of: "Grounded" or "Not Grounded", followed by a dash and one short reason (max 10 words).

Context:
{context}

Answer:
{answer}

Verdict:"""
)

RERANK_PROMPT = PromptTemplate(
    input_variables=["question", "chunks"],
    template="""Given the question below, rank the following text chunks from MOST to LEAST relevant.
Return ONLY a comma-separated list of chunk numbers in order of relevance (e.g. 3,1,4,2).

Question: {question}

Chunks:
{chunks}

Ranking:"""
)

# ── Reranker ───────────────────────────────────────────────────────────────────
def rerank_chunks(question: str, docs_with_scores: list, llm) -> list:
    """Re-rank retrieved chunks using LLM for better relevance."""
    if len(docs_with_scores) <= 2:
        return docs_with_scores

    chunks_text = "\n\n".join([
        f"[{i+1}] {doc.page_content[:300]}"
        for i, (doc, _) in enumerate(docs_with_scores)
    ])

    try:
        chain = RERANK_PROMPT | llm
        result = chain.invoke({"question": question, "chunks": chunks_text})
        ranking_str = result.content.strip().replace(" ", "")
        indices = [int(x) - 1 for x in ranking_str.split(",") if x.isdigit()]
        reranked = [docs_with_scores[i] for i in indices if i < len(docs_with_scores)]
        # Add any missed
        for i, item in enumerate(docs_with_scores):
            if i not in indices:
                reranked.append(item)
        return reranked
    except Exception:
        return docs_with_scores  # fallback to original order

# ── Format chat history ────────────────────────────────────────────────────────
def format_history(history: list) -> str:
    if not history:
        return "No prior conversation."
    lines = []
    for msg in history[-6:]:  # last 3 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(lines)

# ── Confidence score ───────────────────────────────────────────────────────────
def compute_confidence(scores: list) -> int:
    """Convert similarity distances to a 0-100 confidence score."""
    if not scores:
        return 0
    # ChromaDB returns L2 distances — lower = better
    avg = sum(scores) / len(scores)
    # Map: 0.0 → 100%, 2.0 → 0%
    confidence = max(0, min(100, int((1 - avg / 2.0) * 100)))
    return confidence

# ── Main retrieval function ────────────────────────────────────────────────────
def retrieve_and_answer(
    query: str,
    k: int = 4,
    model_name: str = "llama-3.3-70b-versatile",
    history: list = None,
    use_reranker: bool = True,
    selected_docs: list = None
) -> dict:
    if history is None:
        history = []

    vectorstore = load_vectorstore()
    if vectorstore is None:
        return {
            "answer": "No documents have been ingested yet. Please upload PDFs first.",
            "sources": [],
            "hallucination_check": "N/A",
            "confidence": 0
        }

    llm = get_llm(model_name)

    # ── Retrieve ──
    raw_results = vectorstore.similarity_search_with_score(query, k=k * 2)

    # ── Filter by selected docs ──
    if selected_docs:
        raw_results = [
            (doc, score) for doc, score in raw_results
            if any(sel in doc.metadata.get("source", "") for sel in selected_docs)
        ]

    if not raw_results:
        return {
            "answer": "No relevant content found in the selected documents.",
            "sources": [],
            "hallucination_check": "N/A",
            "confidence": 0
        }

    # ── Rerank ──
    if use_reranker and len(raw_results) > 2:
        results = rerank_chunks(query, raw_results, llm)[:k]
    else:
        results = raw_results[:k]

    # ── Build context ──
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    scores = [float(score) for _, score in results]
    confidence = compute_confidence(scores)

    sources = [
        {
            "content": doc.page_content[:250] + "…",
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": round(float(score), 3)
        }
        for doc, score in results
    ]

    # ── Answer ──
    history_text = format_history(history)
    answer_chain = ANSWER_PROMPT | llm
    answer = answer_chain.invoke({
        "context": context,
        "question": query,
        "history": history_text
    }).content

    # ── Hallucination check ──
    hallucination_chain = HALLUCINATION_PROMPT | llm
    verdict = hallucination_chain.invoke({
        "context": context,
        "answer": answer
    }).content.strip()

    return {
        "answer": answer,
        "sources": sources,
        "hallucination_check": verdict,
        "confidence": confidence
    }

# ── Streaming version ──────────────────────────────────────────────────────────
def retrieve_and_stream(
    query: str,
    k: int = 4,
    model_name: str = "llama-3.3-70b-versatile",
    history: list = None,
    use_reranker: bool = True,
    selected_docs: list = None
):
    """Generator that yields answer tokens as they stream in."""
    if history is None:
        history = []

    vectorstore = load_vectorstore()
    if vectorstore is None:
        yield "⚠ No documents ingested yet. Please upload PDFs first."
        return

    llm = get_llm(model_name, temperature=0.2)

    raw_results = vectorstore.similarity_search_with_score(query, k=k * 2)

    if selected_docs:
        raw_results = [
            (doc, score) for doc, score in raw_results
            if any(sel in doc.metadata.get("source", "") for sel in selected_docs)
        ]

    if not raw_results:
        yield "No relevant content found in the selected documents."
        return

    if use_reranker and len(raw_results) > 2:
        results = rerank_chunks(query, raw_results, llm)[:k]
    else:
        results = raw_results[:k]

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    history_text = format_history(history)

    prompt = ANSWER_PROMPT.format(
        context=context,
        question=query,
        history=history_text
    )

    for chunk in llm.stream(prompt):
        yield chunk.content

    # Store results in a way app.py can access after streaming
    scores = [float(score) for _, score in results]
    retrieve_and_stream.last_results = {
        "sources": [
            {
                "content": doc.page_content[:250] + "…",
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "score": round(float(score), 3)
            }
            for doc, score in results
        ],
        "context": context,
        "confidence": compute_confidence(scores)
    }

retrieve_and_stream.last_results = {}

# ── Get hallucination verdict separately ──────────────────────────────────────
def check_hallucination(answer: str, context: str, model_name: str = "llama-3.3-70b-versatile") -> str:
    llm = get_llm(model_name)
    chain = HALLUCINATION_PROMPT | llm
    return chain.invoke({"context": context, "answer": answer}).content.strip()

# ── List ingested docs ─────────────────────────────────────────────────────────
def list_ingested_docs() -> list:
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return []
    try:
        all_docs = vectorstore.get()
        sources = set()
        for meta in all_docs.get("metadatas", []):
            src = meta.get("source", "")
            if src:
                sources.add(os.path.basename(src))
        return sorted(list(sources))
    except Exception:
        return []
