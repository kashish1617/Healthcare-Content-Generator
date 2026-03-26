"""Vector store for healthcare terminology and guidelines (RAG)."""
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer


def _load_guidelines() -> dict:
    """Load healthcare guidelines JSON. Returns empty dict if file is missing."""
    data_path = Path(__file__).parent.parent / "data" / "healthcare_guidelines.json"
    if not data_path.exists():
        return {}
    try:
        with open(data_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _documents_from_guidelines(data: dict) -> tuple[list[str], list[dict]]:
    """Convert guidelines JSON into flat documents for embedding."""
    docs: list[str] = []
    meta: list[dict] = []
    for t in data.get("terminology", []):
        text = f"{t['term']}: {t['definition']}"
        docs.append(text)
        meta.append({"type": "terminology", "term": t["term"]})
    for i, rule in enumerate(data.get("formatting_rules", [])):
        docs.append(rule)
        meta.append({"type": "formatting", "index": i})
    template = data.get("patient_summary_template", {})
    for section in template.get("sections", []):
        docs.append(f"Patient summary section: {section}")
        meta.append({"type": "template", "section": section})
    return docs, meta


def get_vector_store(persist_dir: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
    """Build or load Chroma collection with healthcare guidelines."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection_name = "healthcare_guidelines"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Healthcare terminology and formatting guidelines"},
    )

    if collection.count() == 0:
        data = _load_guidelines()
        doc_list, meta_list = _documents_from_guidelines(data)
        if not doc_list:
            return collection, None

        embedder = SentenceTransformer(embedding_model_name)
        embeddings = embedder.encode(doc_list).tolist()
        ids = [f"doc_{i}" for i in range(len(doc_list))]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=doc_list,
            metadatas=meta_list,
        )

    embedder = SentenceTransformer(embedding_model_name)
    return collection, embedder


def retrieve_context(collection, embedder, query: str, top_k: int = 5) -> str:
    """Retrieve relevant guidelines for a generation query."""
    if embedder is None or collection.count() == 0:
        return ""
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k, collection.count()),
        include=["documents"],
    )
    docs = results["documents"][0] if results["documents"] else []
    return "\n".join(docs) if docs else ""
