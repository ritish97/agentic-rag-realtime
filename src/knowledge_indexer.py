"""
src/knowledge_indexer.py

Indexes documents into a FAISS vector store for semantic similarity retrieval.
Acts as the knowledge retrieval engine for the Agentic RAG system, enabling
the agent to ground its reasoning in private enterprise documentation.
"""

import os
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore


# ---------------------------------------------------------------------------
# Embedding model: sentence-transformers/all-MiniLM-L6-v2
# A lightweight, highly efficient local embedding model that runs entirely on CPU.
# Avoids external embedding API costs and ensures private document confidentiality.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_CACHED_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance."""
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is None:
        _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _CACHED_EMBEDDINGS


def get_indexed_documents_info(kb_dir: str) -> List[Dict[str, Any]]:
    """
    Inspect the knowledge base directory and return metadata for all supported files.

    Returns:
        List of dicts with: name, path, size_kb, extension, modified_time
    """
    if not os.path.exists(kb_dir):
        return []

    info_list = []
    valid_exts = {".txt", ".pdf", ".md"}

    for filename in sorted(os.listdir(kb_dir)):
        if filename.startswith("."):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_exts:
            full_path = os.path.join(kb_dir, filename)
            try:
                stat = os.stat(full_path)
                info_list.append({
                    "name": filename,
                    "path": full_path,
                    "size_kb": stat.st_size / 1024,
                    "extension": ext,
                    "modified_time": stat.st_mtime,
                })
            except Exception:
                continue

    return info_list


def index_knowledge_base(
    kb_dir: str,
    index_path: str = "kb_faiss_index",
    force_reindex: bool = False,
) -> FAISS:
    """
    Load every .pdf, .txt, and .md file from kb_dir, chunk the text, build a FAISS
    vector index, and persist it to disk.

    If the index already exists on disk it is loaded directly — re-indexing is
    skipped so the agent starts up fast after the first run.

    Args:
        kb_dir:         Directory containing source documents (.pdf, .txt, .md).
        index_path:     Path where the FAISS index folder will be saved.
        force_reindex:  If True, forces rebuilding even if index files exist.

    Returns:
        A ready-to-query FAISS vector store.
    """
    # If the index was already built and not forced, load it and return immediately.
    if os.path.exists(index_path) and not force_reindex:
        try:
            print(f"[Indexer] Loading existing FAISS index from '{index_path}'")
            return load_index(index_path)
        except Exception as exc:
            print(f"[Indexer] Could not load existing index ({exc}). Re-indexing from '{kb_dir}' …")

    print(f"[Indexer] Building new FAISS index from '{kb_dir}'")

    # Ensure kb_dir exists
    os.makedirs(kb_dir, exist_ok=True)

    # --- Step 1: Load documents using loaders for PDF, TXT, and Markdown ---
    documents = []

    pdf_loader = DirectoryLoader(
        kb_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
    )
    txt_loader = DirectoryLoader(
        kb_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    md_loader = DirectoryLoader(
        kb_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )

    for loader in (pdf_loader, txt_loader, md_loader):
        try:
            docs = loader.load()
            if docs:
                documents.extend(docs)
                print(f"[Indexer]   Loaded {len(docs)} document chunk(s) via {loader.__class__.__name__}")
        except Exception as exc:
            print(f"[Indexer]   Warning: loader {loader.__class__.__name__} failed — {exc}")

    if not documents:
        print(f"[Indexer] No documents found in '{kb_dir}'. Creating default starter guide.")
        from langchain_core.documents import Document
        documents = [
            Document(
                page_content=(
                    "Agentic RAG Knowledge Base: Ready for document uploads. "
                    "Place .pdf, .txt, or .md files in the data/knowledge_base directory "
                    "to ground the agent in proprietary company documents, policies, and forecasts."
                ),
                metadata={"source": "system_starter_guide.txt"},
            )
        ]

    # --- Step 2: Split documents into overlapping chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Indexer] Split into {len(chunks)} chunks with sentence preservation")

    # --- Step 3: Embed and index ---
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # --- Step 4: Persist to disk ---
    vector_store.save_local(index_path)
    print(f"[Indexer] FAISS index saved to '{index_path}'")

    return vector_store


def load_index(index_path: str) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        index_path: Path to the directory created by FAISS.save_local().

    Returns:
        Loaded FAISS vector store ready for similarity search.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[Indexer] Loaded FAISS index from '{index_path}'")
    return vector_store


def search_knowledge_base_with_metadata(
    query: str,
    vector_store: FAISS,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Perform a similarity search and return chunks with their document source metadata.

    Args:
        query:        Natural-language search query.
        vector_store: Loaded FAISS index.
        k:            Number of top results to return.

    Returns:
        List of dicts: [{"content": str, "source": str, "score": Optional[float]}]
    """
    results = []
    try:
        # Use similarity_search_with_score if available
        docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
        for doc, score in docs_and_scores:
            source = os.path.basename(doc.metadata.get("source", "knowledge_base"))
            results.append({
                "content": doc.page_content,
                "source": source,
                "score": float(score),
            })
    except Exception:
        docs = vector_store.similarity_search(query, k=k)
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "knowledge_base"))
            results.append({
                "content": doc.page_content,
                "source": source,
                "score": None,
            })
    return results


def search_knowledge_base(
    query: str,
    vector_store: FAISS,
    k: int = 3,
) -> List[str]:
    """
    Perform a similarity search and return top-k text strings.
    Maintains backward compatibility with rag_tool.py.
    """
    items = search_knowledge_base_with_metadata(query, vector_store, k=k)
    return [f"[{item['source']}] {item['content']}" for item in items]
