from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config.settings import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    TOP_K,
)


def get_embeddings():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def build_vector_store(chunks):
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vs


def load_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def retrieve_context(query: str, k: int = TOP_K):
    vs = load_vector_store()
    results = vs.similarity_search(query, k=k)
    return results