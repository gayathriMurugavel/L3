import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR


def load_documents(directory: str = DOCUMENTS_DIR):
    docs = []
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return docs

    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)
        try:
            if fname.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
            elif fname.lower().endswith((".txt", ".md")):
                docs.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as e:
            print(f"Failed loading {fname}: {e}")
    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)