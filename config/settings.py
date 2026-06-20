import os
from dotenv import load_dotenv

load_dotenv()

# Ollama config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = "ollama/llama3.1:8b"          # CrewAI format
LLM_MODEL_RAW = "llama3.1:8b"             # For LangChain / RAGAS
EMBED_MODEL = "nomic-embed-text:latest"

# Vector DB
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "enterprise_kb"

# RAG
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# Data
DOCUMENTS_DIR = "./data/documents"

# MCP
MCP_FILESYSTEM_ROOT = "./data/documents"