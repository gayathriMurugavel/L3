import os
import json
from crewai.tools import tool
from config.settings import MCP_FILESYSTEM_ROOT


# ═══════════════════════════════════════════
#  FILESYSTEM MCP TOOLS
# ═══════════════════════════════════════════

@tool("MCP_Filesystem_List")
def mcp_list_files(directory: str = "") -> str:
    """
    MCP Filesystem Server Tool.
    Lists all files inside the enterprise knowledge base directory.
    Use this to discover what documents are available.
    Input: optional sub-directory name (empty string for root).
    """
    target = os.path.join(MCP_FILESYSTEM_ROOT, directory)
    if not os.path.exists(target):
        return json.dumps({"error": f"Path not found: {target}"})
    files = os.listdir(target)
    return json.dumps({
        "path": target,
        "files": files,
        "count": len(files)
    }, indent=2)


@tool("MCP_Filesystem_Read")
def mcp_read_file(filename: str) -> str:
    """
    MCP Filesystem Server Tool.
    Reads the content of a text or markdown file from the knowledge base.
    Input: filename (e.g., 'policy.txt' or 'guide.md').
    Returns first 4000 characters.
    """
    path = os.path.join(MCP_FILESYSTEM_ROOT, filename)
    if not os.path.exists(path):
        return f"[MCP ERROR] File not found: {filename}"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return f"[MCP FILE: {filename}]\n\n{content[:4000]}"
    except Exception as e:
        return f"[MCP ERROR] Could not read {filename}: {e}"


# ═══════════════════════════════════════════
#  SQLITE MCP TOOLS
# ═══════════════════════════════════════════

@tool("MCP_SQLite_Get_Documents")
def mcp_sqlite_get_documents(query: str = "") -> str:
    """
    MCP SQLite Server Tool.
    Retrieves list of all indexed documents from the SQLite knowledge base.
    Shows document names, summaries, chunk counts, and indexing dates.
    Input: any string (parameter not used, lists all documents).
    """
    try:
        from src.mcp.sqlite_mcp_server import get_all_documents, init_db
        init_db()
        docs = get_all_documents()
        if not docs:
            return "[SQLite MCP] No documents indexed yet."
        return json.dumps({"documents": docs, "count": len(docs)}, indent=2)
    except Exception as e:
        return f"[SQLite MCP ERROR] {e}"


@tool("MCP_SQLite_Search_History")
def mcp_sqlite_search_history(keyword: str) -> str:
    """
    MCP SQLite Server Tool.
    Searches query history in the SQLite database for similar past queries.
    Useful for finding if a similar question was already answered.
    Input: keyword or phrase to search in query history.
    """
    try:
        from src.mcp.sqlite_mcp_server import search_query_history, init_db
        init_db()
        results = search_query_history(keyword)
        if not results:
            return f"[SQLite MCP] No history found for: {keyword}"
        return json.dumps({"results": results, "count": len(results)}, indent=2)
    except Exception as e:
        return f"[SQLite MCP ERROR] {e}"


@tool("MCP_SQLite_Search_Knowledge")
def mcp_sqlite_search_knowledge(keyword: str) -> str:
    """
    MCP SQLite Server Tool.
    Searches structured knowledge entries stored in SQLite database.
    Returns matching entries by category, title, or content.
    Input: keyword or topic to search for.
    """
    try:
        from src.mcp.sqlite_mcp_server import search_knowledge_entries, init_db
        init_db()
        results = search_knowledge_entries(keyword)
        if not results:
            return f"[SQLite MCP] No knowledge entries found for: {keyword}"
        return json.dumps({"results": results, "count": len(results)}, indent=2)
    except Exception as e:
        return f"[SQLite MCP ERROR] {e}"


@tool("MCP_SQLite_Save_Answer")
def mcp_sqlite_save_answer(data: str) -> str:
    """
    MCP SQLite Server Tool.
    Saves the generated answer and metadata to SQLite query history.
    Input: JSON string with keys 'query' and 'answer'.
    Example: '{"query": "What is policy X?", "answer": "Policy X states..."}'
    """
    try:
        from src.mcp.sqlite_mcp_server import save_query, init_db
        init_db()
        parsed = json.loads(data)
        save_query(
            query=parsed.get("query", ""),
            answer=parsed.get("answer", ""),
        )
        return "[SQLite MCP] Answer saved to query history successfully."
    except json.JSONDecodeError:
        return "[SQLite MCP ERROR] Input must be valid JSON string."
    except Exception as e:
        return f"[SQLite MCP ERROR] {e}"


@tool("MCP_SQLite_Get_Stats")
def mcp_sqlite_get_stats(query: str = "") -> str:
    """
    MCP SQLite Server Tool.
    Returns statistics about the knowledge base:
    total documents, queries processed, and average RAGAS scores.
    Input: any string (parameter not used).
    """
    try:
        from src.mcp.sqlite_mcp_server import get_stats, init_db
        init_db()
        stats = get_stats()
        return json.dumps({"kb_statistics": stats}, indent=2)
    except Exception as e:
        return f"[SQLite MCP ERROR] {e}"


# ═══════════════════════════════════════════
#  RAG SEARCH TOOL
# ═══════════════════════════════════════════

@tool("Knowledge_Base_Search")
def knowledge_base_search(query: str) -> str:
    """
    RAG-powered semantic search tool.
    Searches the ChromaDB vector store for relevant document chunks
    using semantic similarity to the query.
    Input: the question or topic to search for.
    Returns top matching chunks with source metadata.
    """
    try:
        from src.rag.vector_store import retrieve_context
        docs = retrieve_context(query, k=4)
        if not docs:
            return "[RAG] No relevant context found in the knowledge base."
        chunks = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "")
            chunks.append(
                f"[Chunk {i} | Source: {os.path.basename(src)}"
                f"{f' | Page {page}' if page else ''}]\n{d.page_content}"
            )
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        return f"[RAG ERROR] {e}"