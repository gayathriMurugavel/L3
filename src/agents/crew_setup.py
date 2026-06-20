from crewai import Agent, Task, Crew, Process, LLM
from config.settings import LLM_MODEL, OLLAMA_BASE_URL
from src.mcp.mcp_tools import (
    mcp_list_files,
    mcp_read_file,
    mcp_sqlite_get_documents,
    mcp_sqlite_search_history,
    mcp_sqlite_search_knowledge,
    mcp_sqlite_save_answer,
    mcp_sqlite_get_stats,
    knowledge_base_search,
)


def get_llm():
    return LLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
    )


def get_planner_context(user_query: str) -> str:
    """Fetch deterministic inputs for the planner outside the agent loop."""
    stats = mcp_sqlite_get_stats.run(user_query)
    history = mcp_sqlite_search_history.run(user_query)
    context = (
        "Knowledge base statistics:\n"
        f"{stats}\n\n"
        "Similar query history:\n"
        f"{history}"
    )
    return context.replace("{", "{{").replace("}", "}}")


def build_crew(user_query: str):
    llm = get_llm()
    planner_context = get_planner_context(user_query)

    # ── AGENT 1: PLANNER (Bonus) ───────────────────────────────────────
    planner_agent = Agent(
        role="Query Planner",
        goal=(
            "Analyze the user query and create a clear research plan "
            "with sub-questions to guide the research agent."
        ),
        backstory=(
            "Expert query analyst who breaks complex enterprise questions "
            "into structured sub-questions for efficient research."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ── AGENT 2: RESEARCH ─────────────────────────────────────────────
    research_agent = Agent(
        role="Knowledge Research Specialist",
        goal=(
            "Retrieve the most accurate and relevant information "
            "from the enterprise knowledge base for the given query."
        ),
        backstory=(
            "Senior knowledge engineer with deep expertise in semantic "
            "search, document retrieval, and enterprise knowledge systems. "
            "Uses RAG and MCP tools to find the best context."
        ),
        tools=[
            knowledge_base_search,
            mcp_list_files,
            mcp_read_file,
            mcp_sqlite_get_documents,
            mcp_sqlite_search_knowledge,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    # ── AGENT 3: RESPONSE GENERATION ──────────────────────────────────
    response_agent = Agent(
        role="Response Generation Expert",
        goal=(
            "Generate a clear, accurate, well-structured answer "
            "based solely on the retrieved context, then save it."
        ),
        backstory=(
            "Senior enterprise analyst specializing in translating "
            "raw knowledge context into actionable, well-cited responses "
            "for executives and employees."
        ),
        tools=[mcp_sqlite_save_answer],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ── AGENT 4: REVIEWER (Bonus) ──────────────────────────────────────
    reviewer_agent = Agent(
        role="Answer Reviewer",
        goal=(
            "Review the generated answer for hallucinations, "
            "missing citations, logical gaps, and factual errors."
        ),
        backstory=(
            "Strict quality assurance specialist who ensures all answers "
            "are grounded in retrieved context with zero hallucination. "
            "Flags any claims not supported by the source documents."
        ),
        tools=[knowledge_base_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ── AGENT 5: EVALUATION ───────────────────────────────────────────
    evaluation_agent = Agent(
        role="Response Quality Evaluator",
        goal=(
            "Produce the final polished answer with a quality assessment "
            "score and brief explanation."
        ),
        backstory=(
            "Chief knowledge quality officer responsible for final "
            "approval of all enterprise responses. Provides structured "
            "quality ratings and ensures the best possible output."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ── TASKS ─────────────────────────────────────────────────────────

    plan_task = Task(
        description=(
            "Analyze this query: '{query}'\n\n"
            "Use the pre-fetched planning context below. Do not call tools.\n\n"
            "Planning context:\n"
            f"{planner_context}\n\n"
            "1. Summarize what prior history and KB coverage imply for this query.\n"
            "2. Create a research plan with 2-3 specific sub-questions.\n"
            "3. Keep the output compact and directly useful for the research agent.\n"
            "Output: A structured research plan."
        ),
        expected_output=(
            "Research plan with sub-questions and approach strategy."
        ),
        agent=planner_agent,
    )

    research_task = Task(
        description=(
            "Using the research plan for: '{query}'\n\n"
            "1. Use Knowledge_Base_Search to find relevant context.\n"
            "2. Use MCP_Filesystem tools to check available documents.\n"
            "3. Use MCP_SQLite tools to find structured knowledge entries.\n"
            "4. Collect all relevant context with source labels.\n"
            "Output: All retrieved context chunks."
        ),
        expected_output=(
            "Comprehensive context with labeled sources ready for response generation."
        ),
        agent=research_agent,
        context=[plan_task],
    )

    response_task = Task(
        description=(
            "Using ONLY the retrieved context, answer: '{query}'\n\n"
            "Requirements:\n"
            "- Be factual and concise\n"
            "- Structure with headers if needed\n"
            "- Cite sources at the end\n"
            "- Do NOT hallucinate beyond the context\n"
            "- Save the answer using MCP_SQLite_Save_Answer tool\n"
            "  with JSON: {{\"query\": \"<query>\", \"answer\": \"<answer>\"}}"
        ),
        expected_output=(
            "A structured, well-cited answer grounded in retrieved context."
        ),
        agent=response_agent,
        context=[research_task],
    )

    review_task = Task(
        description=(
            "Review the generated answer for: '{query}'\n\n"
            "Check for:\n"
            "1. Hallucinations (claims not in context)\n"
            "2. Missing or incorrect citations\n"
            "3. Completeness (are all aspects covered?)\n"
            "4. Clarity and structure\n"
            "Use Knowledge_Base_Search to verify any suspicious claims.\n"
            "Output: Reviewed answer with any corrections noted."
        ),
        expected_output=(
            "Verified answer with review notes listing any corrections made."
        ),
        agent=reviewer_agent,
        context=[research_task, response_task],
    )

    eval_task = Task(
        description=(
            "Produce the FINAL output for: '{query}'\n\n"
            "Based on the research, response, and review:\n"
            "1. Write the final polished answer\n"
            "2. Give a quality score (1-10)\n"
            "3. Provide brief evaluation:\n"
            "   - Accuracy: X/10\n"
            "   - Completeness: X/10\n"
            "   - Clarity: X/10\n"
            "   - Source Grounding: X/10\n"
        ),
        expected_output=(
            "Final answer + structured quality evaluation with scores."
        ),
        agent=evaluation_agent,
        context=[research_task, response_task, review_task],
    )

    crew = Crew(
        agents=[
            planner_agent,
            research_agent,
            response_agent,
            reviewer_agent,
            evaluation_agent,
        ],
        tasks=[
            plan_task,
            research_task,
            response_task,
            review_task,
            eval_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew, research_task, response_task