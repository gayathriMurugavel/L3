from crewai import Agent, Crew, LLM, Process, Task
from config.settings import LLM_MODEL, OLLAMA_BASE_URL
from src.mcp.mcp_tools import (
    mcp_sqlite_search_knowledge,
    knowledge_base_search,
)


def get_llm():
    return LLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


def build_crew(user_query: str):
    llm = get_llm()

    research_agent = Agent(
        role="Knowledge Research Specialist",
        goal="Find the most relevant enterprise facts for the user query.",
        backstory=(
            "You use RAG search and SQLite knowledge lookup to gather only the"
            " facts needed for a concise enterprise answer."
        ),
        llm=llm,
        tools=[knowledge_base_search, mcp_sqlite_search_knowledge],
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )

    response_agent = Agent(
        role="Response Generation Expert",
        goal="Write a short grounded answer from the research output.",
        backstory=(
            "You answer clearly, stay within the retrieved evidence, and cite"
            " the source names at the end."
        ),
        llm=llm,
        tools=[],
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )

    evaluation_agent = Agent(
        role="Response Quality Evaluator",
        goal="Polish the final answer and add a compact quality assessment.",
        backstory=(
            "You finalize the answer for display and provide brief quality"
            " ratings for accuracy, completeness, clarity, and source grounding."
        ),
        llm=llm,
        tools=[],
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )

    research_task = Task(
        description=(
            "Question: '{query}'\n\n"
            "Use Knowledge_Base_Search first. Use MCP_SQLite_Search_Knowledge only"
            " if it can add useful policy details.\n"
            "Return exactly 3 short bullets covering:\n"
            "- annual leave entitlement\n"
            "- how to apply for leave\n"
            "- any conditions or exceptions\n"
            "Each bullet must include a source name."
        ),
        expected_output="Three short research bullets with source names.",
        agent=research_agent,
    )

    response_task = Task(
        description=(
            "Using only the research bullets, answer '{query}' in under 140 words.\n"
            "Include a short heading and a 'Sources' section.\n"
            "Do not use tools. Return only the answer text."
        ),
        expected_output="Short final answer with a Sources section.",
        agent=response_agent,
        context=[research_task],
    )

    evaluation_task = Task(
        description=(
            "Polish the drafted answer for display. Keep the answer grounded.\n"
            "Append:\n"
            "Quality Score: X/10\n"
            "Accuracy: X/10\n"
            "Completeness: X/10\n"
            "Clarity: X/10\n"
            "Source Grounding: X/10"
        ),
        expected_output="Polished answer followed by a compact quality score block.",
        agent=evaluation_agent,
        context=[response_task],
    )

    return Crew(
        agents=[research_agent, response_agent, evaluation_agent],
        tasks=[research_task, response_task, evaluation_task],
        process=Process.sequential,
        verbose=False,
    )