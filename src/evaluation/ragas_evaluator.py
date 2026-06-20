from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from config.settings import LLM_MODEL_RAW, EMBED_MODEL, OLLAMA_BASE_URL


def run_ragas_evaluation(
    question: str,
    answer: str,
    contexts: list,
    ground_truth: str = "",
):
    # Fallback ground truth if not provided
    if not ground_truth:
        ground_truth = answer

    data = {
        "question":     [question],
        "answer":       [answer],
        "contexts":     [contexts],
        "ground_truth": [ground_truth],
    }
    dataset = Dataset.from_dict(data)

    llm = ChatOllama(
        model=LLM_MODEL_RAW,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    emb = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
    )

    return result.to_pandas()