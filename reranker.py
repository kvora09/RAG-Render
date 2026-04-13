"""
LLM-based re-ranking using GPT-4o-mini.
"""

import logging
from openai import OpenAI
from typing import List

from config import GENERATION_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)


def rerank_with_llm(
    client: OpenAI,
    query: str,
    documents: List[str],
    top_n: int = RERANK_TOP_N
) -> List[int]:
    """
    Re-rank documents using GPT-4o-mini.
    
    Args:
        client: OpenAI client
        query: User question
        documents: List of retrieved documents
        top_n: Number of top documents to return
    
    Returns:
        List of indices of the top-n most relevant documents
    """
    docs_text = ""
    for i, doc in enumerate(documents):
        docs_text += f"\n[Doc {i}]: {doc[:300]}\n"
    
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance ranker. Given a query and documents "
                        "(which may contain spelling errors), return ONLY the doc "
                        "numbers of the most relevant documents in order, as "
                        "comma-separated numbers. Example: 0,4,2"
                    )
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nDocuments:{docs_text}\n\nReturn the top {top_n} most relevant doc numbers:"
                }
            ],
            max_tokens=20,
            temperature=0
        )
        
        ranked = response.choices[0].message.content.strip().split(",")
        ranked_indices = [int(i.strip()) for i in ranked if i.strip().isdigit()]
        ranked_indices = [i for i in ranked_indices if i < len(documents)]
        
        if not ranked_indices:
            logger.warning("Reranker returned no valid indices. Falling back to original order.")
            return list(range(min(top_n, len(documents))))
        
        return ranked_indices[:top_n]
    
    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to original order.")
        return list(range(min(top_n, len(documents))))
