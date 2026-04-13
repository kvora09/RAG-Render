"""
Answer generation using GPT-4o-mini with engineered prompts.
"""

import logging
from openai import OpenAI
from typing import List, Dict, Tuple

from config import GENERATION_MODEL, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE, SYSTEM_PROMPT
from reranker import rerank_with_llm

logger = logging.getLogger(__name__)


def generate_response(
    client: OpenAI,
    question: str,
    documents: List[str],
    metadatas: List[Dict],
) -> Tuple[str, List[str]]:
    """
    Generate a cited answer from retrieved documents.
    
    Args:
        client: OpenAI client
        question: User question
        documents: Retrieved documents
        metadatas: Document metadata
    
    Returns:
        Tuple of (answer_text, list_of_sources)
    """
    # Rerank to get top 3
    top_indices = rerank_with_llm(client, question, documents)
    top_docs = [documents[i] for i in top_indices]
    top_meta = [metadatas[i] for i in top_indices]
    
    # Build context with source labels
    context = ""
    source_info = []
    for idx, (doc, meta) in enumerate(zip(top_docs, top_meta)):
        if isinstance(meta, str):
            import ast
            meta = ast.literal_eval(meta)
        
        page = meta.get('Page_No.', meta.get('Page_No', 'Unknown'))
        policy = meta.get('Policy_Name', 'Unknown')
        source = f"{policy} | {page}"
        source_info.append(source)
        context += f"\n[Source {idx+1}: {source}]\n{doc}\n"
    
    # Generate answer
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE
        )
        
        answer = response.choices[0].message.content
        logger.info(f"Generated answer for: {question[:50]}...")
        return answer, source_info
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error generating response: {str(e)}", source_info
