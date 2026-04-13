"""
RAG Pipeline orchestrator — ties ingestion, retrieval, reranking, and generation together.
"""

import logging
from openai import OpenAI
from typing import Tuple, List

from ingestion import process_pdfs
from vectorstore import VectorStore
from generator import generate_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for insurance policy Q&A.
    
    Usage:
        pipeline = RAGPipeline(api_key="sk-...", pdf_directory="./pdfs")
        pipeline.ingest()
        answer, sources = pipeline.ask("What is the hospitalization coverage?")
    """
    
    def __init__(self, api_key: str, pdf_directory: str, persist_path: str = "./chroma_db"):
        self.api_key = api_key
        self.pdf_directory = pdf_directory
        self.openai_client = OpenAI(api_key=api_key)
        self.vectorstore = VectorStore(api_key=api_key, persist_path=persist_path)
        self.is_ingested = False
        logger.info("RAG Pipeline initialized")
    
    def ingest(self, force_recreate: bool = False):
        """
        Ingest PDFs: extract text, chunk, embed, and store in ChromaDB.
        Only re-embeds if collection is empty or force_recreate=True.
        """
        logger.info("Starting ingestion...")
        
        # Create collections
        self.vectorstore.create_collection(force_recreate=force_recreate)
        self.vectorstore.create_cache_collection(force_recreate=force_recreate)
        
        # Check if already ingested
        if self.vectorstore.collection.count() > 0 and not force_recreate:
            logger.info(f"Collection already has {self.vectorstore.collection.count()} documents. Skipping ingestion.")
            self.is_ingested = True
            return
        
        # Process PDFs
        documents, metadatas = process_pdfs(self.pdf_directory)
        
        if not documents:
            logger.warning("No documents extracted. Check PDF directory.")
            return
        
        # Embed and store
        self.vectorstore.add_documents(documents, metadatas)
        self.is_ingested = True
        logger.info("Ingestion complete")
    
    def ask(self, question: str, use_cache: bool = True) -> Tuple[str, List[str]]:
        """
        Ask a question and get a cited answer.
        
        Args:
            question: User question
            use_cache: Whether to check cache first
        
        Returns:
            Tuple of (answer_text, list_of_sources)
        """
        if not self.is_ingested:
            return "Pipeline not ready. Call pipeline.ingest() first.", []
        
        logger.info(f"Question: {question}")
        
        # Retrieve
        if use_cache:
            results, cache_hit = self.vectorstore.search_with_cache(question)
        else:
            results = self.vectorstore.search(question)
            cache_hit = False
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        # Generate answer
        answer, sources = generate_response(
            self.openai_client, question, documents, metadatas
        )
        
        logger.info(f"Cache hit: {cache_hit} | Sources: {sources}")
        return answer, sources


# ============================================
# CLI usage
# ============================================
if __name__ == "__main__":
    import os
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    
    pdf_dir = input("Enter path to PDF directory: ").strip()
    
    # Initialize and ingest
    pipeline = RAGPipeline(
        api_key=api_key,
        pdf_directory=pdf_dir
    )
    pipeline.ingest()
    
    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("Insurance Policy Advisor — type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            break
        
        answer, sources = pipeline.ask(question)
        print(f"\n{answer}")
        print(f"\nSources: {sources}")
