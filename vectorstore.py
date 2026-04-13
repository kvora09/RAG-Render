"""
Vector store: ChromaDB setup, embedding, retrieval, and caching.
"""

import logging
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import List, Dict, Optional, Tuple

from config import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    COLLECTION_NAME, CACHE_COLLECTION_NAME,
    RETRIEVAL_TOP_K, CACHE_THRESHOLD
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB collections for documents and cache."""
    
    def __init__(self, api_key: str, persist_path: str = "./chroma_db"):
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBEDDING_MODEL
        )
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = None
        self.cache_collection = None
        logger.info(f"VectorStore initialized with persist path: {persist_path}")
    
    def create_collection(self, force_recreate: bool = False):
        """Create or get the main document collection."""
        if force_recreate:
            try:
                self.client.delete_collection(COLLECTION_NAME)
                logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
            except Exception:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        logger.info(f"Collection '{COLLECTION_NAME}' ready. Count: {self.collection.count()}")
        return self.collection
    
    def create_cache_collection(self, force_recreate: bool = False):
        """Create or get the cache collection."""
        if force_recreate:
            try:
                self.client.delete_collection(CACHE_COLLECTION_NAME)
            except Exception:
                pass
        
        self.cache_collection = self.client.get_or_create_collection(
            name=CACHE_COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        logger.info(f"Cache collection '{CACHE_COLLECTION_NAME}' ready.")
        return self.cache_collection
    
    def add_documents(self, documents: List[str], metadatas: List[Dict]):
        """Add documents to the main collection in batches."""
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        if self.collection.count() > 0:
            logger.info(f"Collection already has {self.collection.count()} documents. Skipping.")
            return
        
        total = len(documents)
        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            batch_docs = documents[i:i + EMBEDDING_BATCH_SIZE]
            batch_ids = [str(j) for j in range(i, i + len(batch_docs))]
            batch_meta = metadatas[i:i + EMBEDDING_BATCH_SIZE]
            
            self.collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_meta
            )
            logger.debug(f"Added batch {i // EMBEDDING_BATCH_SIZE + 1}: {len(batch_docs)} docs")
        
        logger.info(f"Total documents in collection: {self.collection.count()}")
    
    def search(self, query: str, n_results: int = RETRIEVAL_TOP_K) -> Dict:
        """Search the main collection."""
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        results = self.collection.query(
            query_texts=query,
            n_results=n_results
        )
        return results
    
    def search_with_cache(self, query: str) -> Tuple[Dict, bool]:
        """
        Search with cache. Returns results and whether it was a cache hit.
        """
        cache_hit = False
        
        if self.cache_collection is not None:
            cache_results = self.cache_collection.query(
                query_texts=query,
                n_results=1
            )
            
            if (cache_results["distances"][0] != [] and 
                cache_results["distances"][0][0] <= CACHE_THRESHOLD):
                logger.info(f"Cache hit for query: {query[:50]}...")
                cache_hit = True
                return cache_results, cache_hit
        
        # Cache miss — search main collection
        results = self.search(query)
        logger.info(f"Cache miss for query: {query[:50]}...")
        
        # Store in cache
        if self.cache_collection is not None:
            try:
                keys, values = [], []
                for key, val in results.items():
                    if key != "embeddings" and val is not None and val[0] is not None:
                        for i in range(min(10, len(val[0]))):
                            keys.append(str(key) + str(i))
                            values.append(str(val[0][i]))
                
                self.cache_collection.add(
                    documents=[query],
                    ids=[query],
                    metadatas=[dict(zip(keys, values))]
                )
                logger.debug("Cached query results")
            except Exception as e:
                logger.warning(f"Failed to cache results: {e}")
        
        return results, cache_hit
