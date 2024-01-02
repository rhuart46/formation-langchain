"""
Utility class allowing to remove duplicate embeddings when retrieving them.
Its instances are meant to be substitutes for Chroma().as_retriever().
"""
from typing import Optional, List, Dict, Any

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.chroma import Chroma
from langchain_core.callbacks import Callbacks


class RedundantFilterRetriever(BaseRetriever):
    embedder: Embeddings
    chroma: Chroma

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        # Calculate the embedding for the "query" string:
        emb = self.embedder.embed_query(query)

        # Use max_marginal_relevance_search_by_vector with this embedding.
        # Its lambda_mult parameter controls the diversity of results
        # (between 0 and 1, the lower it is, the more diversity we get):
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8
        )

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return []
