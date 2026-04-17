"""Query embedding using sentence transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class QueryEmbedder:
    """Embeds text queries using lightweight sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Sentence transformer model name
                       Default: all-MiniLM-L6-v2 (80MB, fast, good quality)
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded. Embedding dimension: {self.model.get_embedding_dimension()}")
    
    def embed(self, texts: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Embed text(s) into vector space.
        
        Args:
            texts: Single text or list of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_queries(self, queries: List[dict], show_progress: bool = True) -> np.ndarray:
        """
        Embed queries from evaluation results.
        
        Args:
            queries: List of query dicts with 'query' field
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        texts = [q.get("query", "") for q in queries]
        return self.embed(texts, show_progress=show_progress)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_embedding_dimension()