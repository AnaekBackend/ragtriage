"""Dimensionality reduction using UMAP or PCA for small datasets."""

import numpy as np
import umap
from sklearn.decomposition import PCA
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Reduces high-dimensional embeddings to lower dimensions using UMAP."""
    
    def __init__(
        self,
        n_components: int = 10,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        metric: str = "cosine"
    ):
        """
        Initialize UMAP reducer.
        
        Args:
            n_components: Target dimensions (10 for clustering, 2 for viz)
            n_neighbors: Local neighborhood size (larger = more global structure)
            min_dist: Minimum distance between points in low-dim space
            random_state: For reproducibility
            metric: Distance metric (cosine works well for text)
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.metric = metric
        self.reducer = None
        
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit dimensionality reduction and transform embeddings.
        Uses PCA for small datasets (<20 samples), UMAP for larger ones.
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, n_features)
            
        Returns:
            Reduced embeddings (n_samples, n_components)
        """
        n_samples = embeddings.shape[0]
        
        # For very small datasets, use PCA instead of UMAP
        if n_samples < 20:
            adjusted_components = min(self.n_components, n_samples - 1, 10)
            if adjusted_components < 1:
                adjusted_components = 1
            
            logger.warning(
                f"Dataset has only {n_samples} samples. "
                f"Using PCA to reduce to {adjusted_components} dimensions."
            )
            
            self.reducer = PCA(n_components=adjusted_components, random_state=self.random_state)
            reduced = self.reducer.fit_transform(embeddings)
            logger.info(f"PCA reduction complete. Shape: {reduced.shape}")
            return reduced
        
        # For larger datasets, use UMAP
        # Adjust n_components for small datasets
        adjusted_components = min(self.n_components, n_samples - 1)
        if adjusted_components < 1:
            adjusted_components = 1
        
        # Adjust n_neighbors for small datasets
        adjusted_neighbors = min(self.n_neighbors, n_samples - 1)
        if adjusted_neighbors < 2:
            adjusted_neighbors = 2
        
        if adjusted_components < self.n_components or adjusted_neighbors < self.n_neighbors:
            logger.warning(
                f"Dataset has only {n_samples} samples. "
                f"Adjusting to {adjusted_components} dimensions, {adjusted_neighbors} neighbors"
            )
        
        logger.info(f"Reducing {n_samples} embeddings to {adjusted_components} dimensions using UMAP...")
        
        self.reducer = umap.UMAP(
            n_components=adjusted_components,
            n_neighbors=adjusted_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            metric=self.metric,
            verbose=False
        )
        
        reduced = self.reducer.fit_transform(embeddings)
        logger.info(f"UMAP reduction complete. Shape: {reduced.shape}")
        
        return reduced
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted reducer."""
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform first.")
        return self.reducer.transform(embeddings)