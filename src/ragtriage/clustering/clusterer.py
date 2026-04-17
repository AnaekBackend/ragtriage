"""Query clustering using HDBSCAN."""

import numpy as np
import hdbscan
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class QueryClusterer:
    """Clusters queries using HDBSCAN (finds natural groups, handles noise)."""
    
    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 1,
        metric: str = "euclidean"
    ):
        """
        Initialize HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum queries to form a cluster
            min_samples: Core point density threshold (higher = stricter)
            metric: Distance metric for clustering
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.clusterer = None
        self.labels_ = None
        
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN.
        
        Args:
            embeddings: Reduced embeddings (n_samples, n_features)
            
        Returns:
            Cluster labels (-1 for noise points)
        """
        n_samples = embeddings.shape[0]
        
        # Adjust parameters for small datasets
        adjusted_min_cluster_size = min(self.min_cluster_size, max(2, n_samples // 10))
        adjusted_min_samples = min(self.min_samples, adjusted_min_cluster_size - 1)
        
        logger.info(
            f"Clustering {n_samples} samples "
            f"(min_cluster_size={adjusted_min_cluster_size}, min_samples={adjusted_min_samples})..."
        )
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=adjusted_min_cluster_size,
            min_samples=adjusted_min_samples,
            metric=self.metric,
            cluster_selection_method="eom"  # Excess of Mass
        )
        
        self.labels_ = self.clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        
        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")
        
        return self.labels_
    
    def get_cluster_summary(self) -> Dict[int, Dict]:
        """
        Get summary statistics for each cluster.
        
        Returns:
            Dict mapping cluster_id -> {size, indices, is_noise}
        """
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit first.")
        
        summary = {}
        unique_labels = set(self.labels_)
        
        for label in unique_labels:
            indices = np.where(self.labels_ == label)[0].tolist()
            is_noise = (label == -1)
            
            summary[label] = {
                "cluster_id": label,
                "size": len(indices),
                "indices": indices,
                "is_noise": is_noise,
                "percentage": len(indices) / len(self.labels_) * 100
            }
        
        return summary
    
    def get_cluster_queries(self, queries: List[dict]) -> Dict[int, List[dict]]:
        """
        Group queries by their cluster assignment.
        
        Args:
            queries: List of query dictionaries
            
        Returns:
            Dict mapping cluster_id -> list of queries in that cluster
        """
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit first.")
        
        clustered = {}
        for idx, label in enumerate(self.labels_):
            if label not in clustered:
                clustered[label] = []
            clustered[label].append(queries[idx])
        
        return clustered