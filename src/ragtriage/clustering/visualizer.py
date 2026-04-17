"""Visualize clusters in 2D."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """Creates 2D visualizations of query clusters."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size in inches (width, height)
        """
        self.figsize = figsize
    
    def create_scatter_plot(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        queries: List[str],
        title: str = "Query Clusters",
        sample_labels: int = 5
    ) -> plt.Figure:
        """
        Create 2D scatter plot of clusters.
        
        Args:
            embeddings_2d: 2D embeddings from UMAP
            labels: Cluster labels from HDBSCAN
            queries: Original query texts
            title: Plot title
            sample_labels: Number of cluster centers to label
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get unique labels
        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l != -1])
        
        # Color map
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))
        
        # Plot each cluster
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            color = colors[idx % len(colors)]
            
            if label == -1:
                # Noise points in gray
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c="gray",
                    alpha=0.3,
                    s=20,
                    label="Noise"
                )
            else:
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[color],
                    alpha=0.6,
                    s=50,
                    label=f"Cluster {label}"
                )
                
                # Add text annotation for cluster center
                if label < sample_labels:
                    center_x = embeddings_2d[mask, 0].mean()
                    center_y = embeddings_2d[mask, 1].mean()
                    
                    # Get a sample query for this cluster
                    sample_idx = np.where(mask)[0][0]
                    sample_query = queries[sample_idx][:30] + "..." if len(queries[sample_idx]) > 30 else queries[sample_idx]
                    
                    ax.annotate(
                        f"C{label}",
                        (center_x, center_y),
                        fontsize=9,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
                    )
        
        ax.set_xlabel("UMAP Dimension 1", fontsize=11)
        ax.set_ylabel("UMAP Dimension 2", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig: plt.Figure, filepath: str, dpi: int = 150):
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filepath: Output path (e.g., "clusters.png")
            dpi: Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved cluster visualization to {filepath}")
        plt.close(fig)