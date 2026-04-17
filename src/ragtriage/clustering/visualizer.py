"""Visualize clusters in 2D with meaningful labels."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """Creates 2D visualizations of query clusters."""

    def __init__(self, figsize: tuple = (14, 10)):
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
        title: str = "Query Clusters"
    ):
        """Create basic scatter plot (when no quality data available)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        unique_labels = sorted([l for l in set(labels) if l != -1])
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            color = colors[idx % len(colors)]
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                alpha=0.6,
                s=50,
                label=f"Cluster {label}"
            )

        # Plot noise
        noise_mask = labels == -1
        if np.any(noise_mask):
            ax.scatter(
                embeddings_2d[noise_mask, 0],
                embeddings_2d[noise_mask, 1],
                c="gray",
                alpha=0.3,
                s=20,
                label="Noise"
            )

        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_quality_scatter_plot(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        queries: List[str],
        cluster_names: Dict[int, str],
        evaluated_results: List[Dict],
        title: str = "Query Clusters (colored by quality)"
    ) -> plt.Figure:
        """
        Create 2D scatter plot colored by answer quality.

        Args:
            embeddings_2d: 2D embeddings from UMAP
            labels: Cluster labels from HDBSCAN
            queries: Original query texts
            cluster_names: Mapping of cluster_id to descriptive name
            evaluated_results: List of evaluation results for each query
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get unique labels (excluding noise)
        unique_labels = sorted([l for l in set(labels) if l != -1])

        # Color each point based on its evaluation
        for label in unique_labels:
            mask = labels == label
            cluster_query_indices = np.where(mask)[0]

            if len(cluster_query_indices) == 0:
                continue

            # Get quality for each point in cluster
            colors = []
            for idx in cluster_query_indices:
                # Color by quality: red = partial, green = well answered
                if idx < len(evaluated_results):
                    bucket = evaluated_results[idx].get("evaluation", {}).get("bucket", "partial")
                    if bucket == "well_answered":
                        colors.append("#2ecc71")  # Green
                    else:
                        colors.append("#e74c3c")  # Red
                else:
                    colors.append("#95a5a6")  # Gray (unknown)

            # Plot cluster points
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors,
                alpha=0.6,
                s=80,
                edgecolors="white",
                linewidth=0.5
            )

            # Add cluster name annotation
            center_x = embeddings_2d[mask, 0].mean()
            center_y = embeddings_2d[mask, 1].mean()
            name = cluster_names.get(label, f"Cluster {label}")
            short_name = name[:25] + "..." if len(name) > 25 else name

            # Calculate cluster stats from evaluated_results
            cluster_indices = np.where(mask)[0]
            total_in_cluster = len(cluster_indices)
            well_answered_count = sum(
                1 for idx in cluster_indices
                if idx < len(evaluated_results)
                and evaluated_results[idx].get("evaluation", {}).get("bucket") == "well_answered"
            )
            quality_pct = (well_answered_count / total_in_cluster * 100) if total_in_cluster > 0 else 0

            ax.annotate(
                f"{short_name}\n({total_in_cluster} queries, {quality_pct:.0f}% good)",
                (center_x, center_y),
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
                fontweight="bold" if quality_pct < 70 else "normal"
            )

        # Plot noise points in gray
        noise_mask = labels == -1
        if np.any(noise_mask):
            ax.scatter(
                embeddings_2d[noise_mask, 0],
                embeddings_2d[noise_mask, 1],
                c="lightgray",
                alpha=0.3,
                s=30,
                label="Misc/Noise"
            )

        # Add legend for quality colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="Well Answered"),
            Patch(facecolor="#e74c3c", label="Partial Answer (needs fix)"),
            Patch(facecolor="lightgray", label="Misc/Noise")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        ax.set_xlabel("UMAP Dimension 1", fontsize=11)
        ax.set_ylabel("UMAP Dimension 2", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add text box with instructions
        instructions = (
            "Red dots = Need documentation work\n"
            "Green dots = Answered well\n"
            "Cluster names show top keywords from queries"
        )
        ax.text(
            0.02, 0.98, instructions,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

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
