"""End-to-end clustering pipeline."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from .embedder import QueryEmbedder
from .reducer import DimensionalityReducer
from .clusterer import QueryClusterer
from .analyzer import ClusterAnalyzer
from .interactive_visualizer import InteractiveClusterVisualizer

logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """Complete query clustering pipeline."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_clusters_dims: int = 10,
        n_viz_dims: int = 2,
        min_cluster_size: int = 3,
        min_cluster_size_viz: int = 5
    ):
        """
        Initialize clustering pipeline.

        Args:
            embedding_model: Sentence transformer model name
            n_clusters_dims: Dimensions for clustering (UMAP)
            n_viz_dims: Dimensions for visualization (UMAP)
            min_cluster_size: Minimum queries per cluster for HDBSCAN
            min_cluster_size_viz: Minimum queries to show label in visualization
        """
        self.embedding_model = embedding_model
        self.n_clusters_dims = n_clusters_dims
        self.n_viz_dims = n_viz_dims
        self.min_cluster_size = min_cluster_size
        self.min_cluster_size_viz = min_cluster_size_viz

        # Initialize components
        self.embedder = QueryEmbedder(model_name=embedding_model)
        self.cluster_reducer = DimensionalityReducer(n_components=n_clusters_dims)
        self.viz_reducer = DimensionalityReducer(n_components=n_viz_dims)
        self.clusterer = QueryClusterer(min_cluster_size=min_cluster_size)
        self.analyzer = ClusterAnalyzer()
        self.visualizer = InteractiveClusterVisualizer()

    def run(
        self,
        queries: List[dict],
        evaluated_results: Optional[List[Dict]] = None,
        create_visualization: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run full clustering pipeline.

        Args:
            queries: List of query dicts with 'query' field
            evaluated_results: Optional evaluation results for quality analysis
            create_visualization: Whether to create 2D plot
            output_dir: Directory to save outputs

        Returns:
            Results dict with clusters, analyses, and recommendations
        """
        logger.info(f"Starting clustering pipeline for {len(queries)} queries...")

        # Step 1: Embed queries
        logger.info("Step 1: Embedding queries...")
        embeddings = self.embedder.embed_queries(queries)

        # Step 2: Reduce dimensions for clustering
        logger.info("Step 2: Reducing dimensions...")
        embeddings_cluster = self.cluster_reducer.fit_transform(embeddings)

        # Step 3: Cluster
        logger.info("Step 3: Clustering queries...")
        labels = self.clusterer.fit(embeddings_cluster)

        # Step 4: Extract cluster names
        query_texts = [q.get("query", "") for q in queries]
        unique_labels = sorted([l for l in set(labels) if l != -1])

        cluster_names = {}
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            cluster_queries = [query_texts[i] for i in indices]
            cluster_names[label] = self.analyzer.extract_cluster_name(cluster_queries)

        # Step 5: Analyze cluster quality (always generate basic stats, enhanced if evaluation results provided)
        logger.info("Step 4: Analyzing cluster quality...")
        cluster_quality = self.analyzer.analyze_cluster_quality(
            query_texts, labels, evaluated_results
        )

        # Step 6: Create visualization
        viz_path = None
        summary_path = None
        if create_visualization and output_dir:
            logger.info("Step 5: Creating interactive visualization...")
            embeddings_viz = self.viz_reducer.fit_transform(embeddings)

            if evaluated_results and len(evaluated_results) == len(queries):
                # Create interactive HTML plot with quality coloring
                html_content = self.visualizer.create_interactive_plot(
                    embeddings_viz,
                    labels,
                    query_texts,
                    cluster_names,
                    evaluated_results,
                    title="Query Clusters by Quality (Click legend to filter, hover for details)",
                    min_cluster_size_viz=self.min_cluster_size_viz
                )
            else:
                # Basic plot without quality data
                html_content = self.visualizer.create_interactive_plot(
                    embeddings_viz,
                    labels,
                    query_texts,
                    cluster_names,
                    [],
                    title="Query Clusters (Click legend to filter, hover for details)",
                    min_cluster_size_viz=self.min_cluster_size_viz
                )

            viz_path = Path(output_dir) / "cluster_visualization.html"
            self.visualizer.save_html(html_content, str(viz_path))

            # Also create treemap visualization (better for many clusters)
            logger.info("Creating treemap visualization...")
            treemap_html = self.visualizer.create_treemap(
                cluster_names,
                cluster_quality,
                evaluated_results if evaluated_results else [],
                title="Query Clusters - Treemap View (Sized by query count)"
            )
            treemap_path = Path(output_dir) / "cluster_treemap.html"
            self.visualizer.save_html(treemap_html, str(treemap_path))
            logger.info(f"Saved treemap to {treemap_path}")

            # Generate and save text summary (only if we have evaluation data)
            if evaluated_results:
                summary = self.analyzer.generate_cluster_summary(cluster_quality)
                summary_path = Path(output_dir) / "cluster_summary.txt"
                with open(summary_path, 'w') as f:
                    f.write(summary)
                logger.info(f"Saved cluster summary to {summary_path}")

        # Compile results
        n_clusters = len([l for l in set(labels) if l != -1])
        n_noise = sum(1 for l in labels if l == -1)

        results = {
            "n_queries": len(queries),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_names": cluster_names,
            "cluster_quality": cluster_quality,
            "visualization_path": str(viz_path) if viz_path else None,
            "treemap_path": str(treemap_path) if 'treemap_path' in locals() and treemap_path else None,
            "summary_path": str(summary_path) if summary_path else None
        }

        logger.info("Clustering pipeline complete!")
        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save clustering results to JSON file.

        Args:
            results: Results dict from run()
            output_path: Path to save JSON
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {convert_to_native(k): convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        results_native = convert_to_native(results)

        with open(output_path, 'w') as f:
            json.dump(results_native, f, indent=2)
        logger.info(f"Saved clustering results to {output_path}")
