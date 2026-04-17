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
from .visualizer import ClusterVisualizer

logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """Complete query clustering pipeline."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_clusters_dims: int = 10,
        n_viz_dims: int = 2,
        min_cluster_size: int = 3
    ):
        """
        Initialize clustering pipeline.
        
        Args:
            embedding_model: Sentence transformer model name
            n_clusters_dims: Dimensions for clustering (UMAP)
            n_viz_dims: Dimensions for visualization (UMAP)
            min_cluster_size: Minimum queries per cluster
        """
        self.embedding_model = embedding_model
        self.n_clusters_dims = n_clusters_dims
        self.n_viz_dims = n_viz_dims
        self.min_cluster_size = min_cluster_size
        
        # Initialize components
        self.embedder = QueryEmbedder(model_name=embedding_model)
        self.cluster_reducer = DimensionalityReducer(n_components=n_clusters_dims)
        self.viz_reducer = DimensionalityReducer(n_components=n_viz_dims)
        self.clusterer = QueryClusterer(min_cluster_size=min_cluster_size)
        self.analyzer = ClusterAnalyzer()
        self.visualizer = ClusterVisualizer()
        
        # Store intermediate results
        self.embeddings = None
        self.embeddings_cluster = None
        self.embeddings_viz = None
        self.labels = None
        self.clustered_queries = None
        self.cluster_analyses = None
    
    def run(
        self,
        queries: List[dict],
        create_visualization: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run full clustering pipeline.
        
        Args:
            queries: List of query dicts with 'query' field
            create_visualization: Whether to create 2D plot
            output_dir: Directory to save outputs
            
        Returns:
            Results dict with clusters, analyses, and recommendations
        """
        logger.info(f"Starting clustering pipeline for {len(queries)} queries...")
        
        # Step 1: Embed queries
        logger.info("Step 1: Embedding queries...")
        self.embeddings = self.embedder.embed_queries(queries)
        
        # Step 2: Reduce dimensions for clustering
        logger.info("Step 2: Reducing dimensions for clustering...")
        self.embeddings_cluster = self.cluster_reducer.fit_transform(self.embeddings)
        
        # Step 3: Cluster
        logger.info("Step 3: Clustering queries...")
        self.labels = self.clusterer.fit(self.embeddings_cluster)
        
        # Step 4: Group queries by cluster
        self.clustered_queries = self.clusterer.get_cluster_queries(queries)
        
        # Step 5: Analyze clusters
        logger.info("Step 4: Analyzing clusters...")
        self.cluster_analyses = self.analyzer.analyze_all_clusters(self.clustered_queries)
        
        # Step 6: Generate recommendations
        logger.info("Step 5: Generating recommendations...")
        recommendations = self.analyzer.generate_recommendations(self.cluster_analyses)
        
        # Step 7: Create visualization (optional)
        viz_path = None
        if create_visualization:
            logger.info("Step 6: Creating visualization...")
            self.embeddings_viz = self.viz_reducer.fit_transform(self.embeddings)
            
            query_texts = [q.get("query", "") for q in queries]
            fig = self.visualizer.create_scatter_plot(
                self.embeddings_viz,
                self.labels,
                query_texts,
                title=f"Query Clusters (n={len(queries)})"
            )
            
            if output_dir:
                viz_path = Path(output_dir) / "cluster_visualization.png"
                self.visualizer.save_plot(fig, str(viz_path))
        
        # Compile results
        results = {
            "n_queries": len(queries),
            "n_clusters": len([c for c in self.cluster_analyses if not c.get("is_noise", False)]),
            "n_noise": len([c for c in self.cluster_analyses if c.get("is_noise", False)]),
            "clusters": self.cluster_analyses,
            "recommendations": recommendations,
            "visualization_path": str(viz_path) if viz_path else None
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
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results_native = convert_to_native(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_native, f, indent=2)
        logger.info(f"Saved clustering results to {output_path}")