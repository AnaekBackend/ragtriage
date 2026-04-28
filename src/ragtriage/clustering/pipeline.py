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
        output_dir: Optional[str] = None,
        filter_issues_only: bool = True,
        use_actionable_grouping: bool = False
    ) -> Dict[str, Any]:
        """
        Run full clustering pipeline.

        Args:
            queries: List of query dicts with 'query' field
            evaluated_results: Optional evaluation results for quality analysis
            create_visualization: Whether to create 2D plot
            output_dir: Directory to save outputs
            filter_issues_only: If True and evaluated_results provided, only cluster 
                               queries with partial answers or content gaps
            use_actionable_grouping: If True, group by Category→Topic→Action instead of semantic clustering

        Returns:
            Results dict with clusters, analyses, and recommendations
        """
        # Option D: Use actionable grouping instead of semantic clustering
        if use_actionable_grouping and evaluated_results:
            return self._run_actionable_grouping(
                queries, evaluated_results, create_visualization, output_dir
            )
        
        # Standard semantic clustering path
        # Filter to only problematic queries if evaluation data available
        if filter_issues_only and evaluated_results:
            issue_indices = []
            for i, result in enumerate(evaluated_results):
                bucket = result.get("evaluation", {}).get("bucket", "")
                if bucket in ["partial", "content_gap"]:
                    issue_indices.append(i)
            
            if issue_indices:
                original_count = len(queries)
                queries = [queries[i] for i in issue_indices]
                evaluated_results = [evaluated_results[i] for i in issue_indices]
                logger.info(f"Filtered to {len(queries)} queries with issues (from {original_count} total)")
            else:
                logger.info("No queries with issues found, clustering all queries")
        
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
        treemap_path = None
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
            "treemap_path": str(treemap_path) if treemap_path else None,
            "summary_path": str(summary_path) if summary_path else None
        }

        logger.info("Clustering pipeline complete!")
        return results

    def _run_actionable_grouping(
        self,
        queries: List[dict],
        evaluated_results: List[Dict],
        create_visualization: bool,
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Option D: Group by Category → Topic → Action instead of semantic clustering.
        
        This creates actionable groupings that map directly to documentation work.
        """
        logger.info("Starting actionable grouping (Option D)...")
        
        # Filter to UNDERSTANDING lane + partial answers only
        actionable_items = []
        for i, result in enumerate(evaluated_results):
            lane = result.get("lane", "")
            action = result.get("action", "")
            bucket = result.get("evaluation", {}).get("bucket", "")
            
            if lane == "UNDERSTANDING" and bucket == "partial" and action in ["DOC_WRITE", "DOC_UPDATE"]:
                actionable_items.append({
                    "index": i,
                    "query": result.get("query", ""),
                    "category": result.get("category", "GENERAL"),
                    "topic": result.get("topic", "unknown"),
                    "action": action,
                    "target_article": result.get("target_article", "Unknown"),
                    "gap": result.get("gap", ""),
                    "evaluation": result.get("evaluation", {})
                })
        
        if not actionable_items:
            logger.warning("No actionable items found (UNDERSTANDING + partial + DOC_WRITE/DOC_UPDATE)")
            return {
                "n_queries": 0,
                "n_clusters": 0,
                "n_noise": 0,
                "cluster_names": {},
                "cluster_quality": {},
                "visualization_path": None,
                "treemap_path": None,
                "summary_path": None
            }
        
        logger.info(f"Found {len(actionable_items)} actionable items to group")
        
        # Group by Category → Topic → Action
        # Structure: {category: {topic: {action: [items]}}}
        hierarchy = {}
        for item in actionable_items:
            cat = item["category"]
            topic = item["topic"]
            action = item["action"]
            
            if cat not in hierarchy:
                hierarchy[cat] = {}
            if topic not in hierarchy[cat]:
                hierarchy[cat][topic] = {"DOC_WRITE": [], "DOC_UPDATE": []}
            hierarchy[cat][topic][action].append(item)
        
        # Create cluster-like structure for compatibility
        cluster_names = {}
        cluster_quality = {}
        cluster_id = 0
        
        for category, topics in sorted(hierarchy.items()):
            for topic, actions in sorted(topics.items()):
                for action, items in actions.items():
                    if not items:
                        continue
                    
                    # Create cluster name: "Category: Topic (Action)"
                    cluster_name = f"{category}: {topic}"
                    cluster_names[cluster_id] = cluster_name
                    
                    cluster_quality[cluster_id] = {
                        "name": cluster_name,
                        "query_count": len(items),
                        "well_answered": 0,
                        "partial_answers": len(items),
                        "quality_pct": 0,
                        "avg_score": 0,
                        "top_partial_queries": [item["query"] for item in items[:3]],
                        "recommended_actions": {action: len(items)},
                        "target_article": items[0]["target_article"] if items else "Unknown",
                        "gap": items[0]["gap"] if items else "",
                        "category": category,
                        "topic": topic,
                        "action": action,
                        "items": items  # Store all items for drill-down
                    }
                    cluster_id += 1
        
        logger.info(f"Created {len(cluster_names)} actionable groups")
        
        # Create visualizations
        viz_path = None
        treemap_path = None
        
        if create_visualization and output_dir:
            # Create actionable treemap (no UMAP scatter for this view)
            logger.info("Creating actionable treemap visualization...")
            treemap_html = self.visualizer.create_actionable_treemap(
                hierarchy,
                title="Actionable Items by Category → Topic → Action"
            )
            treemap_path = Path(output_dir) / "actionable_treemap.html"
            self.visualizer.save_html(treemap_html, str(treemap_path))
            logger.info(f"Saved actionable treemap to {treemap_path}")
        
        results = {
            "n_queries": len(actionable_items),
            "n_clusters": len(cluster_names),
            "n_noise": 0,
            "cluster_names": cluster_names,
            "cluster_quality": cluster_quality,
            "visualization_path": str(viz_path) if viz_path else None,
            "treemap_path": str(treemap_path) if treemap_path else None,
            "summary_path": None,
            "hierarchy": hierarchy  # Include raw hierarchy for drill-down
        }
        
        logger.info("Actionable grouping complete!")
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
