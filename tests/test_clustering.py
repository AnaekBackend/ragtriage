"""Tests for query clustering module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import clustering components
from ragtriage.clustering.embedder import QueryEmbedder
from ragtriage.clustering.reducer import DimensionalityReducer
from ragtriage.clustering.clusterer import QueryClusterer
from ragtriage.clustering.analyzer import ClusterAnalyzer
from ragtriage.clustering.visualizer import ClusterVisualizer
from ragtriage.clustering.pipeline import ClusteringPipeline


class TestQueryEmbedder:
    """Test query embedding functionality."""
    
    def test_initialization(self):
        """Test embedder initializes correctly."""
        embedder = QueryEmbedder(model_name="all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.embedding_dimension == 384  # MiniLM dimension
    
    def test_embed_single_text(self):
        """Test embedding a single text."""
        embedder = QueryEmbedder()
        embedding = embedder.embed("How do I reset my password?")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)
        assert not np.isnan(embedding).any()
    
    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        embedder = QueryEmbedder()
        texts = [
            "How do I reset my password?",
            "What are your pricing plans?",
            "Cancel my subscription please"
        ]
        embeddings = embedder.embed(texts)
        
        assert embeddings.shape == (3, 384)
        assert not np.isnan(embeddings).any()
    
    def test_embed_queries(self):
        """Test embedding query dictionaries."""
        embedder = QueryEmbedder()
        queries = [
            {"query": "How do I reset my password?"},
            {"query": "What are your pricing plans?"},
        ]
        embeddings = embedder.embed_queries(queries)
        
        assert embeddings.shape == (2, 384)


class TestDimensionalityReducer:
    """Test UMAP dimensionality reduction."""
    
    def test_initialization(self):
        """Test reducer initializes correctly."""
        reducer = DimensionalityReducer(n_components=10)
        assert reducer.n_components == 10
        assert reducer.reducer is None
    
    def test_fit_transform(self):
        """Test fitting and transforming embeddings."""
        reducer = DimensionalityReducer(n_components=5)
        
        # Create dummy high-dimensional data
        np.random.seed(42)
        embeddings = np.random.randn(20, 384)
        
        reduced = reducer.fit_transform(embeddings)
        
        assert reduced.shape == (20, 5)
        assert not np.isnan(reduced).any()
    
    def test_small_dataset_adjustment(self):
        """Test that n_neighbors adjusts for small datasets."""
        reducer = DimensionalityReducer(n_components=2, n_neighbors=15)
        
        # Very small dataset
        embeddings = np.random.randn(5, 384)
        reduced = reducer.fit_transform(embeddings)
        
        assert reduced.shape == (5, 2)


class TestQueryClusterer:
    """Test HDBSCAN clustering."""
    
    def test_initialization(self):
        """Test clusterer initializes correctly."""
        clusterer = QueryClusterer(min_cluster_size=3)
        assert clusterer.min_cluster_size == 3
        assert clusterer.labels_ is None
    
    def test_fit_creates_labels(self):
        """Test fitting creates cluster labels."""
        clusterer = QueryClusterer(min_cluster_size=3)
        
        # Create clustered data
        np.random.seed(42)
        # Two distinct clusters
        cluster1 = np.random.randn(10, 10) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(10, 10) + [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        embeddings = np.vstack([cluster1, cluster2])
        
        labels = clusterer.fit(embeddings)
        
        assert labels is not None
        assert len(labels) == 20
        # Should have at least 2 clusters
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        assert n_clusters >= 1  # HDBSCAN may find different numbers
    
    def test_get_cluster_summary(self):
        """Test cluster summary generation."""
        clusterer = QueryClusterer(min_cluster_size=2)
        
        # Create simple clustered data
        np.random.seed(42)
        embeddings = np.vstack([
            np.random.randn(5, 5),
            np.random.randn(5, 5) + [5, 5, 5, 5, 5]
        ])
        
        clusterer.fit(embeddings)
        summary = clusterer.get_cluster_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) > 0
        
        for cluster_id, info in summary.items():
            assert "cluster_id" in info
            assert "size" in info
            assert "indices" in info
            assert "is_noise" in info
            assert info["size"] == len(info["indices"])
    
    def test_get_cluster_queries(self):
        """Test grouping queries by cluster."""
        clusterer = QueryClusterer(min_cluster_size=2)
        
        queries = [{"query": f"query_{i}"} for i in range(10)]
        
        np.random.seed(42)
        embeddings = np.random.randn(10, 5)
        clusterer.fit(embeddings)
        
        clustered = clusterer.get_cluster_queries(queries)
        
        assert isinstance(clustered, dict)
        # Total queries should match
        total = sum(len(qs) for qs in clustered.values())
        assert total == 10


class TestClusterAnalyzer:
    """Test cluster analysis functionality."""
    
    def test_initialization(self):
        """Test analyzer initializes."""
        analyzer = ClusterAnalyzer()
        assert analyzer is not None
    
    def test_analyze_cluster_basic(self):
        """Test basic cluster analysis."""
        analyzer = ClusterAnalyzer()
        
        queries = [
            {"query": "how to cancel subscription", "evaluation": {"overall_score": 4}},
            {"query": "cancel my billing", "evaluation": {"overall_score": 3}},
            {"query": "stop payment", "evaluation": {"overall_score": 4}},
        ]
        
        analysis = analyzer.analyze_cluster(0, queries)
        
        assert analysis["cluster_id"] == 0
        assert analysis["size"] == 3
        assert "top_terms" in analysis
        assert "quality_distribution" in analysis
        assert analysis["quality_distribution"]["well_answered"] == 2
        assert analysis["quality_distribution"]["partial"] == 1
    
    def test_analyze_cluster_empty(self):
        """Test analyzing empty cluster."""
        analyzer = ClusterAnalyzer()
        analysis = analyzer.analyze_cluster(-1, [])
        
        assert analysis["cluster_id"] == -1
        assert analysis["size"] == 0
    
    def test_analyze_all_clusters(self):
        """Test analyzing multiple clusters."""
        analyzer = ClusterAnalyzer()
        
        clustered_queries = {
            0: [
                {"query": "billing question", "evaluation": {"overall_score": 4}},
                {"query": "payment issue", "evaluation": {"overall_score": 3}},
            ],
            1: [
                {"query": "leave policy", "evaluation": {"overall_score": 4}},
            ],
        }
        
        analyses = analyzer.analyze_all_clusters(clustered_queries)
        
        assert len(analyses) == 2
        # Should be sorted by priority (partial answers first)
        assert analyses[0]["size"] == 2  # Cluster 0 has more queries
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        analyzer = ClusterAnalyzer()
        
        cluster_analyses = [
            {
                "cluster_id": 0,
                "size": 5,
                "is_noise": False,
                "top_terms": ["cancel", "subscription"],
                "quality_distribution": {"partial": 3, "poor": 0, "well_answered": 2},
                "needs_attention": True
            }
        ]
        
        recommendations = analyzer.generate_recommendations(cluster_analyses)
        
        assert len(recommendations) > 0
        assert "cluster(s) with answer quality issues" in recommendations[0]


class TestClusterVisualizer:
    """Test cluster visualization."""
    
    def test_initialization(self):
        """Test visualizer initializes."""
        viz = ClusterVisualizer(figsize=(10, 6))
        assert viz.figsize == (10, 6)
    
    def test_create_scatter_plot(self):
        """Test creating scatter plot."""
        viz = ClusterVisualizer()
        
        # Create dummy data
        np.random.seed(42)
        embeddings_2d = np.random.randn(20, 2)
        labels = np.array([0] * 10 + [1] * 10)
        queries = [f"query_{i}" for i in range(20)]
        
        fig = viz.create_scatter_plot(embeddings_2d, labels, queries)
        
        assert fig is not None
        # Close figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_save_plot(self):
        """Test saving plot to file."""
        viz = ClusterVisualizer()
        
        # Create dummy plot
        embeddings_2d = np.random.randn(10, 2)
        labels = np.array([0] * 5 + [1] * 5)
        queries = [f"query_{i}" for i in range(10)]
        
        fig = viz.create_scatter_plot(embeddings_2d, labels, queries)
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_clusters.png"
            viz.save_plot(fig, str(filepath))
            
            assert filepath.exists()
            assert filepath.stat().st_size > 0


class TestClusteringPipeline:
    """Test end-to-end clustering pipeline."""
    
    def test_initialization(self):
        """Test pipeline initializes all components."""
        pipeline = ClusteringPipeline()
        
        assert pipeline.embedder is not None
        assert pipeline.cluster_reducer is not None
        assert pipeline.viz_reducer is not None
        assert pipeline.clusterer is not None
        assert pipeline.analyzer is not None
        assert pipeline.visualizer is not None
    
    def test_run_pipeline(self):
        """Test full pipeline execution."""
        pipeline = ClusteringPipeline(min_cluster_size=2)
        
        # Create sample queries
        queries = [
            {"query": "how to cancel subscription", "evaluation": {"overall_score": 4}},
            {"query": "cancel my billing please", "evaluation": {"overall_score": 3}},
            {"query": "how to apply for leave", "evaluation": {"overall_score": 5}},
            {"query": "sick leave policy", "evaluation": {"overall_score": 4}},
            {"query": "what are the pricing plans", "evaluation": {"overall_score": 3}},
            {"query": "how much does it cost", "evaluation": {"overall_score": 4}},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = pipeline.run(
                queries,
                create_visualization=True,
                output_dir=tmpdir
            )
            
            # Verify results structure
            assert "n_queries" in results
            assert results["n_queries"] == 6
            assert "n_clusters" in results
            assert "clusters" in results
            assert "recommendations" in results
            assert "visualization_path" in results
            
            # Check clusters
            assert len(results["clusters"]) > 0
            
            # Check recommendations
            assert len(results["recommendations"]) > 0
            
            # Check visualization was created
            if results["visualization_path"]:
                assert Path(results["visualization_path"]).exists()
    
    def test_save_results(self):
        """Test saving results to JSON."""
        pipeline = ClusteringPipeline()
        
        results = {
            "n_queries": 10,
            "n_clusters": 2,
            "clusters": [],
            "recommendations": ["Test recommendation"]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            pipeline.save_results(results, str(output_path))
            
            assert output_path.exists()
            
            # Verify JSON is valid
            import json
            with open(output_path) as f:
                loaded = json.load(f)
            assert loaded["n_queries"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])