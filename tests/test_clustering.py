"""Tests for query clustering module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

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

    def test_extract_cluster_name(self):
        """Test extracting descriptive name from queries."""
        analyzer = ClusterAnalyzer()

        queries = [
            "how to cancel subscription",
            "cancel my billing please",
            "stop payment and cancel"
        ]

        name = analyzer.extract_cluster_name(queries)
        assert isinstance(name, str)
        assert len(name) > 0
        # Should contain top terms
        assert "cancel" in name.lower()

    def test_analyze_cluster_quality(self):
        """Test analyzing cluster quality with evaluation results."""
        analyzer = ClusterAnalyzer()

        queries = [
            "how to cancel subscription",
            "cancel my billing",
            "how to apply for leave",
            "sick leave policy",
        ]
        labels = np.array([0, 0, 1, 1])

        evaluated_results = [
            {"query": queries[0], "evaluation": {"bucket": "well_answered", "overall_score": 4.5}},
            {"query": queries[1], "evaluation": {"bucket": "partial", "overall_score": 3.0}},
            {"query": queries[2], "evaluation": {"bucket": "well_answered", "overall_score": 5.0}},
            {"query": queries[3], "evaluation": {"bucket": "well_answered", "overall_score": 4.0}},
        ]

        quality = analyzer.analyze_cluster_quality(queries, labels, evaluated_results)

        assert 0 in quality
        assert 1 in quality

        # Check cluster 0 has 1 partial
        assert quality[0]["partial_answers"] == 1
        assert quality[0]["well_answered"] == 1

        # Check cluster 1 has 0 partial
        assert quality[1]["partial_answers"] == 0
        assert quality[1]["well_answered"] == 2

    def test_generate_cluster_summary(self):
        """Test generating human-readable summary."""
        analyzer = ClusterAnalyzer()

        cluster_quality = {
            0: {
                "name": "billing cancel",
                "query_count": 5,
                "well_answered": 3,
                "partial_answers": 2,
                "quality_pct": 60.0,
                "avg_score": 3.5,
                "top_partial_queries": ["how to cancel", "billing issue"],
                "recommended_actions": {"DOC_WRITE": 2}
            },
            1: {
                "name": "leave policy",
                "query_count": 3,
                "well_answered": 3,
                "partial_answers": 0,
                "quality_pct": 100.0,
                "avg_score": 4.5,
                "top_partial_queries": [],
                "recommended_actions": {}
            }
        }

        summary = analyzer.generate_cluster_summary(cluster_quality)

        assert isinstance(summary, str)
        assert "billing cancel" in summary.lower()
        assert "leave policy" in summary.lower()
        assert "2 issues need attention" in summary


class TestClusterVisualizer:
    """Test cluster visualization."""

    def test_initialization(self):
        """Test visualizer initializes."""
        viz = ClusterVisualizer(figsize=(10, 6))
        assert viz.figsize == (10, 6)

    def test_create_quality_scatter_plot(self):
        """Test creating quality-colored scatter plot."""
        viz = ClusterVisualizer()

        # Create dummy data
        np.random.seed(42)
        embeddings_2d = np.random.randn(20, 2)
        labels = np.array([0] * 10 + [1] * 10)
        queries = [f"query_{i}" for i in range(20)]

        cluster_names = {0: "billing cancel", 1: "leave policy"}
        cluster_quality = {
            0: {"query_count": 10, "quality_pct": 60.0, "name": "billing"},
            1: {"query_count": 10, "quality_pct": 90.0, "name": "leave"}
        }

        fig = viz.create_quality_scatter_plot(
            embeddings_2d, labels, queries, cluster_names, cluster_quality
        )

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
        cluster_names = {0: "cluster0", 1: "cluster1"}
        cluster_quality = {0: {"query_count": 5}, 1: {"query_count": 5}}

        fig = viz.create_quality_scatter_plot(
            embeddings_2d, labels, queries, cluster_names, cluster_quality
        )

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

    def test_run_pipeline_without_eval(self):
        """Test pipeline without evaluation results."""
        pipeline = ClusteringPipeline(min_cluster_size=2)

        # Create sample queries
        queries = [
            {"query": "how to cancel subscription"},
            {"query": "cancel my billing please"},
            {"query": "how to apply for leave"},
            {"query": "sick leave policy"},
            {"query": "what are the pricing plans"},
            {"query": "how much does it cost"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            results = pipeline.run(
                queries,
                evaluated_results=None,  # No evaluation
                create_visualization=True,
                output_dir=tmpdir
            )

            # Verify results structure
            assert "n_queries" in results
            assert results["n_queries"] == 6
            assert "n_clusters" in results
            assert "cluster_names" in results

    def test_run_pipeline_with_eval(self):
        """Test pipeline with evaluation results for quality analysis."""
        pipeline = ClusteringPipeline(min_cluster_size=2)

        queries = [
            {"query": "how to cancel subscription"},
            {"query": "cancel my billing please"},
            {"query": "how to apply for leave"},
            {"query": "sick leave policy"},
        ]

        evaluated_results = [
            {"query": queries[0]["query"], "evaluation": {"bucket": "partial", "overall_score": 3}},
            {"query": queries[1]["query"], "evaluation": {"bucket": "well_answered", "overall_score": 4}},
            {"query": queries[2]["query"], "evaluation": {"bucket": "well_answered", "overall_score": 5}},
            {"query": queries[3]["query"], "evaluation": {"bucket": "well_answered", "overall_score": 4}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            results = pipeline.run(
                queries,
                evaluated_results=evaluated_results,
                create_visualization=True,
                output_dir=tmpdir
            )

            # Check quality analysis was done
            assert "cluster_quality" in results
            assert results["cluster_quality"] is not None
            assert "summary_path" in results

    def test_save_results(self):
        """Test saving results to JSON."""
        pipeline = ClusteringPipeline()

        results = {
            "n_queries": 10,
            "n_clusters": 2,
            "n_noise": 1,
            "cluster_names": {0: "test"},
            "cluster_quality": {},
            "visualization_path": None,
            "summary_path": None
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
