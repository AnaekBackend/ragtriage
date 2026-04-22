"""Analyze clusters to extract insights."""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np


class ClusterAnalyzer:
    """Analyzes clusters to extract actionable insights."""

    # Common stop words to exclude from cluster names
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can",
        "this", "that", "these", "those", "i", "you", "he", "she",
        "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "its", "our", "their",
        "and", "or", "but", "if", "then", "else", "when", "where",
        "why", "how", "what", "which", "who", "whom", "whose"
    }

    def extract_cluster_name(self, queries: List[str], top_n: int = 3) -> str:
        """
        Extract a descriptive name for a cluster from its queries.

        Args:
            queries: List of queries in the cluster
            top_n: Number of top terms to include

        Returns:
            Cluster name (e.g., "billing subscription cancel")
        """
        # Combine all queries
        text = " ".join(queries).lower()

        # Extract words (remove punctuation)
        words = re.findall(r'\b[a-z]+\b', text)

        # Filter out stop words and short words
        words = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # Count frequencies
        word_counts = Counter(words)

        # Get top terms
        top_terms = [term for term, _ in word_counts.most_common(top_n)]

        # Join to form name
        return " ".join(top_terms) if top_terms else "misc"

    def analyze_cluster_quality(
        self,
        queries: List[str],
        labels: np.ndarray,
        evaluated_results: List[Dict] = None
    ) -> Dict[int, Dict]:
        """
        Analyze quality metrics for each cluster.

        Args:
            queries: List of query texts
            labels: Cluster labels
            evaluated_results: Evaluation results for each query (optional)

        Returns:
            Dict mapping cluster_id to quality metrics
        """
        cluster_quality = {}
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            # Get indices for this cluster
            indices = np.where(labels == label)[0]

            # Get queries for this cluster
            cluster_queries = [queries[i] for i in indices]

            # Calculate metrics
            total = len(cluster_queries)

            if evaluated_results and len(evaluated_results) >= len(queries):
                # Full quality analysis with evaluation data
                cluster_results = [evaluated_results[i] for i in indices]
                well_answered = sum(1 for r in cluster_results
                                   if r.get("evaluation", {}).get("bucket") == "well_answered")
                partial = sum(1 for r in cluster_results
                             if r.get("evaluation", {}).get("bucket") == "partial")
                scores = [r.get("evaluation", {}).get("overall_score", 0)
                         for r in cluster_results]
                avg_score = sum(scores) / len(scores) if scores else 0
                partial_queries = [r.get("query", "")
                                 for r in cluster_results
                                 if r.get("evaluation", {}).get("bucket") == "partial"]
                actions = Counter(r.get("action", "UNKNOWN")
                                for r in cluster_results
                                if r.get("evaluation", {}).get("bucket") == "partial")
                recommended_actions = dict(actions.most_common(3))
            else:
                # Basic analysis without evaluation data
                well_answered = 0
                partial = 0
                avg_score = 0
                partial_queries = []
                recommended_actions = {}

            cluster_quality[int(label)] = {
                "name": self.extract_cluster_name(cluster_queries),
                "query_count": len(cluster_queries),
                "well_answered": well_answered,
                "partial_answers": partial,
                "quality_pct": (well_answered / total * 100) if total > 0 else 0,
                "avg_score": avg_score,
                "top_partial_queries": partial_queries[:3],
                "recommended_actions": recommended_actions
            }

        return cluster_quality

    def generate_cluster_summary(
        self,
        cluster_quality: Dict[int, Dict],
        sort_by: str = "partial_answers"
    ) -> str:
        """
        Generate a human-readable summary of clusters.

        Args:
            cluster_quality: Quality metrics per cluster
            sort_by: Field to sort by (partial_answers, query_count, quality_pct)

        Returns:
            Formatted summary text
        """
        # Sort clusters by specified field
        sorted_clusters = sorted(
            cluster_quality.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )

        lines = [
            "=" * 70,
            "QUERY CLUSTER ANALYSIS",
            "=" * 70,
            f"\nFound {len(cluster_quality)} distinct question patterns\n",
        ]

        for cluster_id, metrics in sorted_clusters:
            name = metrics["name"]
            count = metrics["query_count"]
            quality = metrics["quality_pct"]
            partial = metrics["partial_answers"]
            avg_score = metrics["avg_score"]

            lines.extend([
                f"\n{'─' * 70}",
                f"Cluster {cluster_id}: {name.upper()}",
                f"{'─' * 70}",
                f"  Queries: {count}",
                f"  Quality: {quality:.1f}% well answered",
                f"  Issues: {partial} partial answers",
                f"  Avg Score: {avg_score:.1f}/5",
            ])

            if partial > 0:
                lines.append(f"\n  Top Issues:")
                for i, query in enumerate(metrics["top_partial_queries"][:3], 1):
                    short_query = query[:80] + "..." if len(query) > 80 else query
                    lines.append(f"    {i}. {short_query}")

            if metrics["recommended_actions"]:
                lines.append(f"\n  Recommended Actions:")
                for action, count in metrics["recommended_actions"].items():
                    lines.append(f"    • {action}: {count} queries")

        lines.extend([
            "\n" + "=" * 70,
            "PRIORITY RANKING",
            "=" * 70,
            "\nFocus on clusters with most partial answers first:\n"
        ])

        for rank, (cluster_id, metrics) in enumerate(sorted_clusters[:5], 1):
            name = metrics["name"]
            partial = metrics["partial_answers"]
            if partial > 0:
                lines.append(f"  {rank}. {name} ({partial} issues need attention)")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)
