"""Analyze clusters to extract insights."""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Analyzes clusters to extract actionable insights."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set. Set it to use LLM-based cluster naming.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def extract_cluster_name(self, queries: List[str]) -> str:
        """
        Extract a semantic name for a cluster using LLM.
        
        Args:
            queries: List of queries in the cluster
            
        Returns:
            Short semantic cluster name (2-5 words)
        """
        if not queries:
            return "misc"
        
        # Sample up to 10 queries for naming
        sample_queries = queries[:10]
        queries_text = "\n".join(f"- {q}" for q in sample_queries)
        
        system_prompt = """You analyze user query clusters. Generate a short, clear name (2-5 words) describing what these queries are about.

Rules:
- Use 2-5 words maximum
- Be specific but concise  
- Use natural language, not keyword concatenation
- Good examples: "Slack integration setup", "Cancel subscription", "Time off requests", "Attendance tracking issues"
- Bad examples: "slack slack bot", "cancel cancel subscription" (repetition), "how do I" (too generic)

Respond with ONLY the cluster name, no quotes, no explanation."""

        user_prompt = f"Queries in this cluster:\n{queries_text}\n\nCluster name:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=20
            )
            
            name = response.choices[0].message.content.strip()
            # Clean up
            name = name.strip('"\'').strip()
            if len(name) > 60:
                name = name[:60]
            if len(name) < 3:
                return self._fallback_name(queries)
            return name
            
        except Exception as e:
            logger.warning(f"LLM naming failed: {e}, using fallback")
            return self._fallback_name(queries)
    
    def _fallback_name(self, queries: List[str]) -> str:
        """Fallback to key phrase extraction if LLM fails."""
        # Extract bigrams (2-word phrases) instead of single words
        from collections import Counter
        
        all_bigrams = []
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'to', 'of', 'and', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'it', 'this', 'that', 'have', 'has', 'had'}
        
        for query in queries:
            words = re.findall(r'\b[a-z]+\b', query.lower())
            words = [w for w in words if w not in stop_words and len(w) > 2]
            # Generate bigrams
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            all_bigrams.extend(bigrams)
        
        if all_bigrams:
            most_common = Counter(all_bigrams).most_common(1)[0][0]
            return most_common
        
        # Last resort: first few words of first query
        first_query = queries[0] if queries else "misc"
        words = first_query.split()[:3]
        return " ".join(words) if words else "misc"

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
