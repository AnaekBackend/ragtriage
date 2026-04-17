"""Analyze cluster characteristics and priorities."""

from typing import Dict, List, Any
from collections import Counter
import re
import logging

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Analyzes clusters to extract insights and priorities."""
    
    def __init__(self):
        """Initialize analyzer."""
        pass
    
    def analyze_cluster(
        self,
        cluster_id: int,
        queries: List[dict],
        top_terms: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze a single cluster.
        
        Args:
            cluster_id: Cluster identifier
            queries: List of queries in this cluster
            top_terms: Number of top terms to extract
            
        Returns:
            Analysis dict with insights
        """
        if not queries:
            return {"cluster_id": cluster_id, "size": 0}
        
        # Extract text
        texts = [q.get("query", "").lower() for q in queries]
        
        # Get top terms
        all_words = []
        for text in texts:
            # Simple tokenization (could use better NLP)
            words = re.findall(r'\b[a-z]{3,}\b', text)
            # Filter common stop words
            stop_words = {'how', 'what', 'why', 'when', 'where', 'can', 'the', 'and', 'for', 'are', 'with', 'have', 'from', 'that', 'this', 'but', 'not', 'you', 'all', 'any', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'way', 'she', 'her', 'him', 'his', 'how', 'its', 'may', 'say', 'she', 'too', 'old', 'tell', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'will', 'would', 'there', 'their', 'what', 'said', 'each', 'which', 'about', 'could', 'other', 'after', 'first', 'never', 'these', 'think', 'where', 'being', 'every', 'great', 'might', 'shall', 'still', 'those', 'while', 'this', 'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what'}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_terms)
        
        # Analyze answer quality
        total = len(queries)
        well_answered = sum(1 for q in queries if q.get("evaluation", {}).get("overall_score", 0) >= 4)
        partial = sum(1 for q in queries if 2 <= q.get("evaluation", {}).get("overall_score", 0) < 4)
        poor = sum(1 for q in queries if q.get("evaluation", {}).get("overall_score", 0) < 2)
        
        # Lane distribution
        lanes = Counter(q.get("lane", "unknown") for q in queries)
        
        # Action items
        action_counts = Counter(q.get("action", "unknown") for q in queries if q.get("action"))
        
        return {
            "cluster_id": cluster_id,
            "size": total,
            "is_noise": cluster_id == -1,
            "top_terms": [word for word, count in top_words],
            "sample_queries": texts[:3],  # First 3 as examples
            "quality_distribution": {
                "well_answered": well_answered,
                "partial": partial,
                "poor": poor,
                "percent_well": well_answered / total * 100 if total > 0 else 0
            },
            "lane_distribution": dict(lanes),
            "action_distribution": dict(action_counts),
            "needs_attention": partial > 0 or poor > 0
        }
    
    def analyze_all_clusters(
        self,
        clustered_queries: Dict[int, List[dict]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze all clusters and return sorted by priority.
        
        Priority order:
        1. Clusters with partial answers (need doc updates)
        2. Clusters with poor answers (need investigation)
        3. Larger clusters
        4. Smaller clusters
        
        Args:
            clustered_queries: Dict mapping cluster_id -> queries
            
        Returns:
            List of cluster analyses, sorted by priority
        """
        analyses = []
        
        for cluster_id, queries in clustered_queries.items():
            analysis = self.analyze_cluster(cluster_id, queries)
            analyses.append(analysis)
        
        # Sort by priority
        def priority_key(analysis):
            if analysis.get("is_noise", False):
                return (3, 0, 0)  # Noise last
            
            partial = analysis.get("quality_distribution", {}).get("partial", 0)
            poor = analysis.get("quality_distribution", {}).get("poor", 0)
            size = analysis.get("size", 0)
            
            # Priority: partial > 0, then poor > 0, then size descending
            has_issues = 0 if (partial > 0 or poor > 0) else 1
            issue_count = -(partial + poor)  # Negative for descending
            
            return (has_issues, issue_count, -size)
        
        analyses.sort(key=priority_key)
        return analyses
    
    def generate_recommendations(
        self,
        cluster_analyses: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate actionable recommendations from cluster analysis.
        
        Args:
            cluster_analyses: Analyzed clusters
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Identify high-priority clusters
        problem_clusters = [
            c for c in cluster_analyses 
            if c.get("needs_attention") and not c.get("is_noise", False)
        ]
        
        if not problem_clusters:
            recommendations.append("All query clusters are well-answered. No immediate action needed.")
            return recommendations
        
        recommendations.append(f"Found {len(problem_clusters)} cluster(s) with answer quality issues:\n")
        
        for cluster in problem_clusters[:5]:  # Top 5
            cluster_id = cluster["cluster_id"]
            size = cluster["size"]
            top_terms = ", ".join(cluster.get("top_terms", [])[:3])
            partial = cluster.get("quality_distribution", {}).get("partial", 0)
            poor = cluster.get("quality_distribution", {}).get("poor", 0)
            
            rec = f"  Cluster {cluster_id} ({size} queries): {top_terms}\n"
            rec += f"    - {partial} partial answers, {poor} poor answers\n"
            
            if partial > 0:
                rec += f"    → Review and update existing documentation\n"
            if poor > 0:
                rec += f"    → Investigate: potential content gaps or system issues\n"
            
            recommendations.append(rec)
        
        return recommendations