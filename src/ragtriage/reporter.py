"""Generate reports from analyzed results."""

import json
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd


class ReportGenerator:
    """Generate actionable reports for CS teams."""
    
    def generate_report(self, analyzed_results: List[Dict]) -> str:
        """Generate markdown report."""
        # Calculate statistics
        total = len(analyzed_results)
        
        # Lane distribution
        lanes = Counter(r.get("lane", "UNKNOWN") for r in analyzed_results)
        understanding = lanes.get("UNDERSTANDING", 0)
        incident = lanes.get("INCIDENT", 0)
        spam = lanes.get("SPAM", 0)
        
        # Understanding queries breakdown
        understanding_queries = [r for r in analyzed_results if r.get("lane") == "UNDERSTANDING"]
        well_answered = len([r for r in understanding_queries 
                           if r.get("evaluation", {}).get("bucket") == "well_answered"])
        partial = len([r for r in understanding_queries 
                      if r.get("evaluation", {}).get("bucket") == "partial"])
        
        # Action items (only from partial understanding queries)
        action_items = [r for r in analyzed_results 
                       if r.get("lane") == "UNDERSTANDING" 
                       and r.get("evaluation", {}).get("bucket") == "partial"]
        
        doc_write = len([r for r in action_items if r.get("action") == "DOC_WRITE"])
        doc_update = len([r for r in action_items if r.get("action") == "DOC_UPDATE"])
        
        # Group DOC_WRITE items by (category, topic) for better organization
        write_items = [r for r in action_items if r.get("action") == "DOC_WRITE"]
        write_by_topic = defaultdict(list)
        for item in write_items:
            key = (item.get("category", "GENERAL"), item.get("topic", "Unknown"))
            write_by_topic[key].append(item)

        # Sort by count (most questions first)
        sorted_write_topics = sorted(write_by_topic.items(), key=lambda x: -len(x[1]))
        
        # Calculate percentages safely
        understanding_pct = (understanding/total*100) if total > 0 else 0
        incident_pct = (incident/total*100) if total > 0 else 0
        spam_pct = (spam/total*100) if total > 0 else 0
        well_pct = (well_answered/understanding*100) if understanding > 0 else 0
        partial_pct = (partial/understanding*100) if understanding > 0 else 0
        write_pct = (doc_write/len(action_items)*100) if len(action_items) > 0 else 0
        update_pct = (doc_update/len(action_items)*100) if len(action_items) > 0 else 0
        
        # Build report
        report = f"""# RAGTriage Analysis Report

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Queries | {total} | 100% |
| **UNDERSTANDING** | {understanding} | {understanding_pct:.1f}% |
| **INCIDENT** | {incident} | {incident_pct:.1f}% |
| **SPAM** | {spam} | {spam_pct:.1f}% |

### Understanding Queries Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| Well Answered | {well_answered} | {well_pct:.1f}% |
| Partial Answer | {partial} | {partial_pct:.1f}% |

### Action Items ({len(action_items)} total)

| Action | Count | Percentage |
|--------|-------|------------|
| Write New Article | {doc_write} | {write_pct:.1f}% |
| Update Existing | {doc_update} | {update_pct:.1f}% |

---

## Top Priority Articles

### Articles to Write ({doc_write})

"""
        
        # Articles to write - grouped by topic with content gaps
        for (category, topic), items in sorted_write_topics[:15]:
            # Aggregate gaps and sample queries
            gaps = [item.get("gap", "") for item in items if item.get("gap")]
            unique_gaps = list(dict.fromkeys(gaps))[:3]  # Deduplicate, max 3
            sample_queries = [item.get("query", "") for item in items[:2]]

            report += f"\n#### {topic.title()} ({len(items)} questions)\n\n"
            report += f"**Category:** {category}  \n"
            report += f"**Article Name:** {items[0].get('target_article', topic.title())}\n\n"

            report += "**Content to Cover:**\n"
            for gap in unique_gaps:
                report += f"- {gap}\n"

            if len(gaps) > 3:
                report += f"- *...and {len(gaps) - 3} more content areas*\n"

            report += "\n**Sample Questions:**\n"
            for query in sample_queries:
                short_query = query[:80] + "..." if len(query) > 80 else query
                report += f"- \"{short_query}\"\n"

            report += "\n---\n"
        
        report += f"\n### Articles to Update ({doc_update})\n\n"

        # Group DOC_UPDATE items by target_article
        update_items = [r for r in action_items if r.get("action") == "DOC_UPDATE"]
        update_by_article = defaultdict(list)
        for item in update_items:
            key = item.get("target_article", "Unknown")
            update_by_article[key].append(item)

        # Sort by count
        sorted_update_articles = sorted(update_by_article.items(), key=lambda x: -len(x[1]))

        for article, items in sorted_update_articles[:15]:
            # Aggregate gaps and categories
            gaps = [item.get("gap", "") for item in items if item.get("gap")]
            unique_gaps = list(dict.fromkeys(gaps))[:3]
            categories = list(set(item.get("category", "GENERAL") for item in items))
            sample_queries = [item.get("query", "") for item in items[:2]]

            report += f"\n#### {article} ({len(items)} questions)\n\n"
            report += f"**Category:** {', '.join(categories)}\n\n"

            report += "**Updates Needed:**\n"
            for gap in unique_gaps:
                report += f"- {gap}\n"

            if len(gaps) > 3:
                report += f"- *...and {len(gaps) - 3} more updates needed*\n"

            report += "\n**Sample Questions:**\n"
            for query in sample_queries:
                short_query = query[:80] + "..." if len(query) > 80 else query
                report += f"- \"{short_query}\"\n"

            report += "\n---\n"
        
        report += """
---

## How to Use This Report

1. **Start with DOC_WRITE items** - These represent missing documentation
2. **Review DOC_UPDATE items** - Enhance existing articles with missing info
3. **Monitor INCIDENT queries** - These may indicate product issues

*Generated by RAGTriage*
"""

        return report

    def generate_cluster_section(self, cluster_results: Dict) -> str:
        """Generate markdown section for cluster analysis."""
        if not cluster_results or not cluster_results.get('cluster_quality'):
            return ""

        section = """

---

## Query Pattern Analysis (Clusters)

Understanding the types of questions users ask helps prioritize documentation work.

"""

        cluster_quality = cluster_results['cluster_quality']
        cluster_names = cluster_results.get('cluster_names', {})

        # Sort by partial answers (most problematic first)
        sorted_clusters = sorted(
            cluster_quality.items(),
            key=lambda x: x[1].get('partial_answers', 0),
            reverse=True
        )

        for cluster_id, metrics in sorted_clusters:
            name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            count = metrics['query_count']
            quality = metrics['quality_pct']
            partial = metrics['partial_answers']

            section += f"""### {name.title()} ({count} queries)

- **Quality:** {quality:.0f}% well answered ({partial} need attention)
- **Average Score:** {metrics['avg_score']:.1f}/5

"""
            if partial > 0 and metrics.get('top_partial_queries'):
                section += "**Top Issues:**\n"
                for i, query in enumerate(metrics['top_partial_queries'][:3], 1):
                    short = query[:100] + "..." if len(query) > 100 else query
                    section += f"{i}. {short}\n"

                if metrics.get('recommended_actions'):
                    section += "\n**Recommended Actions:**\n"
                    for action, count in metrics['recommended_actions'].items():
                        section += f"- {action}: {count} queries\n"

            section += "\n"

        # Add priority ranking
        section += """### Priority Ranking

Focus on these clusters first (most issues to fix):

"""

        rank = 1
        for cluster_id, metrics in sorted_clusters[:5]:
            if metrics.get('partial_answers', 0) > 0:
                name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
                partial = metrics['partial_answers']
                section += f"{rank}. **{name.title()}** - {partial} queries need documentation work\n"
                rank += 1

        return section
    
    def generate_csv(self, analyzed_results: List[Dict]) -> pd.DataFrame:
        """Generate CSV of action items."""
        rows = []
        
        for item in analyzed_results:
            eval_data = item.get("evaluation", {})
            scores = eval_data.get("scores", {})
            
            row = {
                "query": item.get("query", ""),
                "lane": item.get("lane", ""),
                "category": item.get("category", ""),
                "topic": item.get("topic", ""),
                "action": item.get("action", ""),
                "target_article": item.get("target_article", ""),
                "gap": item.get("gap", ""),
                "reason": item.get("reason", ""),
                "overall_score": eval_data.get("overall_score", 0),
                "correctness": scores.get("correctness", 0),
                "completeness": scores.get("completeness", 0),
                "context_usage": scores.get("context_usage", 0),
                "clarity": scores.get("clarity", 0),
                "conciseness": scores.get("conciseness", 0),
                "bucket": eval_data.get("bucket", ""),
                "generated_answer": item.get("generated_answer", "")[:200],
            }
            
            # Add surface diagnostics if available
            diagnostics = item.get("surface_diagnostics", {})
            if diagnostics:
                coverage = diagnostics.get("coverage", {})
                relevance = diagnostics.get("context_relevance", {})
                contradictions = diagnostics.get("contradictions", {})
                diagnosis = diagnostics.get("overall_diagnosis", {})
                
                row["coverage_score"] = coverage.get("score")
                row["context_relevance"] = relevance.get("avg_relevance")
                row["contradiction_detected"] = contradictions.get("contradiction_detected")
                row["diagnostic_diagnosis"] = diagnosis.get("primary_issue")
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_diagnostics_section(self, analyzed_results: List[Dict]) -> str:
        """Generate markdown section for surface diagnostics."""
        # Find action items with diagnostics
        action_items = [r for r in analyzed_results 
                       if r.get("action") in ["DOC_WRITE", "DOC_UPDATE"]
                       and r.get("surface_diagnostics")]
        
        if not action_items:
            return ""
        
        section = """
---

## Surface Diagnostics (Evidence-Based Analysis)

Detailed analysis of partial answers to distinguish retrieval failures from generation issues.

"""
        
        for item in action_items:
            query = item.get("query", "")[:80]
            diagnostics = item.get("surface_diagnostics", {})
            
            coverage = diagnostics.get("coverage", {})
            relevance = diagnostics.get("context_relevance", {})
            contradictions = diagnostics.get("contradictions", {})
            diagnosis = diagnostics.get("overall_diagnosis", {})
            
            section += f"""### {query}...

**Action**: {item.get('action', 'N/A')} | **Article**: {item.get('target_article', 'N/A')}

| Signal | Value | Interpretation |
|--------|-------|----------------|
| Coverage Score | {coverage.get('score', 'N/A')} | {coverage.get('explanation', 'N/A')[:60]}... |
| Context Relevance | {relevance.get('avg_relevance', 'N/A')} | {relevance.get('explanation', 'N/A')[:60]}... |
| Contradiction | {'Yes' if contradictions.get('contradiction_detected') else 'No'} | {contradictions.get('explanation', 'N/A')[:60] if contradictions.get('contradiction_detected') else 'None detected'} |

**Diagnosis**: {diagnosis.get('explanation', 'N/A')}

"""
        
        return section
