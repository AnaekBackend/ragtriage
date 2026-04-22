"""Interactive HTML visualization of clusters using Plotly."""

import numpy as np
from typing import List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)


class InteractiveClusterVisualizer:
    """Creates interactive HTML visualizations of query clusters."""

    def __init__(self, width: int = 1200, height: int = 800):
        """
        Initialize visualizer.

        Args:
            width: Plot width in pixels
            height: Plot height in pixels
        """
        self.width = width
        self.height = height

    def create_interactive_plot(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        queries: List[str],
        cluster_names: Dict[int, str],
        evaluated_results: List[Dict],
        title: str = "Query Clusters (Interactive)",
        min_cluster_size_viz: int = 5
    ) -> str:
        """
        Create interactive HTML plot.

        Args:
            embeddings_2d: 2D embeddings from UMAP
            labels: Cluster labels from HDBSCAN
            queries: Original query texts
            cluster_names: Mapping of cluster_id to descriptive name
            evaluated_results: List of evaluation results for each query
            title: Plot title

        Returns:
            HTML string for the interactive plot
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly not installed. Run: uv add plotly")
            raise

        # Get unique labels (excluding noise)
        unique_labels = sorted([l for l in set(labels) if l != -1])

        # Create figure
        fig = make_subplots(
            specs=[[{"type": "scatter"}]],
            subplot_titles=[title]
        )

        # Color mapping for quality
        color_map = {
            "well_answered": "#2ecc71",  # Green
            "partial": "#e74c3c",         # Red (partial answer)
            "content_gap": "#f39c12",     # Orange
            "unknown": "#95a5a6"          # Gray
        }

        # Track cluster statistics
        cluster_stats = {}

        # Add traces for each cluster
        annotations_placed = []  # Track annotation positions to avoid overlap
        
        for label in unique_labels:
            mask = labels == label
            cluster_indices = np.where(mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Get data for this cluster
            x_coords = embeddings_2d[mask, 0]
            y_coords = embeddings_2d[mask, 1]

            # Get colors and hover text for each point
            colors = []
            hover_texts = []
            quality_counts = {"well_answered": 0, "partial": 0, "content_gap": 0}

            for idx in cluster_indices:
                if idx < len(evaluated_results):
                    result = evaluated_results[idx]
                    bucket = result.get("evaluation", {}).get("bucket", "unknown")
                    score = result.get("dimensions", {}).get("overall", 0)

                    colors.append(color_map.get(bucket, "#95a5a6"))
                    quality_counts[bucket] = quality_counts.get(bucket, 0) + 1

                    # Create hover text
                    query_text = queries[idx] if idx < len(queries) else "Unknown"
                    hover_text = f"<b>Query:</b> {query_text[:100]}...<br>" if len(query_text) > 100 else f"<b>Query:</b> {query_text}<br>"
                    hover_text += f"<b>Score:</b> {score}/5<br>"
                    hover_text += f"<b>Status:</b> {bucket.replace('_', ' ').title()}<br>"
                    hover_text += f"<b>Cluster:</b> {cluster_names.get(label, f'Cluster {label}')}"
                    hover_texts.append(hover_text)
                else:
                    colors.append("#95a5a6")
                    hover_texts.append(f"Query {idx}<br>No evaluation data")

            # Calculate cluster statistics
            total = len(cluster_indices)
            well_answered = quality_counts.get("well_answered", 0)
            quality_pct = (well_answered / total * 100) if total > 0 else 0
            cluster_stats[label] = {
                "total": total,
                "well_answered": well_answered,
                "partial": quality_counts.get("partial", 0),
                "quality_pct": quality_pct
            }

            # Determine cluster color (based on majority)
            if quality_pct >= 70:
                cluster_color = "rgba(46, 204, 113, 0.3)"  # Green transparent
            elif quality_pct >= 40:
                cluster_color = "rgba(243, 156, 18, 0.3)"  # Orange transparent
            else:
                cluster_color = "rgba(231, 76, 60, 0.3)"   # Red transparent

            # Add scatter trace for this cluster
            cluster_name = cluster_names.get(label, f"Cluster {label}")
            short_name = cluster_name[:30] + "..." if len(cluster_name) > 30 else cluster_name

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    name=f"{short_name} ({total} queries, {quality_pct:.0f}% good)",
                    marker=dict(
                        size=10,
                        color=colors,
                        line=dict(width=1, color='white'),
                        opacity=0.8
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    customdata=[[label]] * len(x_coords)
                )
            )

            # Skip annotation for small clusters (below visualization threshold)
            if total < min_cluster_size_viz:
                continue

            # Add cluster center annotation with jitter to avoid overlap
            center_x = float(x_coords.mean())
            center_y = float(y_coords.mean())
            
            # Apply jitter if position is too close to existing annotations
            jitter_x, jitter_y = 0, 0
            for existing_x, existing_y in annotations_placed:
                distance = np.sqrt((center_x - existing_x)**2 + (center_y - existing_y)**2)
                if distance < 2.0:  # Too close, apply offset
                    jitter_x += 1.5
                    jitter_y += 1.0
            
            final_x = center_x + jitter_x
            final_y = center_y + jitter_y
            annotations_placed.append((final_x, final_y))

            fig.add_annotation(
                x=final_x,
                y=final_y,
                text=f"<b>{short_name}</b><br>{total} queries",
                showarrow=False,
                font=dict(size=9, color="black"),
                bgcolor="white",
                opacity=0.8,
                bordercolor="gray",
                borderwidth=1,
                borderpad=4
            )

        # Add noise points
        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_indices = np.where(noise_mask)[0]
            x_noise = embeddings_2d[noise_mask, 0]
            y_noise = embeddings_2d[noise_mask, 1]

            hover_texts_noise = []
            for idx in noise_indices:
                query_text = queries[idx] if idx < len(queries) else "Unknown"
                short_query = query_text[:80] + "..." if len(query_text) > 80 else query_text
                hover_texts_noise.append(f"<b>Noise:</b> {short_query}")

            fig.add_trace(
                go.Scatter(
                    x=x_noise,
                    y=y_noise,
                    mode='markers',
                    name=f"Noise ({len(noise_indices)} queries)",
                    marker=dict(
                        size=6,
                        color='lightgray',
                        opacity=0.4,
                        line=dict(width=0.5, color='gray')
                    ),
                    text=hover_texts_noise,
                    hovertemplate='%{text}<extra></extra>'
                )
            )

        # Update layout
        fig.update_layout(
            width=self.width,
            height=self.height,
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=False
            ),
            legend=dict(
                title=dict(text="Clusters (click to toggle)"),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                font=dict(size=10)
            ),
            hovermode='closest',
            plot_bgcolor='white'
        )

        # Add quality legend annotation (single annotation, added once)
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<b>Quality Legend:</b><br>" +
                 "🟢 Green = Well answered<br>" +
                 "🔴 Red = Partial answer<br>" +
                 "🟠 Orange = Content gap<br>" +
                 "⚪ Gray = Noise/Uncategorized",
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=8
        )

        # Enable zoom, pan, box select
        fig.update_xaxes(
            rangeslider=dict(visible=False),
            scaleanchor="y",
            scaleratio=1
        )

        # Return HTML string
        return fig.to_html(include_plotlyjs='cdn', full_html=True)

    def create_treemap(
        self,
        cluster_names: Dict[int, str],
        cluster_quality: Dict,
        evaluated_results: List[Dict],
        title: str = "Query Clusters (Treemap View)"
    ) -> str:
        """
        Create interactive treemap visualization.

        Treemaps use space efficiently and eliminate overlap issues.
        Each cluster is a rectangle sized by query count.

        Args:
            cluster_names: Mapping of cluster_id to descriptive name
            cluster_quality: Quality metrics per cluster
            evaluated_results: List of evaluation results
            title: Plot title

        Returns:
            HTML string for the interactive treemap
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Run: uv add plotly")
            raise

        # Build treemap data
        labels = []
        parents = []
        values = []
        colors = []
        hover_texts = []

        # Root node
        labels.append("All Clusters")
        parents.append("")
        values.append(0)  # Will be sum of children
        colors.append("lightgray")
        hover_texts.append("Root")

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(
            cluster_quality.items(),
            key=lambda x: x[1].get('query_count', 0),
            reverse=True
        )

        for cluster_id, metrics in sorted_clusters:
            cluster_id_int = int(cluster_id)
            # Get name from cluster_names if available, otherwise use metrics name
            name = cluster_names.get(cluster_id_int, metrics.get('name', f"Cluster {cluster_id}"))
            count = metrics.get('query_count', 0)
            partial = metrics.get('partial_answers', 0)
            quality_pct = metrics.get('quality_pct', 0)

            # Truncate long names
            short_name = name[:40] + "..." if len(name) > 40 else name

            labels.append(short_name)
            parents.append("All Clusters")
            values.append(count)

            # Color based on quality
            if quality_pct >= 70:
                colors.append("#2ecc71")  # Green
            elif quality_pct >= 40:
                colors.append("#f39c12")  # Orange
            else:
                colors.append("#e74c3c")  # Red

            # Hover text
            hover_text = f"<b>{name}</b><br>"
            hover_text += f"Queries: {count}<br>"
            hover_text += f"Partial answers: {partial}<br>"
            hover_text += f"Quality: {quality_pct:.0f}%"
            hover_texts.append(hover_text)

        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            text=labels,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts,
            marker=dict(
                colors=colors,
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=11),
            pathbar=dict(visible=False)
        ))

        # Update layout
        fig.update_layout(
            width=self.width,
            height=self.height,
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            margin=dict(t=50, l=25, r=25, b=25)
        )

        # Add quality legend annotation
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<b>Quality Legend:</b><br>" +
                 "🟢 Green = Well answered (≥70%)<br>" +
                 "🟠 Orange = Mixed (40-69%)<br>" +
                 "🔴 Red = Needs work (<40%)",
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=8
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=True)

    def save_html(self, html_content: str, filepath: str):
        """
        Save HTML visualization to file.

        Args:
            html_content: HTML string from create_interactive_plot
            filepath: Output path (e.g., "clusters.html")
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Saved interactive cluster visualization to {filepath}")
