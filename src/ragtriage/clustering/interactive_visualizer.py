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
        
        # Store cluster data for drill-down table
        cluster_queries_data = {}

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
            
            # Get queries for this cluster from evaluated_results
            cluster_queries = []
            recommended_actions = metrics.get('recommended_actions', {})
            top_action = list(recommended_actions.keys())[0] if recommended_actions else "UNKNOWN"
            
            # Find queries belonging to this cluster
            if evaluated_results:
                for result in evaluated_results:
                    # Match by cluster name or check if result has cluster_id
                    result_cluster = result.get('cluster_id')
                    if result_cluster == cluster_id_int or result_cluster == str(cluster_id_int):
                        query_text = result.get('query', '')
                        if query_text:
                            cluster_queries.append(query_text)

            # Truncate long names for display
            short_name = name[:35] + "..." if len(name) > 35 else name

            labels.append(short_name)
            parents.append("All Clusters")
            values.append(count)
            
            # Store data for potential drill-down
            cluster_queries_data[short_name] = {
                'full_name': name,
                'queries': cluster_queries[:5],  # Top 5 queries
                'action': top_action,
                'count': count
            }

            # Color based on quality
            if quality_pct >= 70:
                colors.append("#2ecc71")  # Green
            elif quality_pct >= 40:
                colors.append("#f39c12")  # Orange
            else:
                colors.append("#e74c3c")  # Red

            # Enhanced hover text with actionable info
            hover_text = f"<b>{name}</b><br><br>"
            hover_text += f"<b>Action needed:</b> {top_action}<br>"
            hover_text += f"Queries: {count} | Partial: {partial}<br><br>"
            
            # Add sample queries
            if cluster_queries:
                hover_text += "<b>Sample queries:</b><br>"
                for i, q in enumerate(cluster_queries[:3], 1):
                    short_q = q[:60] + "..." if len(q) > 60 else q
                    hover_text += f"{i}. {short_q}<br>"
            
            hover_texts.append(hover_text)

        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts,
            marker=dict(
                colors=colors,
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=11),
            pathbar=dict(visible=False),
            textposition='middle center',
            insidetextfont=dict(size=10)
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

    def create_actionable_treemap(
        self,
        hierarchy: Dict,
        title: str = "Actionable Items by Category → Topic → Action"
    ) -> str:
        """
        Create actionable treemap with click-to-view details panel.
        
        Two-panel layout: treemap on left (65%), details panel on right (35%).
        Click any leaf tile to see full details persistently.
        
        Args:
            hierarchy: {category: {topic: {action: [items]}}}
            title: Plot title
            
        Returns:
            HTML string for the actionable treemap with details panel
        """
        try:
            import plotly.graph_objects as go
            import json
        except ImportError:
            logger.error("Plotly not installed. Run: uv add plotly")
            raise
        
        # Build treemap data and collect item details
        labels = []
        parents = []
        values = []
        colors = []
        ids = []  # Unique IDs for each node
        
        # Store all item data for JS access
        item_data = {}  # id -> {category, topic, action, items}
        
        # Color mapping for actions
        action_colors = {
            "DOC_WRITE": "#e74c3c",  # Red = needs new doc
            "DOC_UPDATE": "#f39c12"  # Orange = update existing
        }
        
        total_items = 0
        node_id = 0
        
        # Root node
        labels.append("Actionable Items")
        parents.append("")
        values.append(0)
        colors.append("lightgray")
        ids.append("root")
        node_id += 1
        
        for category, topics in sorted(hierarchy.items()):
            # Category node
            cat_count = sum(
                len(items) 
                for topics_dict in topics.values() 
                for items in topics_dict.values()
            )
            total_items += cat_count
            
            cat_id = f"cat_{node_id}"
            cat_label = f"📁 {category}"
            labels.append(cat_label)
            parents.append("Actionable Items")
            values.append(cat_count)
            colors.append("#3498db")  # Blue for categories
            ids.append(cat_id)
            node_id += 1
            
            for topic, actions in sorted(topics.items()):
                # Topic node
                topic_count = sum(len(items) for items in actions.values())
                
                topic_id = f"topic_{node_id}"
                topic_label = f"📝 {topic[:30]}"
                labels.append(topic_label)
                parents.append(cat_label)
                values.append(topic_count)
                colors.append("#9b59b6")  # Purple for topics
                ids.append(topic_id)
                node_id += 1
                
                for action, items in actions.items():
                    if not items:
                        continue
                    
                    # Action/Leaf node
                    action_id = f"action_{node_id}"
                    action_label = f"{action} ({len(items)})"
                    labels.append(action_label)
                    parents.append(topic_label)
                    values.append(len(items))
                    colors.append(action_colors.get(action, "#95a5a6"))
                    ids.append(action_id)
                    
                    # Store full item data for this node
                    item_data[action_id] = {
                        "category": category,
                        "topic": topic,
                        "action": action,
                        "count": len(items),
                        "items": items
                    }
                    node_id += 1
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            marker=dict(
                colors=colors,
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=11),
            pathbar=dict(visible=True),
            textposition='middle center',
            insidetextfont=dict(size=10)
        ))
        
        # Update layout for left panel
        fig.update_layout(
            width=900,
            height=700,
            title=dict(
                text=f"{title}<br><sub>{total_items} items needing documentation work</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            margin=dict(t=80, l=25, r=25, b=25)
        )
        
        # Get treemap HTML div
        treemap_div = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="treemap")
        
        # Serialize item data for JavaScript
        item_data_json = json.dumps(item_data)
        
        # Create full HTML with two-panel layout
        html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }}
        .header {{
            background: #2c3e50;
            color: white;
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 18px;
            font-weight: 500;
        }}
        .header .count {{
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 14px;
        }}
        .container {{
            display: flex;
            height: calc(100vh - 60px);
        }}
        .treemap-panel {{
            flex: 0 0 65%;
            background: white;
            padding: 15px;
            overflow: auto;
        }}
        .details-panel {{
            flex: 0 0 35%;
            background: #fafafa;
            border-left: 1px solid #ddd;
            padding: 20px;
            overflow-y: auto;
        }}
        .details-panel h2 {{
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ddd;
        }}
        .empty-state {{
            color: #999;
            font-style: italic;
            text-align: center;
            padding: 40px 20px;
        }}
        .item-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid #ddd;
        }}
        .item-card.doc-write {{ border-left-color: #e74c3c; }}
        .item-card.doc-update {{ border-left-color: #f39c12; }}
        .item-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .item-number {{
            background: #2c3e50;
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
        }}
        .item-action {{
            font-size: 11px;
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 3px;
            text-transform: uppercase;
        }}
        .item-action.doc-write {{ background: #fee; color: #c0392b; }}
        .item-action.doc-update {{ background: #fff3e0; color: #e65100; }}
        .item-query {{
            font-size: 14px;
            color: #333;
            margin-bottom: 10px;
            line-height: 1.4;
        }}
        .item-meta {{
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
        }}
        .item-meta strong {{
            color: #333;
        }}
        .item-gap {{
            font-size: 12px;
            color: #555;
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            line-height: 1.5;
        }}
        .breadcrumb {{
            background: #e8f4f8;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 13px;
        }}
        .breadcrumb span {{
            color: #666;
        }}
        .breadcrumb strong {{
            color: #2c3e50;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📋 {title}</h1>
        <div style="display: flex; align-items: center; gap: 20px;">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot" style="background: #e74c3c;"></div>
                    <span>DOC_WRITE (new article)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #f39c12;"></div>
                    <span>DOC_UPDATE (existing)</span>
                </div>
            </div>
            <div class="count">{total_items} items</div>
        </div>
    </div>
    <div class="container">
        <div class="treemap-panel">
            {treemap_div}
        </div>
        <div class="details-panel" id="detailsPanel">
            <h2>Item Details</h2>
            <div class="empty-state">
                👆 Click on any DOC_WRITE or DOC_UPDATE tile in the treemap to view details
            </div>
        </div>
    </div>
    <script>
        const itemData = {item_data_json};
        
        document.addEventListener('DOMContentLoaded', function() {{
            const treemapElement = document.getElementById('treemap');
            
            // Wait for Plotly to render
            setTimeout(function() {{
                const treemap = document.querySelector('.treemap-panel .js-plotly-plot');
                if (treemap) {{
                    treemap.on('plotly_click', function(data) {{
                        const point = data.points[0];
                        const nodeId = point.id;
                        
                        if (itemData[nodeId]) {{
                            showDetails(itemData[nodeId]);
                        }}
                    }});
                }}
            }}, 500);
        }});
        
        function showDetails(data) {{
            const panel = document.getElementById('detailsPanel');
            
            let itemsHtml = '';
            data.items.forEach((item, idx) => {{
                const actionClass = data.action === 'DOC_WRITE' ? 'doc-write' : 'doc-update';
                itemsHtml += `
                    <div class="item-card ${{actionClass}}">
                        <div class="item-header">
                            <div class="item-number">${{idx + 1}}</div>
                            <div class="item-action ${{actionClass}}">${{data.action}}</div>
                        </div>
                        <div class="item-query">${{escapeHtml(item.query)}}</div>
                        <div class="item-meta">
                            <strong>Target:</strong> ${{item.target_article || 'N/A'}}
                        </div>
                        <div class="item-gap">
                            <strong>Gap:</strong> ${{escapeHtml(item.gap)}}
                        </div>
                    </div>
                `;
            }});
            
            panel.innerHTML = `
                <h2>Item Details</h2>
                <div class="breadcrumb">
                    <strong>${{data.category}}</strong> <span>›</span> 
                    <strong>${{data.topic}}</strong> <span>›</span> 
                    <strong>${{data.action}}</strong>
                    <span style="float: right; color: #999;">${{data.count}} item${{data.count > 1 ? 's' : ''}}</span>
                </div>
                ${{itemsHtml}}
            `;
        }}
        
        function escapeHtml(text) {{
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
    </script>
</body>
</html>'''
        
        return html_template

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
