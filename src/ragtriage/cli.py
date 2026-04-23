"""Command-line interface for ragtriage."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .analyzer import QueryAnalyzer
from .evaluator import RAGEvaluator
from .reporter import ReportGenerator


def load_queries(path: str) -> list:
    """Load queries from JSONL file."""
    queries = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def check_api_key():
    """Check if OpenAI API key is set."""
    # Load .env from project root (where pyproject.toml is)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Fallback to cwd
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)


def run_evaluation(args):
    """Run full evaluation pipeline."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    print(f"Loading queries from {args.input}...")
    try:
        queries = load_queries(args.input)
        print(f"Loaded {len(queries)} queries")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        print("\nTo get started:")
        print("1. Copy data/sample_queries.jsonl to your own file")
        print("2. Update it with your RAG queries, contexts, and answers")
        print("3. Run: uv run ragtriage-eval --input your_file.jsonl")
        sys.exit(1)

    # Check API key for OpenAI-dependent operations
    check_api_key()

    # Step 1: Evaluate
    print("\n=== Step 1: Evaluating RAG answers ===")
    evaluator = RAGEvaluator(model=args.model)
    eval_path = output_dir / "evaluation_results.json"
    evaluated = evaluator.evaluate_dataset(queries, str(eval_path))
    print(f"✓ Evaluation complete. Results saved to {eval_path}")

    # Step 2: Analyze
    print("\n=== Step 2: Analyzing queries and generating action items ===")
    analyzer = QueryAnalyzer(model=args.model)
    analyzed = analyzer.analyze_results(evaluated)
    analyzed_path = output_dir / "analyzed_results.json"
    with open(analyzed_path, 'w') as f:
        json.dump(analyzed, f, indent=2)
    print(f"✓ Analysis complete. Results saved to {analyzed_path}")

    # Step 3: Clustering (if requested) - Do this BEFORE report generation
    cluster_results = None
    if args.cluster:
        print("\n=== Step 3: Clustering queries ===")
        from .clustering.pipeline import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=3)

        cluster_results = pipeline.run(
            queries,
            evaluated_results=analyzed,
            create_visualization=True,
            output_dir=str(output_dir)
        )

        # Save results
        cluster_path = output_dir / "clustering_results.json"
        pipeline.save_results(cluster_results, str(cluster_path))

        # Print summary
        print(f"\n✓ Clustering complete")
        print(f"  Found {cluster_results['n_clusters']} clusters from {cluster_results['n_queries']} queries")
        print(f"  Results saved to {cluster_path}")

        if cluster_results.get('visualization_path'):
            print(f"  Visualization: {cluster_results['visualization_path']}")
        if cluster_results.get('summary_path'):
            print(f"  Summary: {cluster_results['summary_path']}")

    # Step 4: Generate reports
    print("\n=== Step 4: Generating reports ===")
    reporter = ReportGenerator()

    # Markdown report
    report = reporter.generate_report(analyzed)

    # Add cluster section if clustering was done
    if cluster_results:
        cluster_section = reporter.generate_cluster_section(cluster_results)
        report += cluster_section
    
    # Add diagnostics section
    diagnostics_section = reporter.generate_diagnostics_section(analyzed)
    if diagnostics_section:
        report += diagnostics_section

    report_path = output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to {report_path}")

    # CSV
    df = reporter.generate_csv(analyzed)
    csv_path = output_dir / "action_items.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Action items saved to {csv_path}")

    # Summary
    understanding_count = len([r for r in analyzed if r.get("lane") == "UNDERSTANDING"])
    action_count = len([r for r in analyzed
                       if r.get("lane") == "UNDERSTANDING"
                       and r.get("action") in ["DOC_WRITE", "DOC_UPDATE"]])

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total queries: {len(analyzed)}")
    print(f"Understanding queries: {understanding_count}")
    print(f"Action items generated: {action_count}")

    if args.cluster:
        print(f"Query clusters found: {cluster_results['n_clusters']}")

    print(f"\nOutput files:")
    print(f"  - {report_path}")
    print(f"  - {csv_path}")
    if args.cluster:
        print(f"  - {cluster_path}")

    print(f"\nNext steps:")
    print(f"1. Review {report_path.name} for overview")
    print(f"2. Open {csv_path.name} in Excel/Sheets for detailed action items")
    print(f"3. Start with DOC_WRITE items (new articles needed)")


def run_clustering(args):
    """Run only clustering analysis."""
    # Load .env from project root first
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    print(f"Loading queries from {args.input}...")
    try:
        queries = load_queries(args.input)
        print(f"Loaded {len(queries)} queries")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Run clustering (no OpenAI needed)
    print("\n=== Clustering queries ===")
    from .clustering.pipeline import ClusteringPipeline

    pipeline = ClusteringPipeline(min_cluster_size=3)

    cluster_results = pipeline.run(
        queries,
        evaluated_results=None,  # No evaluation, just clustering
        create_visualization=True,
        output_dir=str(output_dir)
    )

    # Save results
    cluster_path = output_dir / "clustering_results.json"
    pipeline.save_results(cluster_results, str(cluster_path))

    # Print summary
    print(f"\n✓ Clustering complete")
    print(f"  Found {cluster_results['n_clusters']} clusters from {cluster_results['n_queries']} queries")

    if cluster_results.get('visualization_path'):
        print(f"  Visualization: {cluster_results['visualization_path']}")

    # Generate and print text summary
    print("\n" + "="*70)
    print("CLUSTER SUMMARY")
    print("="*70)

    for label, name in sorted(cluster_results['cluster_names'].items()):
        # Count queries in this cluster
        # We need to reload results to get this info
        pass

    print(f"\nTo get detailed quality analysis, run:")
    print(f"  uv run ragtriage-eval --cluster -i {args.input}")


def main():
    """Main entry point."""
    # Load .env from project root first (before anything else)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="RAGTriage: Turn RAG failures into a prioritized backlog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  uv run ragtriage-eval              # Full evaluation
  uv run ragtriage-eval --cluster    # Evaluation + clustering
  uv run ragtriage-cluster           # Clustering only (no LLM needed)

Examples:
  uv run ragtriage-eval                                    # Use sample data
  uv run ragtriage-eval -i my_queries.jsonl                # Your data
  uv run ragtriage-eval --cluster -o results/              # With clustering
  uv run ragtriage-cluster -i queries.jsonl -o clusters/   # Cluster only
        """
    )

    parser.add_argument(
        "--input", "-i",
        default="data/sample_queries.jsonl",
        help="Path to input queries (JSONL format)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Also run query clustering analysis"
    )

    args = parser.parse_args()

    # Check which command was invoked
    import sys
    cmd = sys.argv[0].split('/')[-1] if sys.argv[0] else 'ragtriage-eval'

    if 'ragtriage-cluster' in cmd:
        run_clustering(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
