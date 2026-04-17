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
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAGTriage: Turn RAG failures into a prioritized backlog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run ragtriage-eval                    # Run evaluation only
  uv run ragtriage-eval --cluster          # Evaluation + clustering
  uv run ragtriage-eval --full             # Run all analyses
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full analysis including all available features"
    )
    
    args = parser.parse_args()
    
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
    
    # Step 1: Evaluate (always run)
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
    
    # Step 3: Generate reports
    print("\n=== Step 3: Generating reports ===")
    reporter = ReportGenerator()
    
    # Markdown report
    report = reporter.generate_report(analyzed)
    report_path = output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to {report_path}")
    
    # CSV
    df = reporter.generate_csv(analyzed)
    csv_path = output_dir / "action_items.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Action items saved to {csv_path}")
    
    # Step 4: Clustering (if requested)
    if args.cluster or args.full:
        print("\n=== Step 4: Clustering queries ===")
        from .clustering.pipeline import ClusteringPipeline
        
        pipeline = ClusteringPipeline(min_cluster_size=3)
        
        cluster_results = pipeline.run(
            queries,
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
    
    if args.cluster or args.full:
        cluster_count = cluster_results['n_clusters']
        print(f"Query clusters found: {cluster_count}")
    
    print(f"\nOutput files:")
    print(f"  - {report_path}")
    print(f"  - {csv_path}")
    if args.cluster or args.full:
        print(f"  - {cluster_path}")
    
    print(f"\nNext steps:")
    print(f"1. Review {report_path.name} for overview")
    print(f"2. Open {csv_path.name} in Excel/Sheets for detailed action items")
    print(f"3. Start with DOC_WRITE items (new articles needed)")


if __name__ == "__main__":
    main()
