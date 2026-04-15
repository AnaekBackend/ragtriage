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


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        print("Please create a .env file with: OPENAI_API_KEY=your_key")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="RAGTriage: Turn RAG failures into a prioritized backlog"
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
        print("3. Run: uv run eval --input your_file.jsonl")
        sys.exit(1)
    
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
    
    # Summary
    understanding_count = len([r for r in analyzed if r.get("lane") == "UNDERSTANDING"])
    action_count = len([r for r in analyzed 
                       if r.get("lane") == "UNDERSTANDING" 
                       and r.get("action") in ["DOC_WRITE", "DOC_UPDATE"]])
    
    print(f"\n=== Summary ===")
    print(f"Total queries: {len(analyzed)}")
    print(f"Understanding queries: {understanding_count}")
    print(f"Action items generated: {action_count}")
    print(f"\nNext steps:")
    print(f"1. Review {report_path} for overview")
    print(f"2. Open {csv_path} in Excel/Sheets for detailed action items")
    print(f"3. Start with DOC_WRITE items (new articles needed)")


if __name__ == "__main__":
    main()
