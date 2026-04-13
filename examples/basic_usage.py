#!/usr/bin/env python3
"""
Basic usage example for ragtriage

This script shows how to:
1. Load support queries
2. Run lane classification
3. Evaluate RAG responses
4. Generate action items
"""

import json
from pathlib import Path

# In real usage, these would be:
# from ragtriage.lanes import classify_lanes
# from ragtriage.eval import evaluate_rag
# from ragtriage.analyze import analyze_gaps

def load_queries(filepath: str):
    """Load queries from JSONL file."""
    queries = []
    with open(filepath, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def main():
    # Step 1: Load your support queries
    print("Loading queries...")
    queries = load_queries("../data/sample_queries.jsonl")
    print(f"Loaded {len(queries)} queries")
    
    # Step 2: Classify into lanes
    print("\nClassifying lanes...")
    # results = classify_lanes(queries)
    # This would use LLM to classify each query
    
    # Example output:
    lanes = {
        "understanding": 150,
        "incident": 30,
        "workflow": 20,
        "spam": 10
    }
    print(f"Lane distribution: {lanes}")
    
    # Step 3: Evaluate RAG (for understanding queries only)
    print("\nEvaluating RAG responses...")
    # evaluation = evaluate_rag(
    #     queries=understanding_queries,
    #     rag_endpoint="http://your-rag.com/ask"
    # )
    
    # Example output:
    buckets = {
        "well_answered": 87,
        "partial": 48,
        "content_gap": 10,
        "error": 5
    }
    print(f"Bucket distribution: {buckets}")
    
    # Step 4: Analyze gaps (for partial answers only)
    print("\nAnalyzing gaps...")
    # action_items = analyze_gaps(partial_answers)
    
    # Example output:
    print("\nTop action items:")
    print("1. Write 'How to Cancel Your Subscription' (BILLING, 5 queries)")
    print("2. Update 'How to set up auto-punch out' (TIMESHEET, 4 queries)")
    print("3. Write 'How to assign managers' (MANAGER, 3 queries)")
    
    # Step 5: Export results
    print("\nExporting to CSV and Markdown...")
    # export_results(action_items, format=["csv", "markdown"])
    print("Done! Check output/action_items.csv and output/action_plan.md")

if __name__ == "__main__":
    main()
