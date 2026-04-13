#!/usr/bin/env python3
"""
Final analysis using actual RAG data.
Determines for each partial answer:
- RAG_FAILURE: Retrieved docs exist but RAG didn't use them well
- DOC_GAP: Docs don't actually contain the needed info
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

# Load API key
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def normalize_analysis(result, record):
    """Normalize LLM output to consistent format."""
    # Handle various key formats the LLM might return
    def get_key(*keys):
        for k in keys:
            if k in result:
                return result[k]
        return None
    
    return {
        'query_id': record.get('query_id'),
        'question': record.get('question'),
        'score': record.get('overall_score'),
        'root_cause': get_key('root_cause', 'Root_Cause', 'is_RAG_FAILURE', 'RAG_FAILURE'),
        'category': get_key('category', 'Category', 'classification'),
        'topic': get_key('topic', 'Topic', 'subject'),
        'rag_retrieved_right_docs': get_key('rag_retrieved_right_docs', 'Relevant_Docs_Retrieved', 'relevant_docs_retrieved'),
        'article_exists_with_info': get_key('article_exists_with_info', 'Doc_Exists_With_Info', 'doc_exists_with_info'),
        'recommendation': get_key('recommendation', 'Recommendation'),
        'target_article': get_key('target_article', 'Target_Article', 'target_article_name', 'Target_Article_Name'),
        'specific_gap': get_key('specific_gap', 'Specific_Gap', 'specifically_missing', 'Missing_Info', "What's_Specifically_Missing"),
        'confidence': get_key('confidence', 'Confidence'),
        'rag_sources': record.get('rag_sources', []),
        'related_articles': record.get('related_articles_found', [])
    }

def analyze_record(record):
    """Analyze a single partial answer."""
    question = record.get('question', '')
    rag_answer = record.get('rag_answer', '')
    rag_sources = record.get('rag_sources', [])
    related_articles = record.get('related_articles_found', [])
    why_failed = record.get('why_failed', '')
    scores = record.get('scores', {})
    
    prompt = f"""You are analyzing a failed RAG (Retrieval-Augmented Generation) answer.

USER QUESTION:
{question}

RAG SYSTEM'S ANSWER:
{rag_answer[:800]}

ARTICLES RAG RETRIEVED: {json.dumps(rag_sources[:5])}

RELATED ARTICLES IN KNOWLEDGE BASE: {json.dumps(related_articles[:5])}

WHY THE ANSWER FAILED: {why_failed}

SCORES: Relevance {scores.get('relevance', 'N/A')}/5, Completeness {scores.get('completeness', 'N/A')}/5

---

Determine the ROOT CAUSE:

1. RAG_FAILURE: Articles exist with the info, but RAG either:
   - Retrieved wrong documents
   - Retrieved right docs but synthesized poorly
   - Failed to extract the specific info from the docs

2. DOC_GAP: The retrieved articles genuinely don't contain the information needed

BE CRITICAL: If the articles mention the topic but lack the SPECIFIC answer, that's DOC_GAP (content missing), not RAG_FAILURE.

Respond in this EXACT format (JSON):
{{
  "root_cause": "RAG_FAILURE" or "DOC_GAP",
  "category": "One of: TIMESHEET, LEAVE, BILLING, SETUP, INTEGRATIONS, REPORTS, MANAGER, NOTIFICATIONS, GENERAL",
  "topic": "3-5 word description",
  "rag_retrieved_relevant_docs": true or false,
  "doc_contains_answer": true or false,
  "recommendation": "doc_update" or "doc_write" or "rag_improvement",
  "target_article": "Name of article to update or create",
  "specific_gap": "What info is missing from docs OR what RAG did wrong",
  "confidence": "high" or "medium" or "low"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You analyze RAG system failures precisely. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500
        )
        result = json.loads(response.choices[0].message.content)
        return normalize_analysis(result, record)
    except Exception as e:
        return {
            'query_id': record.get('query_id'),
            'question': record.get('question'),
            'score': record.get('overall_score'),
            'error': str(e),
            'root_cause': 'ERROR',
            'category': 'ERROR',
            'recommendation': 'ERROR'
        }

def main():
    # Load data
    print("Loading evaluation data...")
    with open('rag_evaluation.jsonl', 'r') as f:
        records = [json.loads(line) for line in f]
    
    # Get partial answers (these are where RAG tried but failed)
    partial = [r for r in records if r.get('bucket') == 'partial' and not r.get('skipped')]
    
    print(f"Found {len(partial)} partial answers to analyze")
    print(f"Estimated cost: ${len(partial) * 0.002:.2f}\n")
    
    # Process in batches to show progress
    analyzed = []
    batch_size = 50
    total_cost = 0
    
    for batch_start in range(0, len(partial), batch_size):
        batch_end = min(batch_start + batch_size, len(partial))
        batch = partial[batch_start:batch_end]
        
        print(f"Processing batch {batch_start+1}-{batch_end} of {len(partial)}...")
        
        for record in batch:
            result = analyze_record(record)
            analyzed.append(result)
            total_cost += 0.002  # Approximate cost per analysis
        
        print(f"  ✓ Batch complete. Running total: ${total_cost:.2f}")
    
    print(f"\n✓ Analysis complete!")
    print(f"Total cost: ${total_cost:.2f}")
    
    # Filter out errors
    valid = [r for r in analyzed if 'error' not in r]
    errors = [r for r in analyzed if 'error' in r]
    
    print(f"Valid analyses: {len(valid)}, Errors: {len(errors)}")
    
    # Group by recommendation
    by_rec = defaultdict(list)
    for r in valid:
        by_rec[r.get('recommendation', 'UNKNOWN')].append(r)
    
    print(f"\nRecommendations:")
    for rec, items in sorted(by_rec.items()):
        print(f"  - {rec}: {len(items)}")
    
    # Group by category and recommendation
    by_cat_rec = defaultdict(lambda: defaultdict(list))
    for r in valid:
        cat = r.get('category', 'UNKNOWN')
        rec = r.get('recommendation', 'UNKNOWN')
        by_cat_rec[cat][rec].append(r)
    
    # Generate report
    report = f"""# RAG Root Cause Analysis

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Analyzed: {len(valid)} partial answers (RAG tried but failed)

## Summary

| Recommendation | Count | Description |
|----------------|-------|-------------|
"""
    
    rec_descriptions = {
        'doc_update': 'Article exists, needs content added',
        'doc_write': 'No relevant article exists, write new one',
        'rag_improvement': 'Docs exist with info, RAG retrieval/prompting issue'
    }
    
    for rec in ['doc_write', 'doc_update', 'rag_improvement']:
        if rec in by_rec:
            desc = rec_descriptions.get(rec, '')
            report += f"| {rec} | {len(by_rec[rec])} | {desc} |\n"
    
    report += "\n## By Category\n\n"
    
    for category in sorted(by_cat_rec.keys()):
        report += f"### {category}\n\n"
        
        cat_data = by_cat_rec[category]
        
        # DOC WRITES
        if 'doc_write' in cat_data:
            items = cat_data['doc_write']
            report += f"**NEW Articles to Write ({len(items)}):**\n\n"
            
            # Group by target article
            by_article = defaultdict(list)
            for r in items:
                by_article[r.get('target_article', 'Unknown')].append(r)
            
            for article, recs in sorted(by_article.items(), key=lambda x: -len(x[1])):
                report += f"- **{article}** ({len(recs)} questions)\n"
                for r in recs[:2]:
                    report += f"  - {r['question'][:70]}...\n"
                if len(recs) > 2:
                    report += f"  - *...and {len(recs)-2} more*\n"
            report += "\n"
        
        # DOC UPDATES
        if 'doc_update' in cat_data:
            items = cat_data['doc_update']
            report += f"**Articles to UPDATE ({len(items)}):**\n\n"
            
            by_article = defaultdict(list)
            for r in items:
                by_article[r.get('target_article', 'Unknown')].append(r)
            
            for article, recs in sorted(by_article.items(), key=lambda x: -len(x[1])):
                report += f"- **{article}** ({len(recs)} questions)\n"
                for r in recs[:2]:
                    report += f"  - {r['question'][:70]}...\n"
                    report += f"    Gap: {r.get('specific_gap', 'N/A')[:80]}...\n"
            report += "\n"
        
        # RAG IMPROVEMENTS
        if 'rag_improvement' in cat_data:
            items = cat_data['rag_improvement']
            report += f"**RAG Improvements ({len(items)}):**\n"
            report += f"Docs exist but RAG failed to use them. Engineering task.\n\n"
            for r in items[:3]:
                report += f"- {r['question'][:70]}...\n"
            if len(items) > 3:
                report += f"- *...and {len(items)-3} more*\n"
            report += "\n"
        
        report += "---\n\n"
    
    # CS Priority List
    report += """## CS Team Action List

### Priority 1: Write New Articles
"""
    
    all_writes = []
    for cat, data in by_cat_rec.items():
        if 'doc_write' in data:
            by_article = defaultdict(list)
            for r in data['doc_write']:
                by_article[r.get('target_article', 'Unknown')].append(r)
            for article, recs in by_article.items():
                all_writes.append((article, len(recs), cat))
    
    all_writes.sort(key=lambda x: -x[1])
    for article, count, cat in all_writes[:15]:
        report += f"1. **{article}** ({count} questions) - {cat}\n"
    
    report += "\n### Priority 2: Update Existing Articles\n"
    
    all_updates = []
    for cat, data in by_cat_rec.items():
        if 'doc_update' in data:
            by_article = defaultdict(list)
            for r in data['doc_update']:
                by_article[r.get('target_article', 'Unknown')].append(r)
            for article, recs in by_article.items():
                all_updates.append((article, len(recs), cat))
    
    all_updates.sort(key=lambda x: -x[1])
    for article, count, cat in all_updates[:15]:
        report += f"1. **{article}** ({count} questions) - {cat}\n"
    
    report += """
### Priority 3: RAG System Improvements
Work with engineering on retrieval/prompting improvements for cases where docs exist but RAG didn't use them.

---

*Full data: analyzed_rag_detailed.json*
"""
    
    # Save outputs
    with open('rag_root_cause_analysis.md', 'w') as f:
        f.write(report)
    
    with open('analyzed_rag_detailed.json', 'w') as f:
        json.dump(valid, f, indent=2)
    
    print(f"\n✓ Report saved: rag_root_cause_analysis.md")
    print(f"✓ Data saved: analyzed_rag_detailed.json")

if __name__ == "__main__":
    main()
