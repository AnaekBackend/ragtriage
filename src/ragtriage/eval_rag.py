#!/usr/bin/env python3
"""
RAG Evaluation System
Evaluates Voiceflow transcripts against our RAG app
Outputs one JSON line per evaluation for flexible analysis
"""

import json
import requests
import time
from datetime import datetime
from typing import Dict, List
import os

# Load .env file if present
try:
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key, value)
except FileNotFoundError:
    pass

RAG_API_URL = "http://localhost:5000/assist/"

def query_rag(question: str, platform: str = "slack") -> Dict:
    """Query the RAG API and return response."""
    try:
        start_time = time.time()
        resp = requests.post(
            RAG_API_URL,
            json={"query_string": question, "platform": platform},
            timeout=30
        )
        latency = time.time() - start_time
        
        data = resp.json()
        return {
            "success": data.get("ok", False),
            "answer": data.get("answer", ""),
            "sources": data.get("sources", []),
            "latency_ms": round(latency * 1000, 2),
            "error": data.get("reason") if not data.get("ok") else None
        }
    except Exception as e:
        return {
            "success": False,
            "answer": "",
            "sources": [],
            "latency_ms": 0,
            "error": str(e)
        }

def evaluate_with_llm(question: str, rag_answer: str, original_answer: str, sources: List[Dict]) -> Dict:
    """Use GPT-4o-mini to evaluate the RAG response."""
    
    sources_text = "\n".join([f"- {s.get('title', 'N/A')}: {s.get('url', 'N/A')}" for s in sources[:3]])
    
    prompt = f"""You are evaluating a RAG (Retrieval Augmented Generation) system for AttendanceBot support.

USER QUESTION: {question}

RAG SYSTEM ANSWER: {rag_answer}

ORIGINAL VOICEFLOW ANSWER (for reference): {original_answer[:300]}

SOURCES RETRIEVED:
{sources_text}

Evaluate the RAG answer on these 5 criteria (score 1-5):

1. RELEVANCE: Does it directly address the question?
2. FACTUAL_ACCURACY: Is the info correct based on typical AttendanceBot docs?
3. COMPLETENESS: Can the user take action with this answer?
4. CONCISENESS: Is it appropriately brief (under 150 tokens)?
5. SOURCE_QUALITY: Did it use the right article with correct URL?

Also classify:
- BUCKET: "well_answered" (score 4-5 avg), "partial" (score 2-3), or "content_gap" (score 1-2)
- BEST_POSSIBLE_ANSWER: What COULD we answer given current docs? (even if imperfect)
- CONTENT_GAP_SEVERITY: "none", "related_exists", or "complete_gap"
- RECOMMENDED_ACTION: "none", "update_existing", or "write_new"
- RELATED_ARTICLES_FOUND: List titles of any related docs that almost answered it
- WHY_FAILED: Brief explanation if score < 4

Respond with ONLY valid JSON in this exact format:
{{
  "scores": {{"relevance": 0, "factual_accuracy": 0, "completeness": 0, "conciseness": 0, "source_quality": 0}},
  "overall_score": 0.0,
  "bucket": "string",
  "best_possible_answer": "string",
  "content_gap_severity": "string",
  "recommended_action": "string",
  "related_articles_found": ["string"],
  "why_failed": "string or null"
}}"""

    try:
        # Note: Using OpenAI API - you need OPENAI_API_KEY in env
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800
        )
        
        result = json.loads(resp.choices[0].message.content)
        return result
        
    except Exception as e:
        return {
            "scores": {"relevance": 0, "factual_accuracy": 0, "completeness": 0, "conciseness": 0, "source_quality": 0},
            "overall_score": 0.0,
            "bucket": "error",
            "best_possible_answer": "",
            "content_gap_severity": "unknown",
            "recommended_action": "retry",
            "related_articles_found": [],
            "why_failed": f"Evaluation error: {str(e)}"
        }

def should_skip_evaluation(question: str) -> tuple[bool, str]:
    """
    Check if this query should be skipped from evaluation.
    Returns (should_skip, reason)
    """
    if not question:
        return True, "empty_question"
    
    # Normalize
    q = question.strip().lower()
    
    # Skip single word queries (hi, hello, plan, ticket, etc.)
    if len(q.split()) <= 1:
        return True, f"single_word:{q}"
    
    # Skip ticket-related messages (hardcoded Voiceflow flow)
    ticket_words = ['ticket', 'dialm', 'support', 'help']
    if q in ticket_words or q.startswith('ticket '):
        return True, f"ticket_flow:{q[:20]}"
    
    # Skip email addresses (part of ticketing flow)
    if '@' in q and '.' in q.split('@')[-1]:
        return True, "email_address"
    
    # Skip very short questions (less than 10 chars after stripping)
    if len(q) < 10:
        return True, f"too_short:{len(q)}chars"
    
    # Skip greetings
    greetings = ['hi ', 'hello ', 'hey ', 'good morning', 'good afternoon', 'good evening']
    if any(q.startswith(g) for g in greetings):
        return True, "greeting"
    
    return False, ""


def evaluate_single(query_data: Dict) -> Dict:
    """Evaluate one query and return complete evaluation record."""
    
    question = query_data.get("question", "")
    original_answer = query_data.get("answer", "")
    session_id = query_data.get("session_id", "")
    created_at = query_data.get("created_at", "")
    
    # Check if we should skip this query
    should_skip, skip_reason = should_skip_evaluation(question)
    if should_skip:
        return {
            "query_id": f"{session_id}_{hash(question) % 10000}",
            "session_id": session_id,
            "created_at": created_at,
            "question": question,
            "skipped": True,
            "skip_reason": skip_reason,
            "evaluated_at": datetime.now().isoformat()
        }
    
    # Query RAG
    rag_response = query_rag(question)
    
    # Evaluate
    evaluation = evaluate_with_llm(
        question=question,
        rag_answer=rag_response["answer"],
        original_answer=original_answer,
        sources=rag_response["sources"]
    )
    
    # Build complete record
    record = {
        # Original data
        "query_id": f"{session_id}_{hash(question) % 10000}",
        "session_id": session_id,
        "created_at": created_at,
        "question": question,
        "original_voiceflow_answer": original_answer[:500],  # Truncate for size
        
        # RAG response
        "rag_answer": rag_response["answer"],
        "rag_success": rag_response["success"],
        "rag_error": rag_response["error"],
        "rag_latency_ms": rag_response["latency_ms"],
        "rag_sources": [s.get("title", "") for s in rag_response["sources"]],
        
        # Evaluation scores
        "scores": evaluation["scores"],
        "overall_score": evaluation["overall_score"],
        "bucket": evaluation["bucket"],
        
        # Analysis
        "best_possible_answer": evaluation["best_possible_answer"],
        "content_gap_severity": evaluation["content_gap_severity"],
        "recommended_action": evaluation["recommended_action"],
        "related_articles_found": evaluation["related_articles_found"],
        "why_failed": evaluation["why_failed"],
        
        # Metadata
        "evaluated_at": datetime.now().isoformat(),
        "eval_version": "1.0"
    }
    
    return record

def run_evaluation(input_file: str = "voiceflow_qa.json", output_file: str = "rag_evaluation.jsonl", limit: int = None):
    """Run evaluation on all queries."""
    
    # Load queries
    with open(input_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    if limit:
        queries = queries[:limit]
    
    print(f"Evaluating {len(queries)} queries...")
    print(f"Output: {output_file}")
    print("=" * 70)
    
    # Process and write JSONL
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Evaluating: {query['question'][:50]}...")
            
            record = evaluate_single(query)
            
            # Write as JSON line
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Progress
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(queries)} completed")
    
    print("\n" + "=" * 70)
    print(f"Evaluation complete! Results in {output_file}")
    print("=" * 70)

def generate_summary_report(jsonl_file: str = "rag_evaluation.jsonl"):
    """Generate markdown summary from JSONL results."""
    
    buckets = {"well_answered": 0, "partial": 0, "content_gap": 0, "error": 0}
    skipped_reasons = {}
    content_gaps = {}
    partial_cases = []
    scores = []
    total_skipped = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            
            # Handle skipped entries
            if record.get("skipped"):
                total_skipped += 1
                reason = record.get("skip_reason", "unknown")
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue
            
            bucket = record.get("bucket", "error")
            buckets[bucket] += 1
            scores.append(record.get("overall_score", 0))
            
            # Track content gaps
            if bucket == "content_gap":
                q = record.get("question", "")[:50]
                content_gaps[q] = content_gaps.get(q, 0) + 1
            
            # Track partial answers
            if bucket == "partial":
                partial_cases.append({
                    "question": record.get("question", ""),
                    "why": record.get("why_failed", ""),
                    "related": record.get("related_articles_found", [])
                })
    
    total_evaluated = sum(buckets.values())
    total_processed = total_evaluated + total_skipped
    avg_score = sum(scores) / len(scores) if scores else 0
    
    report = f"""# RAG Evaluation Report

## Summary
- **Total queries processed:** {total_processed}
- **Queries evaluated:** {total_evaluated}
- **Queries skipped:** {total_skipped}
- **Average overall score:** {avg_score:.2f}/5.0
- **Evaluation timestamp:** {datetime.now().isoformat()}

## Distribution (of {total_evaluated} evaluated)
| Category | Count | Percentage |
|----------|-------|------------|
| ✅ Well Answered | {buckets['well_answered']} | {buckets['well_answered']/total_evaluated*100:.1f}% |
| ⚠️ Partial Answer | {buckets['partial']} | {buckets['partial']/total_evaluated*100:.1f}% |
| ❌ Content Gap | {buckets['content_gap']} | {buckets['content_gap']/total_evaluated*100:.1f}% |
| ⚠️ Errors | {buckets['error']} | {buckets['error']/total_evaluated*100:.1f}% |

## Skipped Queries ({total_skipped} total)
| Reason | Count |
|--------|-------|"""
    
    for reason, count in sorted(skipped_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
        report += f"\n| {reason} | {count} |"
    
    report += "\n\n## Content Gaps (Priority Articles to Write)\n"
    
    # Top content gaps
    sorted_gaps = sorted(content_gaps.items(), key=lambda x: x[1], reverse=True)[:15]
    for q, count in sorted_gaps:
        report += f"- **{q}...** ({count} queries)\n"
    
    report += "\n## Partial Answers (Review for Article Updates)\n"
    for case in partial_cases[:10]:
        report += f"- **Q:** {case['question'][:60]}...\n"
        report += f"  - *Why:* {case['why'][:100]}...\n"
        if case['related']:
            report += f"  - *Related docs:* {', '.join(case['related'][:2])}\n"
    
    well_answered_pct = buckets['well_answered']/total_evaluated*100 if total_evaluated else 0
    
    report += "\n## Recommendations\n"
    report += f"1. **Write new articles** for the top {len(sorted_gaps)} content gaps\n"
    report += f"2. **Update existing articles** to cover {buckets['partial']} edge cases\n"
    report += f"3. **Target:** Increase 'well answered' from {well_answered_pct:.1f}% to 99%\n\n"
    report += f"---\n*Full data available in {jsonl_file}*\n"
    
    with open("rag_evaluation_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nReport saved to: rag_evaluation_report.md")
    return report

if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        requests.get("http://localhost:5000/", timeout=5)
    except:
        print("ERROR: RAG server not running at localhost:5000")
        print("Start it first: python3 app.py")
        exit(1)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        exit(1)
    
    # Run evaluation (limit to 50 for testing, remove limit for full run)
    limit = 50 if len(sys.argv) > 1 and sys.argv[1] == "--test" else None
    
    run_evaluation(limit=limit)
    
    # Generate report
    report = generate_summary_report()
    print("\n" + report)
