# ragtriage Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Support Queries (JSONL)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: LANE CLASSIFICATION (LLM)                          │
│  - INCIDENT: Product bugs → Engineering                     │
│  - UNDERSTANDING: How-to → RAG Evaluation                   │
│  - WORKFLOW: Commands (ticket, demo) → Skip                 │
│  - SPAM: Marketing, gibberish → Filter                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ (Understanding queries only)
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: RAG EVALUATION                                     │
│  - Query your actual RAG system                             │
│  - Get answer + sources                                     │
│  - Score: relevance, accuracy, completeness, etc.           │
│  - Grade: well_answered | partial | content_gap | error     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ (Partial answers only)
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: GAP ANALYSIS (LLM)                                 │
│  - Categorize: BILLING, LEAVE, TIMESHEET, etc.              │
│  - Classify: doc_write vs doc_update                        │
│  - Identify: Specific gap in existing docs                  │
│  - Recommend: Target article to fix                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUTS                                                    │
│  - CSV: All queries with categories, actions, priorities    │
│  - Markdown: Executive summary + top action items           │
└─────────────────────────────────────────────────────────────┘
```

## Lane Classification

### Purpose
Separate queries that need different handling strategies.

### Implementation
Uses GPT-4/Claude with few-shot prompting:

```python
prompt = """
Classify this support query into one of four lanes:

INCIDENT: User reports something is broken/not working
Example: "My timesheet totals are not accurate"

UNDERSTANDING: User asks how to do something
Example: "How do I cancel my subscription?"

WORKFLOW: User types a command word
Example: "ticket", "demo", "support"

SPAM: Marketing pitch, gibberish, off-topic
Example: "I am a marketing director, interested in partnership"

Query: {query}
Lane:"""
```

### Why This Matters
- **Don't evaluate RAG on incident reports** - It's not a knowledge problem
- **Don't waste time on spam** - 9-10% of queries are noise
- **Focus documentation effort** - Only understanding queries need docs

## RAG Evaluation

### Scoring Dimensions (1-5 scale)

1. **Relevance**: Does the answer address the user's question?
2. **Factual Accuracy**: Is the information correct?
3. **Completeness**: Does it cover all aspects of the question?
4. **Conciseness**: Is it appropriately brief?
5. **Source Quality**: Are citations relevant and authoritative?

### Grading Logic

```python
overall_score = average(dimensions)

if overall_score >= 4.0:
    bucket = "well_answered"
elif score >= 2.0:
    if has_relevant_articles:
        bucket = "partial"  # Doc exists, needs improvement
    else:
        bucket = "content_gap"  # Missing documentation
else:
    bucket = "error"  # Something went wrong
```

### Why Multiple Dimensions?

A high-scoring answer can still be useless:
- Score 5/5 on accuracy, 1/5 on completeness → Partial answer
- Score 5/5 on relevance, 2/5 on source quality → Risk of hallucination

## Gap Analysis

### Category Taxonomy

Categories are business-impact prioritized:

```python
CATEGORIES = {
    "BILLING": 1,      # Revenue critical
    "LEAVE": 2,        # Core feature
    "TIMESHEET": 3,    # Core feature
    "MANAGER": 4,      # Admin experience
    "REPORTS": 5,      # Power users
    "INTEGRATIONS": 6, # Expansion
    "SETUP": 7,        # Onboarding
    "NOTIFICATIONS": 8, # Nice to have
    "GENERAL": 9,      # Catch-all
}
```

### Action Classification

**doc_write**: Information doesn't exist
- User asks: "How do I cancel my subscription?"
- Current docs: Deactivation, leave cancellation
- Action: Write "How to Cancel Your Subscription"

**doc_update**: Information exists but incomplete
- User asks: "Can I set up auto punch-out for everyone?"
- Current docs: "How to set up auto-punch out" (individual only)
- Action: Update with bulk setup instructions

**rag_improvement**: Right docs retrieved, but answer poor
- User asks: "How do I cancel?"
- Retrieved docs: Right ones
- RAG answer: Wrong, incomplete, or misleading
- Action: Fix RAG prompt, embeddings, or reranking

## Data Flow

### Input
```jsonl
{"query_id": "abc123", "question": "How do I cancel?", "timestamp": "2024-01-15"}
{"query_id": "abc124", "question": "My balance is wrong", "timestamp": "2024-01-16"}
```

### Intermediate (after evaluation)
```jsonl
{
  "query_id": "abc123",
  "question": "How do I cancel?",
  "lane": "understanding",
  "rag_answer": "To cancel...",
  "scores": {"relevance": 3, "accuracy": 5, ...},
  "overall_score": 3.0,
  "bucket": "partial",
  "related_articles": ["How to deactivate"],
  "why_failed": "Answer is about deactivation, not subscription cancellation"
}
```

### Final (after analysis)
```jsonl
{
  "query_id": "abc123",
  "question": "How do I cancel?",
  "lane": "understanding",
  "category": "BILLING",
  "action": "doc_write",
  "target_article": "How to Cancel Your Subscription",
  "gap": "No article exists covering subscription cancellation process",
  "priority": 1,
  "related_queries_count": 15
}
```

## Cost Considerations

### Per-Query Costs (OpenAI GPT-4)

| Step | Tokens | Cost |
|------|--------|------|
| Lane classification | ~500 | $0.0075 |
| RAG evaluation | ~800 | $0.012 |
| Gap analysis | ~1000 | $0.015 |
| **Total** | ~2300 | **~$0.035** |

For 1,000 queries: **~$35**

### Optimization Strategies

1. **Batch lane classification**: Process 10 queries per prompt
2. **Skip evaluation for obvious cases**: If lane != understanding, skip
3. **Cache RAG responses**: Don't re-query for same questions
4. **Use cheaper models for classification**: GPT-3.5 for lanes, GPT-4 for evaluation

## Extensibility

### Adding New Categories

Edit `config/categories.yaml`:

```yaml
categories:
  - name: SECURITY
    priority: 2
    keywords: ["password", "2fa", "authentication", "sso"]
    description: "Security and access control"
```

### Custom Lane Classifier

Implement `BaseLaneClassifier`:

```python
class MyLaneClassifier(BaseLaneClassifier):
    def classify(self, query: str) -> Lane:
        # Your logic here
        return Lane.UNDERSTANDING
```

### Different Output Formats

Implement `BaseExporter`:

```python
class NotionExporter(BaseExporter):
    def export(self, results: List[Result]) -> None:
        # Create Notion pages
        pass
```

## Limitations

1. **Requires representative queries**: Won't find gaps in features no one uses
2. **LLM costs add up**: ~$35 per 1,000 queries
3. **Not real-time**: Batch analysis, not live monitoring
4. **English only**: Current prompts are English-centric
5. **Domain-specific taxonomy**: Default categories are SaaS-focused

## Future Directions

- **Trend analysis**: Compare month-over-month to spot emerging issues
- **Agent integration**: Auto-suggest doc updates to writers
- **Live monitoring**: Webhook for new queries
- **A/B testing**: Compare RAG versions on same queries
- **Multi-language**: Support for Spanish, German, etc.
