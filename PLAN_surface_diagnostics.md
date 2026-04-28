# Surface Diagnostics: Implementation Plan

## Problem Statement

Current RAGTriage identifies:
- DOC_WRITE: No relevant contexts retrieved (content gap)
- DOC_UPDATE: Contexts exist but answer incomplete

But it doesn't diagnose **why** answers fail when contexts exist. We need to detect:
- Hallucinations (answer invents info not in context)
- Poor coverage (context has answer, but LLM didn't use it)
- Contradictions (answer says X, context says not-X)
- Context relevance (retrieved contexts don't match query)

**Constraint**: No vector DB access. Only query + retrieved contexts + generated answer.

---

## First-Principles Analysis

### What Information Do We Have?
1. Original query
2. Retrieved contexts (what RAG fetched)
3. Generated answer (what LLM produced)

### What Can We Infer Without Vector DB?

| Diagnostic | Detection Method | Signal |
|------------|-----------------|--------|
| **Coverage Gap** | Semantic overlap between answer and contexts | Answer discusses topics not in retrieved text |
| **Hallucination** | Entity/claim extraction + verification | Claims in answer unsupported by any context |
| **Contradiction** | LLM-based entailment check | Answer asserts X, context explicitly states not-X |
| **Context Irrelevance** | Query-context semantic similarity | Retrieved text doesn't address query intent |
| **Context Redundancy** | Cross-context similarity | Multiple contexts say same thing (retrieval waste) |
| **Attribution Failure** | Answer-context alignment | Answer is correct but can't be traced to specific context |

### What We CANNOT Detect (Acknowledged Limitations)
- Whether better contexts exist in the vector DB
- Retrieval accuracy (precision@k)
- Ground truth comparison (no golden docs)

---

## Proposed Architecture

### New Module: `surface_diagnostics.py`

```
src/ragtriage/
├── surface_diagnostics/
│   ├── __init__.py
│   ├── analyzer.py          # Main SurfaceDiagnostics class
│   ├── coverage.py          # Coverage scoring
│   ├── contradiction.py     # Contradiction detection
│   ├── hallucination.py     # Hallucination detection
│   └── relevance.py         # Context relevance scoring
```

### Core Class: `SurfaceDiagnostics`

**Responsibility**: Analyze (query, contexts[], answer) tuple and return diagnostic report.

**Methods**:

| Method | Purpose | Implementation |
|--------|---------|----------------|
| `analyze_coverage()` | What % of answer is grounded in context? | Embedding similarity + entity overlap |
| `detect_contradictions()` | Does answer conflict with context? | LLM entailment check (NLP) |
| `detect_hallucinations()` | What claims lack support? | NER + claim verification per context |
| `score_context_relevance()` | Do contexts match query? | Query-context embedding similarity |
| `check_context_redundancy()` | Are contexts too similar? | Pairwise context similarity |
| `full_diagnostic()` | Run all checks, return report | Orchestrates above + aggregates |

### Diagnostic Report Schema

```json
{
  "coverage": {
    "score": 0.72,              // 0-1, semantic coverage
    "ungrounded_segments": ["..."],  // Answer parts not in context
    "well_supported_segments": ["..."]
  },
  "contradictions": [
    {
      "answer_claim": "Users can cancel anytime",
      "context_evidence": "Cancellations require 30-day notice",
      "severity": "high"
    }
  ],
  "hallucinations": {
    "score": 0.15,              // % of answer hallucinated
    "unsupported_claims": ["..."],
    "suspicious_entities": ["..."]
  },
  "context_quality": {
    "relevance_scores": [0.9, 0.4, 0.3],  // Per-context relevance
    "avg_relevance": 0.53,
    "redundancy": 0.8,          // 1.0 = all contexts identical
    "retrieval_gaps": ["topic X not covered"]
  },
  "overall_diagnosis": {
    "primary_issue": "poor_coverage",  // hallucination | contradiction | poor_retrieval | good
    "confidence": 0.85,
    "recommended_action": "DOC_UPDATE"  // or DOC_WRITE if coverage gap
  }
}
```

---

## Implementation Details

### 1. Coverage Analysis

**Technique**: Hybrid semantic + lexical

```python
def analyze_coverage(answer, contexts):
    # Semantic: Embed answer chunks, compare to context chunks
    answer_chunks = chunk_text(answer)
    context_chunks = chunk_text("\n".join(contexts))
    
    # For each answer chunk, find max similarity to any context chunk
    coverage_scores = []
    for a_chunk in answer_chunks:
        similarities = [cosine_sim(embed(a_chunk), embed(c_chunk)) 
                       for c_chunk in context_chunks]
        coverage_scores.append(max(similarities))
    
    # Threshold: >0.7 = covered, <0.5 = ungrounded
    coverage = mean(coverage_scores)
    ungrounded = [chunks[i] for i, s in enumerate(coverage_scores) if s < 0.5]
    
    return {"score": coverage, "ungrounded": ungrounded}
```

**Why this works**: Even if answer paraphrases context, embeddings capture semantic similarity.

### 2. Contradiction Detection

**Technique**: LLM-based natural language inference

```python
CONTRADICTION_PROMPT = """
Given CONTEXT and ANSWER, determine if ANSWER contradicts CONTEXT.

Contradiction = Answer states X, Context explicitly states not-X or incompatible fact.
Neutral = Answer adds info not in Context (not contradiction, just extension)
Entailment = Answer is supported by Context

Return JSON:
{
  "relation": "contradiction" | "neutral" | "entailment",
  "confidence": 0-1,
  "explanation": "..."
}
"""

# Check answer against each context
for context in contexts:
    result = llm_check(context, answer)
    if result["relation"] == "contradiction":
        contradictions.append(result)
```

**Why LLM**: Contradictions are often semantic, not lexical. "No cancellation fee" vs "$10 cancellation charge" requires understanding, not string matching.

### 3. Hallucination Detection

**Technique**: Entity extraction + verification

```python
def detect_hallucinations(answer, contexts):
    # Step 1: Extract entities and claims from answer
    entities = extract_entities(answer)  # NER
    claims = extract_claims(answer)      # Simple NLP: subject-verb-object
    
    # Step 2: Check if each exists in any context
    unsupported = []
    for claim in claims:
        # Semantic search for claim in contexts
        claim_embedding = embed(claim)
        max_sim = max([cosine_sim(claim_embedding, embed(ctx)) 
                      for ctx in contexts])
        if max_sim < 0.6:
            unsupported.append(claim)
    
    hallucination_score = len(unsupported) / len(claims)
    return {"score": hallucination_score, "claims": unsupported}
```

**Edge case**: Answer synthesizes from multiple contexts. This is valid, not hallucination. Solution: Check semantic similarity, not exact match.

### 4. Context Relevance

**Technique**: Query-context embedding similarity

```python
def score_context_relevance(query, contexts):
    query_embed = embed(query)
    scores = []
    for ctx in contexts:
        # Use first sentence or chunk for context embedding
        ctx_preview = ctx[:200]
        ctx_embed = embed(ctx_preview)
        scores.append(cosine_sim(query_embed, ctx_embed))
    
    # Low scores = retrieval fetched irrelevant docs
    return {"scores": scores, "avg": mean(scores)}
```

**Interpretation**: 
- Score > 0.7: Context is relevant
- Score 0.4-0.7: Marginally relevant
- Score < 0.4: Retrieval failure (wrong docs fetched)

### 5. Context Redundancy

**Technique**: Pairwise similarity between contexts

```python
def check_redundancy(contexts):
    if len(contexts) < 2:
        return 0.0
    
    similarities = []
    for i, ctx1 in enumerate(contexts):
        for ctx2 in contexts[i+1:]:
            sim = cosine_sim(embed(ctx1), embed(ctx2))
            similarities.append(sim)
    
    # High avg similarity = contexts are redundant
    return mean(similarities)
```

**Interpretation**: Redundancy > 0.8 suggests retrieval is fetching the same info multiple times (waste of context window).

---

## Integration Points

### 1. CLI Extension

Add flag: `--surface-diagnostics` or include by default in evaluation

```bash
uv run ragtriage-eval --surface-diagnostics -i data.jsonl
```

### 2. Report Enhancement

Add section to `report.md`:

```markdown
## Surface Diagnostics

| Query Pattern | Coverage | Hallucination | Contradiction | Primary Issue |
|--------------|----------|---------------|---------------|---------------|
| Cancel subscription | 45% | 15% | None | Poor coverage |
| Export timesheet | 92% | 2% | None | Good |

### Top Coverage Gaps
1. "How do I delete my account?" - Answer invents 30-day policy not in context

### Contradictions Detected
1. Query #23: Answer says "no fee", context says "$10 charge"
```

### 3. Action Item Refinement

Use diagnostics to improve action classification:

| Current | With Diagnostics | New Action |
|---------|------------------|------------|
| DOC_UPDATE | Coverage 95%, but hallucinated detail | DOC_UPDATE (fix prompt) |
| DOC_UPDATE | Coverage 30%, contexts irrelevant | DOC_WRITE (retrieval fails) |
| partial | Contradiction detected | DOC_UPDATE + escalate |

---

## Dependencies

**New**:
- `spacy` or `transformers` for NER (entity extraction)
- Already have: `sentence-transformers` (embeddings), `openai` (LLM checks)

**No new API costs for basic diagnostics** (uses local embeddings). Optional LLM calls for contradiction detection (~$0.001/query).

---

## Success Criteria

1. **Coverage score** correlates with human judgment of "did answer use context?" (target: >0.8 correlation)
2. **Contradiction detection** catches obvious errors (e.g., "24/7 support" vs "business hours only")
3. **Hallucination flag** reduces false positives (distinguish synthesis from invention)
4. **Zero vector DB queries** (constraint satisfied)
5. **Execution time** < 500ms per query (embedding-based, no heavy LLM)

---

## Phased Implementation

**Phase 1**: Coverage + Context Relevance (embeddings only, fast)
**Phase 2**: Hallucination Detection (entity verification)
**Phase 3**: Contradiction Detection (LLM-based, optional flag)
**Phase 4**: Report integration + visualization

---

## Open Questions

1. **Threshold tuning**: What coverage score triggers "poor coverage"? (0.5? 0.7?)
2. **Synthesis vs Hallucination**: How to distinguish valid synthesis from invention?
3. **Multi-hop reasoning**: Answer combines context A + B to infer C. Is C hallucinated?
4. **Contradiction severity**: Should we auto-escalate contradictions as high-priority?

---

Ready for review. What's your take on the approach? Any red flags or additions?