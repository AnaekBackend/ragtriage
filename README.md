# ragtriage

> From "How good is my RAG?" to "What should my team write this week?"

**ragtriage** turns your RAG system's failures into a prioritized content backlog. Stop measuring metrics you can't act on. Start fixing the gaps that matter.

## The Problem

Existing RAG evaluation tools (RAGAS, RAGChecker, DeepEval) tell you:
```
Faithfulness: 0.72
Answer Relevancy: 0.68
Context Precision: 0.65
```

But **what do you actually do with that?**

Meanwhile, your support team gets 500 queries/month and has no idea which docs to write, update, or ignore.

## Our Approach: Triage, Don't Just Score

ragtriage uses a **three-lane system** to process real user queries:

```
Support Queries (100%)
       ↓
┌─────────────────┐
│   Spam Filter   │ → Marketing pitches, gibberish (9%)
└─────────────────┘
       ↓
┌─────────────────┐
│  Lane Classifier│
└─────────────────┘
       ↓
┌──────────┬──────────┬──────────┐
│Incident  │Understand│  Workflow│
│  (14%)   │   (76%)  │  (10%)   │
└──────────┴──────────┴──────────┘
       ↓
Separate    RAG Eval +       "ticket",
for CS      Categorization   "demo",
triage      → Action Items   etc.
```

### Lane 1: Incidents (14%)
Users reporting product bugs: "sick leave not updating", "system error". These need engineering investigation, not better docs.

### Lane 2: Understanding (76%)
Users asking how-to questions: "How do I cancel?", "Can I set up auto punch-out?" These are documentation gaps we can fix.

### Lane 3: Workflow Commands (10%)
Users typing "ticket", "demo", "support". These are bot commands, not RAG questions.

## What You Get

Instead of abstract scores, you get a **CSV file** your CS team can act on today:

| query | category | action | target_article | gap |
|-------|----------|--------|----------------|-----|
| "I want to cancel my subscription" | BILLING | doc_write | "How to Cancel Your Subscription" | No cancellation article exists |
| "How do I set auto punch-out for everyone?" | TIMESHEET | doc_update | "How to set up auto-punch out" | Missing bulk setup instructions |

Plus a **Markdown report** with:
- Executive summary (well answered %, improvement opportunity)
- Top articles to write
- Top articles to update
- Category breakdown

## Real Example: AttendanceBot Case Study

We ran ragtriage on **1,144 real support queries** over 6 months:

| Metric | Count | % |
|--------|-------|---|
| Total queries | 1,144 | 100% |
| Spam/workflow commands | 453 | 40% |
| **Real questions** | **691** | **60%** |
| Well answered | 401 | 58% |
| Partially answered | **290** | **42%** |
| Content gaps | 0 | 0% |

**The 290 partial answers broke down into:**
- 236 understanding issues → Doc updates
- 54 incident reports → Engineering bugs

**Top action items:**
1. Write "How to Cancel Your Subscription" (2 queries/week)
2. Update "How do I assign managers" with permission details (12 queries/month)
3. Add troubleshooting section to timesheet docs (22 queries/month)

**Result:** CS team had a prioritized backlog instead of guessing what to write.

## Why This Is Different

| Tool | Output | Audience | Actionable? |
|------|--------|----------|-------------|
| RAGAS | Scores (0-1) | ML Engineers | ❌ |
| RAGChecker | Precision/Recall | Data Scientists | ❌ |
| DeepEval | Test results | Dev Teams | ⚠️ |
| **ragtriage** | **Doc write/update list** | **CS Teams** | **✅** |

## Installation

```bash
pip install ragtriage
```

Or from source:
```bash
git clone https://github.com/yourusername/ragtriage.git
cd ragtriage
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Export your support queries as JSONL:
```json
{"query": "How do I cancel my subscription?", "timestamp": "2024-01-15"}
{"query": "My sick leave balance is wrong", "timestamp": "2024-01-16"}
```

### 2. Evaluate Against Your RAG

```bash
python -m ragtriage.eval \
  --queries support_queries.jsonl \
  --rag-endpoint http://your-rag.com/ask \
  --output evaluation.jsonl
```

### 3. Categorize and Generate Actions

```bash
python -m ragtriage.analyze \
  --evaluation evaluation.jsonl \
  --output-dir ./output
```

### 4. Review Results

```bash
cat output/action_plan.md
open output/action_items.csv
```

## How It Works

### Step 1: Lane Classification (LLM-based)

Each query is classified into:
- **INCIDENT**: User reports something is broken
- **UNDERSTANDING**: User asks how to do something
- **WORKFLOW**: Command word (ticket, demo, etc.)
- **SPAM**: Marketing, gibberish, off-topic

Uses few-shot prompting with real examples.

### Step 2: RAG Evaluation (for Understanding lane)

For understanding queries, we:
1. Query your actual RAG system
2. Get the answer + retrieved docs
3. Grade quality (1-5) across 5 dimensions:
   - Correctness
   - Completeness
   - Relevance
   - Source attribution
   - Actionability

### Step 3: Gap Analysis (LLM-based)

For partially answered queries, we determine:
- **Category**: BILLING, LEAVE, TIMESHEET, etc.
- **Action**: doc_write (new article) or doc_update (improve existing)
- **Target Article**: Which doc to create/update
- **Gap Description**: What's missing

### Step 4: Prioritization

Results are sorted by:
1. Category (revenue-critical first: billing > timesheet > reports)
2. Query volume (more asks = higher priority)
3. RAG score (lower score = more urgent)

## Configuration

Create `ragtriage.yaml`:

```yaml
# RAG System
rag:
  endpoint: http://localhost:8000/ask
  timeout: 30

# Lane Classification
lanes:
  # Few-shot examples for each lane
  incident_examples:
    - "My timesheet totals are not accurate"
    - "Sick leave balance is wrong for my team"
  understanding_examples:
    - "How do I cancel my subscription?"
    - "Can I set up auto punch-out?"

# Categories
categories:
  - name: BILLING
    priority: 1
    keywords: ["cancel", "subscription", "billing", "refund"]
  - name: LEAVE
    priority: 2
    keywords: ["leave", "vacation", "sick", "pto"]
  - name: TIMESHEET
    priority: 3
    keywords: ["hours", "timesheet", "punch", "time tracking"]

# Output
output:
  format: ["csv", "markdown"]
  include_examples: 5  # Number of example queries per category
```

## Data Privacy

- All processing is local (your queries never leave your machine)
- Uses your own OpenAI/Azure API keys
- No telemetry or tracking
- Logs stored locally only

## Roadmap

### v0.1 (Current)
- ✅ Lane classification
- ✅ RAG evaluation
- ✅ Gap analysis
- ✅ CSV/MD output

### v0.2 (Planned)
- [ ] Support for more LLM providers (Claude, local models)
- [ ] Custom categorization taxonomy
- [ ] Integration with Freshdesk/Zendesk APIs
- [ ] Incident pattern clustering

### v0.3 (Planned)
- [ ] Trend analysis (compare month-over-month)
- [ ] Agent integration (auto-suggest doc updates)
- [ ] Team config analysis (cross-reference with settings)

## Contributing

This project was born from real frustration maintaining docs at [AttendanceBot](https://attendancebot.com). We evaluated 1,100+ support queries manually before building this.

**Contributions welcome:**
- More lane classification examples
- Better categorization prompts
- Additional output formats
- Case studies from other companies

## License

MIT License - See [LICENSE](LICENSE)

## Acknowledgments

- Inspired by RAGAS, but frustrated by scores we couldn't act on
- Built for CS teams who need answers, not benchmarks
- Thanks to the AttendanceBot team for letting us share real (anonymized) data

---

**Stop measuring your RAG. Start fixing it.**
