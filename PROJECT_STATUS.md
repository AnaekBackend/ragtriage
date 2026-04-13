# ragtriage Project Status

## What We've Built

**ragtriage** is an open-source tool that turns RAG evaluation from abstract metrics into actionable content backlog. Instead of "faithfulness: 0.72", you get "Write a cancellation article, update 12 docs, ignore 45 spam messages."

## Current State (v0.1.0-alpha)

### ✅ What's Working

1. **Real Dataset**: 1,144 evaluated support queries from AttendanceBot
   - 6 months of production data
   - Already categorized and analyzed
   - Can be used for demos and testing

2. **Core Evaluation Pipeline**:
   - `eval_rag.py` - Evaluates RAG responses with 5-dimension scoring
   - `analyze_rag_final.py` - LLM-based categorization and gap analysis
   - Lane classification (incident/understanding/workflow/spam)
   - Category taxonomy (BILLING, LEAVE, TIMESHEET, etc.)
   - Action classification (doc_write, doc_update)

3. **Documentation**:
   - Comprehensive README with philosophy and approach
   - Architecture guide explaining the three-lane system
   - Configuration examples
   - Usage examples
   - Contributing guidelines

4. **Example Outputs**:
   - `cs_action_items.csv` - Full dataset of 236 action items
   - `rag_root_cause_analysis.md` - Executive summary report
   - Shows what the tool actually produces

### 📊 Real Results from AttendanceBot

From 1,144 support queries over 6 months:

| Metric | Value |
|--------|-------|
| Total queries | 1,144 |
| Spam/workflow | 453 (40%) |
| **Real questions** | 691 (60%) |
| Well answered | 401 (58%) |
| **Partially answered** | **290 (42%)** ← Improvement opportunity |

**Top findings:**
- Billing cancellation: Most requested new article (5+ queries/week)
- Timesheet troubleshooting: Needs doc updates (22 queries/month)
- Manager permissions: Confusing UI/docs (12 queries/month)

### 🏗️ Project Structure

```
ragtriage/
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── src/ragtriage/               # Core package
│   ├── __init__.py
│   ├── eval_rag.py             # RAG evaluation engine
│   ├── analyze_rag_final.py    # Gap analysis & categorization
│   ├── app.py                  # FastAPI server (optional)
│   └── build_index.py          # Document indexing
│
├── data/                        # Sample data
│   └── rag_evaluation.jsonl    # 1,144 evaluated queries
│
├── examples/                    # Example outputs
│   ├── cs_action_items.csv     # Action items (236 rows)
│   ├── rag_root_cause_analysis.md  # Executive report
│   ├── basic_usage.py          # Usage example
│   └── ragtriage.yaml          # Configuration template
│
├── docs/                        # Documentation
│   └── architecture.md         # System architecture
│
└── tests/                       # Test suite (empty)
```

## What's Next

### Before Public Release

1. **Code Cleanup**:
   - Remove AttendanceBot-specific hardcoding
   - Make categories configurable
   - Add proper CLI interface
   - Add comprehensive tests

2. **Documentation**:
   - Installation guide
   - Quick start tutorial
   - API reference
   - Case studies from beta users

3. **Features**:
   - Support for Claude, local LLMs
   - Freshdesk/Zendesk API integration
   - Trend analysis (month-over-month)
   - Notion/Jira export

4. **Examples**:
   - Synthetic dataset for testing
   - Jupyter notebook walkthrough
   - Video demo

### Beta Testing

We need 3-5 design partners to:
- Run ragtriage on their support data
- Provide feedback on categorization accuracy
- Help refine the taxonomy for different domains
- Test the action items with their CS teams

## How to Use Right Now

```bash
# Clone repo
git clone https://github.com/yourusername/ragtriage.git
cd ragtriage

# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY=your-key

# Evaluate your queries (WIP - needs CLI interface)
python src/ragtriage/eval_rag.py --help

# See example outputs
cat examples/rag_root_cause_analysis.md
open examples/cs_action_items.csv
```

## Differentiation

| Tool | Output | Actionable? |
|------|--------|-------------|
| RAGAS | Scores (0-1) | ❌ |
| RAGChecker | Precision/Recall | ❌ |
| DeepEval | Test results | ⚠️ |
| **ragtriage** | **Doc write/update list** | **✅** |

## Vision

**Short-term**: Open source project that CS teams actually use
**Medium-term**: Hosted service with dashboards and integrations
**Long-term**: Autonomous documentation maintenance - agent suggests, human approves, system updates

## Contact

- GitHub Issues: Bug reports, feature requests
- Discussions: General questions
- Email: your.email@example.com

---

**Status**: Ready for beta testing with 3-5 design partners. Not yet ready for public release (needs CLI, tests, cleanup).

**Next milestone**: Working CLI that anyone can run on their data.
