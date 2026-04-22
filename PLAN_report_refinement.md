# Report Refinement Implementation Plan

> **Goal:** Make the RAGTriage report actionable for CS managers by replacing repetitive query listings with specific content requirements for articles to write/update.

**Current Problem:**
- Report shows article headlines that are just title-cased queries (e.g., "Cancel Membership")
- Then repeats the same queries as bullet points under each headline
- CS managers can't easily assign work because there's no clear "what to write" guidance

**Solution:**
- Group related items by topic/category
- Use the `gap` field to describe what content is needed
- Keep 1-2 example queries for context
- Provide clear "Content to Cover" section derived from gaps

---

## Task 1: Create Test for Report Format

**Objective:** Write tests that verify the new actionable report format.

**Files:**
- Create: `tests/test_reporter.py`

**Step 1: Write failing test for DOC_WRITE section format**

```python
"""Tests for reporter module."""

import pytest
from ragtriage.reporter import ReportGenerator


class TestReportGenerator:
    """Test report generation with actionable format."""

    def test_generate_report_doc_write_shows_gap_not_query(self):
        """DOC_WRITE items should show content gap, not repeat queries."""
        generator = ReportGenerator()
        
        analyzed_results = [
            {
                "query": "How do I cancel my subscription?",
                "lane": "UNDERSTANDING",
                "evaluation": {"bucket": "partial"},
                "action": "DOC_WRITE",
                "target_article": "Subscription Cancellation",
                "gap": "Document cancellation process, refund policy, and timeline",
                "reason": "No relevant documentation found",
                "category": "BILLING",
                "topic": "subscription cancellation",
            },
            {
                "query": "Can I get a refund after canceling?",
                "lane": "UNDERSTANDING", 
                "evaluation": {"bucket": "partial"},
                "action": "DOC_WRITE",
                "target_article": "Subscription Cancellation",
                "gap": "Refund policy details and eligibility criteria",
                "reason": "No relevant documentation found",
                "category": "BILLING",
                "topic": "subscription cancellation",
            }
        ]
        
        report = generator.generate_report(analyzed_results)
        
        # Should group by topic, not list duplicate queries
        assert "## Articles to Write" in report
        # Should show what content to cover (the gap)
        assert "Document cancellation process" in report or "Refund policy" in report
        # Should show category
        assert "BILLING" in report

    def test_generate_report_doc_update_shows_target_and_gap(self):
        """DOC_UPDATE items should show target article and what's missing."""
        generator = ReportGenerator()
        
        analyzed_results = [
            {
                "query": "How do I export to Excel?",
                "lane": "UNDERSTANDING",
                "evaluation": {"bucket": "partial"},
                "action": "DOC_UPDATE",
                "target_article": "Timesheet Export Guide",
                "gap": "Add Excel-specific formatting instructions and column mapping",
                "reason": "Right article retrieved but missing Excel details",
                "category": "REPORTS",
                "topic": "timesheet export",
            }
        ]
        
        report = generator.generate_report(analyzed_results)
        
        # Should show target article to update
        assert "Timesheet Export Guide" in report
        # Should show what to add
        assert "Excel-specific formatting" in report or "column mapping" in report

    def test_generate_report_groups_by_topic(self):
        """Related items should be grouped under single topic header."""
        generator = ReportGenerator()
        
        analyzed_results = [
            {
                "query": "Question 1",
                "lane": "UNDERSTANDING",
                "evaluation": {"bucket": "partial"},
                "action": "DOC_WRITE",
                "target_article": "Topic A",
                "gap": "Gap 1",
                "category": "CATEGORY",
                "topic": "shared topic",
            },
            {
                "query": "Question 2",
                "lane": "UNDERSTANDING",
                "evaluation": {"bucket": "partial"},
                "action": "DOC_WRITE",
                "target_article": "Topic A",
                "gap": "Gap 2",
                "category": "CATEGORY",
                "topic": "shared topic",
            }
        ]
        
        report = generator.generate_report(analyzed_results)
        
        # Should not repeat the topic header multiple times
        topic_count = report.count("shared topic")
        assert topic_count == 1, f"Topic should appear once, found {topic_count} times"
```

**Step 2: Run test to verify failure**

```bash
cd ~/ragtriage
python -m pytest tests/test_reporter.py -v
```

Expected: 3 FAILs — "ModuleNotFoundError" or assertion errors (tests don't exist yet)

**Step 3: Commit**

```bash
git add tests/test_reporter.py
git commit -m "test: add tests for actionable report format"
```

---

## Task 2: Implement Actionable Report Format - DOC_WRITE Section

**Objective:** Rewrite the "Articles to Write" section to show content gaps grouped by topic.

**Files:**
- Modify: `src/ragtriage/reporter.py:87-99`

**Step 1: Create new grouping logic for DOC_WRITE**

Replace lines 87-99 in reporter.py with:

```python
        # Group DOC_WRITE items by (category, topic) for better organization
        write_items = [r for r in action_items if r.get("action") == "DOC_WRITE"]
        write_by_topic = defaultdict(list)
        for item in write_items:
            key = (item.get("category", "GENERAL"), item.get("topic", "Unknown"))
            write_by_topic[key].append(item)
        
        # Sort by count (most questions first)
        sorted_write_topics = sorted(write_by_topic.items(), key=lambda x: -len(x[1]))
```

**Step 2: Replace DOC_WRITE section rendering**

Replace the rendering loop (lines 92-99) with:

```python
        for (category, topic), items in sorted_write_topics[:15]:
            # Aggregate gaps and sample queries
            gaps = [item.get("gap", "") for item in items if item.get("gap")]
            unique_gaps = list(dict.fromkeys(gaps))[:3]  # Deduplicate, max 3
            sample_queries = [item.get("query", "") for item in items[:2]]
            
            report += f"\n#### {topic.title()} ({len(items)} questions)\n\n"
            report += f"**Category:** {category}  \n"
            report += f"**Article Name:** {items[0].get('target_article', topic.title())}\n\n"
            
            report += "**Content to Cover:**\n"
            for gap in unique_gaps:
                report += f"- {gap}\n"
            
            if len(gaps) > 3:
                report += f"- *...and {len(gaps) - 3} more content areas*\n"
            
            report += "\n**Sample Questions:**\n"
            for query in sample_queries:
                short_query = query[:80] + "..." if len(query) > 80 else query
                report += f"- \"{short_query}\"\n"
            
            report += "\n---\n"
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_reporter.py::TestReportGenerator::test_generate_report_doc_write_shows_gap_not_query -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add src/ragtriage/reporter.py
git commit -m "feat: rewrite DOC_WRITE section with content gaps grouped by topic"
```

---

## Task 3: Implement Actionable Report Format - DOC_UPDATE Section

**Objective:** Rewrite the "Articles to Update" section similarly.

**Files:**
- Modify: `src/ragtriage/reporter.py:101-111`

**Step 1: Create grouping logic for DOC_UPDATE**

Replace the DOC_UPDATE section (lines 101-111) with:

```python
        report += f"\n### Articles to Update ({doc_update})\n\n"
        
        # Group DOC_UPDATE items by target_article
        update_items = [r for r in action_items if r.get("action") == "DOC_UPDATE"]
        update_by_article = defaultdict(list)
        for item in update_items:
            key = item.get("target_article", "Unknown")
            update_by_article[key].append(item)
        
        # Sort by count
        sorted_update_articles = sorted(update_by_article.items(), key=lambda x: -len(x[1]))
        
        for article, items in sorted_update_articles[:15]:
            # Aggregate gaps and categories
            gaps = [item.get("gap", "") for item in items if item.get("gap")]
            unique_gaps = list(dict.fromkeys(gaps))[:3]
            categories = list(set(item.get("category", "GENERAL") for item in items))
            sample_queries = [item.get("query", "") for item in items[:2]]
            
            report += f"\n#### {article} ({len(items)} questions)\n\n"
            report += f"**Category:** {', '.join(categories)}\n\n"
            
            report += "**Updates Needed:**\n"
            for gap in unique_gaps:
                report += f"- {gap}\n"
            
            if len(gaps) > 3:
                report += f"- *...and {len(gaps) - 3} more updates needed*\n"
            
            report += "\n**Sample Questions:**\n"
            for query in sample_queries:
                short_query = query[:80] + "..." if len(query) > 80 else query
                report += f"- \"{short_query}\"\n"
            
            report += "\n---\n"
```

**Step 2: Run tests**

```bash
python -m pytest tests/test_reporter.py::TestReportGenerator::test_generate_report_doc_update_shows_target_and_gap -v
python -m pytest tests/test_reporter.py::TestReportGenerator::test_generate_report_groups_by_topic -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add src/ragtriage/reporter.py
git commit -m "feat: rewrite DOC_UPDATE section with specific update requirements"
```

---

## Task 4: Add Import for defaultdict

**Objective:** Ensure defaultdict is imported (should already be there, but verify).

**Files:**
- Modify: `src/ragtriage/reporter.py:1-8`

**Step 1: Verify import exists**

Line 4 should have:
```python
from collections import Counter, defaultdict
```

If missing, add `defaultdict`.

**Step 2: Run all tests**

```bash
python -m pytest tests/test_reporter.py -v
```

Expected: 3 passed

**Step 3: Commit**

```bash
git add src/ragtriage/reporter.py
git commit -m "chore: ensure defaultdict import for reporter"
```

---

## Task 5: Run Existing Tests to Ensure No Regression

**Objective:** Make sure the changes don't break existing functionality.

**Step 1: Run all project tests**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All existing tests should still pass

**Step 2: Fix any failures**

If tests fail due to report format changes, update them to match new format.

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: update tests for new report format"
```

---

## Task 6: Run Project on Sample Data to Verify Output

**Objective:** Generate a report with the sample data to verify the new format looks good.

**Step 1: Run the full pipeline on sample data**

```bash
cd ~/ragtriage
uv run python -m ragtriage.cli evaluate --input data/sample_queries.jsonl --output output/test_report
```

**Step 2: Check the generated report**

```bash
cat output/test_report/report.md | head -100
```

**Verify:**
- [ ] Articles to Write section shows "Content to Cover" with gaps, not repeated queries
- [ ] Items are grouped by topic
- [ ] Sample queries appear but don't dominate
- [ ] Articles to Update shows "Updates Needed" section
- [ ] No more repetitive "Query..." headlines

**Step 3: If needed, make minor adjustments**

If the output needs tweaks (formatting, section headers, etc.):

```bash
# Edit reporter.py
# Re-run test
# Commit fixes
git add src/ragtriage/reporter.py
git commit -m "fix: adjust report formatting based on sample output"
```

---

## Task 7: Final Verification and Commit

**Objective:** Final check and push to branch.

**Step 1: Run full test suite one more time**

```bash
python -m pytest tests/ -v
```

Expected: All pass

**Step 2: Review changes**

```bash
git log --oneline -7
git diff HEAD~7 --stat
```

**Step 3: Create summary commit if needed**

All changes should already be committed task-by-task, but verify.

---

## Expected Final Report Format

The new report should look like:

```markdown
### Articles to Write (452)

#### Subscription Cancellation (15 questions)

**Category:** BILLING  
**Article Name:** How To Cancel My Subscription

**Content to Cover:**
- Document cancellation process, refund policy, and timeline
- Explain how to pause vs cancel subscription
- Cover data retention after cancellation

**Sample Questions:**
- "How do I cancel my subscription?"
- "Can I get a refund after canceling?"

---

#### Auto-Punch Setup (8 questions)

**Category:** TIMESHEET  
**Article Name:** Setting Up Auto-Punch

**Content to Cover:**
- Step-by-step auto-punch configuration
- Troubleshooting automatic clock-in issues

**Sample Questions:**
- "How do I set up auto-punch for my team?"
- "There's an automatic Day Change punch in. How to avoid that?"

---

### Articles to Update (23)

#### Timesheet Export Guide (5 questions)

**Category:** REPORTS

**Updates Needed:**
- Add Excel-specific formatting instructions and column mapping
- Include screenshots of export dialog
- Document CSV vs Excel differences

**Sample Questions:**
- "How do I export timesheet data to Excel?"
- "Can I customize which columns export to Excel?"

---
```

This format gives CS managers:
1. Clear article names to write/update
2. Specific content requirements (gaps)
3. Sample questions for context
4. Grouped by topic for efficient assignment

---

## Summary of Changes

| File | Change |
|------|--------|
| `tests/test_reporter.py` | New tests for actionable format |
| `src/ragtriage/reporter.py` | Rewrite DOC_WRITE/DOC_UPDATE sections |

**Key improvements:**
- Groups by (category, topic) instead of individual queries
- Shows `gap` field as "Content to Cover" / "Updates Needed"
- Lists 1-2 sample queries for context only
- Clear structure for CS managers to assign work
