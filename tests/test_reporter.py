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
        
        # Should not repeat the topic header multiple times (topic is title-cased in report)
        topic_count = report.count("Shared Topic")
        assert topic_count == 1, f"Topic should appear once, found {topic_count} times"
