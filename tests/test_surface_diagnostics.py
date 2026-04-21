"""Tests for surface diagnostics module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from ragtriage.surface_diagnostics.analyzer import SurfaceDiagnostics


class TestSurfaceDiagnostics:
    """Test surface diagnostics analyzer."""

    def test_initialization(self):
        """Test diagnostics initializes correctly."""
        diag = SurfaceDiagnostics()
        assert diag.coverage_threshold == 0.5
        assert diag.relevance_threshold == 0.4
        assert diag.embedding_model is not None

    def test_chunk_text_short(self):
        """Test chunking short text returns single chunk."""
        diag = SurfaceDiagnostics()
        text = "Short text"
        chunks = diag.chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long(self):
        """Test chunking long text creates multiple chunks."""
        diag = SurfaceDiagnostics()
        text = " ".join(["word"] * 100)
        chunks = diag.chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        # Verify overlap
        assert len(chunks[0].split()) == 30

    def test_analyze_coverage_no_contexts(self):
        """Test coverage analysis with no contexts."""
        diag = SurfaceDiagnostics()
        result = diag.analyze_coverage("Some answer", [])
        
        assert result["score"] == 0.0
        assert len(result["ungrounded_segments"]) == 1
        assert "No contexts provided" in result["explanation"]

    def test_analyze_coverage_empty_answer(self):
        """Test coverage analysis with empty answer."""
        diag = SurfaceDiagnostics()
        result = diag.analyze_coverage("", ["Some context"])
        
        assert result["score"] == 0.0
        assert len(result["ungrounded_segments"]) == 0

    def test_analyze_coverage_perfect_match(self):
        """Test coverage when answer matches context closely."""
        diag = SurfaceDiagnostics()
        answer = "To set up auto-punch, go to Settings."
        contexts = ["To set up auto-punch, go to Settings and configure work hours."]
        
        result = diag.analyze_coverage(answer, contexts)
        
        # Should have decent coverage since semantic content matches
        assert result["score"] > 0.3
        assert result["score"] <= 1.0
        assert "explanation" in result

    def test_analyze_coverage_no_overlap(self):
        """Test coverage when answer has no relation to contexts."""
        diag = SurfaceDiagnostics()
        answer = "The weather is sunny today."
        contexts = ["To cancel subscription, go to billing settings."]
        
        result = diag.analyze_coverage(answer, contexts)
        
        # Should have low coverage
        assert result["score"] < 0.4
        assert len(result["ungrounded_segments"]) > 0

    def test_score_context_relevance_no_contexts(self):
        """Test relevance scoring with no contexts."""
        diag = SurfaceDiagnostics()
        result = diag.score_context_relevance("query", [])
        
        assert result["avg_relevance"] == 0.0
        assert len(result["scores"]) == 0
        assert "No contexts provided" in result["explanation"]

    def test_score_context_relevance_high_match(self):
        """Test relevance when context matches query."""
        diag = SurfaceDiagnostics()
        query = "How do I cancel my subscription?"
        contexts = ["To cancel your subscription, go to Settings > Billing"]
        
        result = diag.score_context_relevance(query, contexts)
        
        # Should have high relevance
        assert result["avg_relevance"] > 0.5
        assert result["max_relevance"] > 0.5
        assert len(result["scores"]) == 1

    def test_score_context_relevance_low_match(self):
        """Test relevance when context doesn't match query."""
        diag = SurfaceDiagnostics()
        query = "How do I cancel my subscription?"
        contexts = ["To set up auto-punch, configure work hours in Settings."]
        
        result = diag.score_context_relevance(query, contexts)
        
        # Should have lower relevance
        assert result["avg_relevance"] < 0.7  # Not completely unrelated but not matching

    def test_score_context_relevance_multiple_contexts(self):
        """Test relevance with multiple contexts of varying relevance."""
        diag = SurfaceDiagnostics()
        query = "How do I export timesheet data?"
        contexts = [
            "To export timesheet data, go to Reports.",  # High relevance
            "Auto-punch allows automatic clock in.",  # Low relevance
        ]
        
        result = diag.score_context_relevance(query, contexts)
        
        assert len(result["scores"]) == 2
        assert result["max_relevance"] > result["min_relevance"]
        assert result["irrelevant_count"] >= 0

    def test_detect_contradictions_no_contexts(self):
        """Test contradiction detection with no contexts."""
        diag = SurfaceDiagnostics()
        result = diag.detect_contradictions("answer", [])
        
        assert result["contradiction_detected"] is False
        assert len(result["contradictions"]) == 0

    def test_detect_contradictions_no_answer(self):
        """Test contradiction detection with empty answer."""
        diag = SurfaceDiagnostics()
        result = diag.detect_contradictions("", ["context"])
        
        assert result["contradiction_detected"] is False

    @patch.object(SurfaceDiagnostics, '_check_contradiction_llm')
    def test_detect_contradictions_with_contradiction(self, mock_check):
        """Test contradiction detection finds contradictions."""
        mock_check.return_value = {
            "relation": "contradiction",
            "confidence": 0.9,
            "answer_claim": "No cancellation fee",
            "context_claim": "$10 cancellation fee",
            "explanation": "Answer says no fee, context says $10"
        }
        
        diag = SurfaceDiagnostics()
        answer = "You can cancel anytime with no fee."
        contexts = ["A $10 cancellation fee applies."]
        
        result = diag.detect_contradictions(answer, contexts)
        
        assert result["contradiction_detected"] is True
        assert len(result["contradictions"]) > 0

    @patch.object(SurfaceDiagnostics, '_check_contradiction_llm')
    def test_detect_contradictions_skips_distant_contexts(self, mock_check):
        """Test contradiction detection skips semantically distant contexts."""
        mock_check.return_value = {"relation": "neutral", "confidence": 0.5}
        
        diag = SurfaceDiagnostics()
        answer = "The weather is sunny."
        contexts = ["To cancel subscription, go to billing."]  # Completely different topic
        
        result = diag.detect_contradictions(answer, contexts)
        
        # Should skip due to low semantic similarity
        assert result["contradiction_detected"] is False
        mock_check.assert_not_called()

    def test_generate_diagnosis_retrieval_failure(self):
        """Test diagnosis for retrieval failure pattern."""
        diag = SurfaceDiagnostics()
        
        coverage = {"score": 0.2}
        relevance = {"avg_relevance": 0.3, "irrelevant_count": 2}
        contradictions = {"contradiction_detected": False}
        
        diagnosis = diag._generate_diagnosis(coverage, relevance, contradictions)
        
        assert diagnosis["primary_issue"] == "retrieval_failure"
        assert diagnosis["recommended_action"] == "DOC_WRITE"
        assert "Low coverage" in diagnosis["explanation"]

    def test_generate_diagnosis_poor_generation(self):
        """Test diagnosis for poor generation pattern."""
        diag = SurfaceDiagnostics()
        
        coverage = {"score": 0.3}  # Low coverage
        relevance = {"avg_relevance": 0.6, "irrelevant_count": 0}  # But relevant contexts
        contradictions = {"contradiction_detected": False}
        
        diagnosis = diag._generate_diagnosis(coverage, relevance, contradictions)
        
        assert diagnosis["primary_issue"] == "poor_generation"
        assert diagnosis["recommended_action"] == "DOC_UPDATE"
        assert "Relevant contexts retrieved" in diagnosis["explanation"]

    def test_generate_diagnosis_contradiction(self):
        """Test diagnosis when contradiction detected."""
        diag = SurfaceDiagnostics()
        
        coverage = {"score": 0.5}
        relevance = {"avg_relevance": 0.5, "irrelevant_count": 0}
        contradictions = {"contradiction_detected": True}
        
        diagnosis = diag._generate_diagnosis(coverage, relevance, contradictions)
        
        assert diagnosis["primary_issue"] == "contradiction"
        assert diagnosis["recommended_action"] == "DOC_UPDATE"
        assert "contradicts" in diagnosis["explanation"].lower()

    def test_generate_diagnosis_good(self):
        """Test diagnosis for good answer."""
        diag = SurfaceDiagnostics()
        
        coverage = {"score": 0.8}
        relevance = {"avg_relevance": 0.8, "irrelevant_count": 0}
        contradictions = {"contradiction_detected": False}
        
        diagnosis = diag._generate_diagnosis(coverage, relevance, contradictions)
        
        assert diagnosis["primary_issue"] == "good"
        assert diagnosis["recommended_action"] == "NONE"

    def test_generate_diagnosis_noisy_retrieval(self):
        """Test diagnosis for noisy retrieval."""
        diag = SurfaceDiagnostics()
        
        coverage = {"score": 0.5}
        relevance = {"avg_relevance": 0.5, "irrelevant_count": 3}
        contradictions = {"contradiction_detected": False}
        
        diagnosis = diag._generate_diagnosis(coverage, relevance, contradictions)
        
        assert diagnosis["primary_issue"] == "noisy_retrieval"
        assert "irrelevant" in diagnosis["explanation"].lower()

    def test_full_diagnostic_integration(self):
        """Test full diagnostic runs all checks."""
        diag = SurfaceDiagnostics()
        
        query = "How do I cancel?"
        contexts = ["To cancel, go to Settings > Billing"]
        answer = "Go to Settings > Billing to cancel."
        
        result = diag.full_diagnostic(query, contexts, answer)
        
        assert "coverage" in result
        assert "context_relevance" in result
        assert "contradictions" in result
        assert "overall_diagnosis" in result
        
        # Should have reasonable scores
        assert 0 <= result["coverage"]["score"] <= 1
        assert 0 <= result["context_relevance"]["avg_relevance"] <= 1

    def test_full_diagnostic_low_coverage(self):
        """Test full diagnostic identifies low coverage."""
        diag = SurfaceDiagnostics()
        
        query = "How do I migrate data?"
        contexts = ["Migration is available on Enterprise plans."]
        answer = "Contact support for detailed migration steps including API keys and data format conversions."  # Not in context
        
        result = diag.full_diagnostic(query, contexts, answer)
        
        # Should identify lower coverage (answer has details not in context)
        assert result["coverage"]["score"] < 0.8  # Should be less than perfect
        diagnosis = result["overall_diagnosis"]
        assert diagnosis["primary_issue"] in ["retrieval_failure", "partial_coverage", "poor_generation", "noisy_retrieval"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
