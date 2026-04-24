"""Analyze evaluated queries and generate actionable insights."""

import json
import os
from collections import defaultdict
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm

from .surface_diagnostics import SurfaceDiagnostics


LANE_PROMPT = """Classify this support query into one of three lanes:

LANES:
1. UNDERSTANDING - User wants to know HOW to do something (how-to, setup, configuration, integration)
2. INCIDENT - User reports something is BROKEN (errors, wrong data, not working, bugs)
3. SPAM - Marketing pitches, sales outreach, gibberish

Return JSON:
{
    "lane": "UNDERSTANDING" | "INCIDENT" | "SPAM",
    "confidence": "high" | "medium" | "low",
    "reason": "brief explanation"
}"""

CATEGORIZATION_PROMPT = """Categorize this UNDERSTANDING query by topic.

CATEGORIES (choose the best fit):
- BILLING: pricing, plans, payment, invoices, cancellation, refunds
- LEAVE: vacation, sick leave, PTO, time off, leave balance, accrual
- TIMESHEET: hours, logging time, punch in/out, time tracking, approvals
- MANAGER: admin permissions, hierarchy, team setup, approvals
- INTEGRATIONS: API, webhooks, third-party tools, exports
- SETUP: onboarding, installation, adding users, configuration
- REPORTS: analytics, data export, custom reports
- NOTIFICATIONS: reminders, alerts, email settings
- GENERAL: other questions

Return JSON:
{
    "category": "BILLING" | "LEAVE" | "TIMESHEET" | "MANAGER" | "INTEGRATIONS" | "SETUP" | "REPORTS" | "NOTIFICATIONS" | "GENERAL",
    "topic": "3-5 word specific topic",
    "confidence": "high" | "medium" | "low"
}"""

ACTION_PROMPT = """Given this UNDERSTANDING query with a PARTIAL answer, determine the action needed.

Based on the retrieved contexts and the answer quality, decide:
- DOC_UPDATE: Right documents retrieved but answer incomplete/wrong → Update existing docs
- DOC_WRITE: No relevant documents retrieved OR answer completely missing → Write new article

**IMPORTANT:**
- For DOC_UPDATE: The target_article MUST be the title of the existing document that needs updating (from Retrieved Contexts)
- For DOC_WRITE: The target_article should be a descriptive name for the new article to create

SURFACE DIAGNOSTICS (evidence-based signals):
{coverage_info}

Return JSON:
{
    "action": "DOC_UPDATE" | "DOC_WRITE",
    "target_article": "For DOC_UPDATE: exact title of existing doc to update. For DOC_WRITE: descriptive name for new article",
    "gap": "Specifically what info is missing or wrong in the existing doc (DOC_UPDATE) or what new doc should cover (DOC_WRITE)",
    "reason": "Brief justification"
}"""


class QueryAnalyzer:
    """Analyze queries and generate action items with surface diagnostics."""
    
    def __init__(self, model: str = "gpt-4o-mini", use_diagnostics: bool = True):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.use_diagnostics = use_diagnostics
        if use_diagnostics:
            self.diagnostics = SurfaceDiagnostics()
    
    def classify_lane(self, query: str) -> Dict:
        """Classify query into lane (understanding/incident/spam)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LANE_PROMPT},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"lane": "UNKNOWN", "confidence": "low", "reason": str(e)}
    
    def categorize(self, query: str) -> Dict:
        """Categorize understanding query by topic."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CATEGORIZATION_PROMPT},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"category": "GENERAL", "topic": "unknown", "confidence": "low"}
    
    def run_diagnostics(
        self,
        query: str,
        contexts: List[str],
        generated_answer: str
    ) -> Dict:
        """Run surface diagnostics if enabled."""
        if not self.use_diagnostics:
            return {}
        
        try:
            return self.diagnostics.full_diagnostic(query, contexts, generated_answer)
        except Exception as e:
            return {
                "error": str(e),
                "coverage": {"score": 0, "explanation": "Diagnostic failed"},
                "context_relevance": {"avg_relevance": 0, "explanation": "Diagnostic failed"},
                "contradictions": {"contradiction_detected": False, "explanation": "Diagnostic failed"},
                "overall_diagnosis": {
                    "primary_issue": "unknown",
                    "recommended_action": "DOC_UPDATE",
                    "explanation": f"Diagnostic error: {str(e)}"
                }
            }
    
    def determine_action(
        self,
        query: str,
        contexts: List[str],
        generated_answer: str,
        evaluation: Dict,
        diagnostics: Dict = None
    ) -> Dict:
        """Determine action for partial answer with diagnostic evidence."""
        
        # Build diagnostic info string for LLM prompt
        coverage_info = "No diagnostics available"
        if diagnostics:
            coverage = diagnostics.get("coverage", {})
            relevance = diagnostics.get("context_relevance", {})
            diagnosis = diagnostics.get("overall_diagnosis", {})
            contradictions = diagnostics.get("contradictions", {})
            
            coverage_info = f"""
- Coverage Score: {coverage.get('score', 'N/A')} (0-1, higher = more answer supported by contexts)
- Context Relevance: {relevance.get('avg_relevance', 'N/A')} (0-1, higher = retrieved docs match query)
- Contradiction Detected: {contradictions.get('contradiction_detected', 'N/A')}
- Diagnostic Diagnosis: {diagnosis.get('primary_issue', 'N/A')} - {diagnosis.get('explanation', 'N/A')}

Use these signals to make an evidence-based decision."""
        
        try:
            prompt = f"""Query: {query}

Retrieved Contexts:
{chr(10).join(f"- {ctx[:300]}..." for ctx in contexts) if contexts else "No contexts retrieved"}

Generated Answer: {generated_answer}

Evaluation: {evaluation.get('why_failed', 'No explanation')}

Determine the action needed."""

            # If we have strong diagnostic signal, use it directly
            if diagnostics:
                diagnosis = diagnostics.get("overall_diagnosis", {})
                signals = diagnosis.get("signals", {})
                coverage_score = signals.get("coverage_score", 0.5)
                avg_relevance = signals.get("avg_relevance", 0.5)
                has_contradiction = signals.get("has_contradiction", False)
                
                # Override with diagnostic-based decision when confidence is high
                if coverage_score < 0.3 and avg_relevance < 0.4:
                    # Strong signal: retrieval failure
                    return {
                        "action": "DOC_WRITE",
                        "target_article": self._infer_article_name(query),
                        "gap": f"Low coverage ({coverage_score:.0%}) + low relevance ({avg_relevance:.0%}). Retrieved contexts don't contain answer.",
                        "reason": f"Evidence-based: Coverage {coverage_score:.0%}, Relevance {avg_relevance:.0%}. No relevant context found.",
                        "diagnostic_override": True
                    }
                elif has_contradiction:
                    return {
                        "action": "DOC_UPDATE",
                        "target_article": self._infer_article_name(query),
                        "gap": "Answer contradicts retrieved context. Document may be outdated or incorrect.",
                        "reason": "Contradiction detected between answer and context. Update required.",
                        "diagnostic_override": True
                    }
            
            # Fall back to LLM-based decision
            action_prompt = ACTION_PROMPT.format(coverage_info=coverage_info)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": action_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            # Post-process: ensure we always have meaningful values
            query_text = query[:100]  # First 100 chars of query
            
            # Fix empty/weak target_article
            if not result.get("target_article") or result.get("target_article") in ["Unknown", "N/A", "", " "]:
                result["target_article"] = self._infer_article_name(query)
            
            # Fix empty/weak gap description  
            if not result.get("gap") or result.get("gap") in ["Unable to determine", "N/A", "", " "]:
                bucket = item.get("evaluation", {}).get("bucket", "partial")
                why_failed = item.get("evaluation", {}).get("why_failed", "")
                
                if bucket == "partial":
                    result["gap"] = f"Partial answer: {why_failed[:100]}" if why_failed else "Answer needs expansion with more details"
                elif bucket == "incorrect":
                    result["gap"] = f"Incorrect answer: {why_failed[:100]}" if why_failed else "Answer contains errors or contradicts documentation"
                else:
                    result["gap"] = "Documentation gap identified - review needed"
            
            # Add diagnostics to result
            if diagnostics:
                result["diagnostics"] = {
                    "coverage_score": diagnostics.get("coverage", {}).get("score"),
                    "relevance_score": diagnostics.get("context_relevance", {}).get("avg_relevance"),
                    "contradiction_detected": diagnostics.get("contradictions", {}).get("contradiction_detected"),
                    "diagnostic_diagnosis": diagnostics.get("overall_diagnosis", {}).get("primary_issue")
                }
            
            return result
            
        except Exception as e:
            # Even on error, provide meaningful fallback values
            bucket = item.get("evaluation", {}).get("bucket", "partial")
            why_failed = item.get("evaluation", {}).get("why_failed", "")
            
            return {
                "action": "DOC_UPDATE",
                "target_article": self._infer_article_name(query),
                "gap": f"Analysis issue ({str(e)[:50]}), but {bucket} answer detected: {why_failed[:80]}",
                "reason": str(e)
            }
    
    def _infer_article_name(self, query: str) -> str:
        """Infer article name from query with improved heuristics."""
        # Remove common question prefixes
        cleaned = query
        prefixes = [
            "How do I ", "How to ", "What is ", "How can I ", "How should I ",
            "Can I ", "How does ", "Where can I ", "Where do I ", "When should I ",
            "How to:", "I need to ", "I want to ", "Help with ", "Guide for "
        ]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):]
                break
        
        # Remove question marks and extra whitespace
        cleaned = cleaned.replace("?", "").strip()
        
        # If very short or empty, use the full query
        if len(cleaned) < 5:
            cleaned = query.replace("?", "").strip()
        
        # Title case for readability
        return cleaned.title() if cleaned else "Documentation Update Needed"
    
    def analyze_results(self, evaluated_results: List[Dict]) -> List[Dict]:
        """Analyze all evaluated results and produce action items.
        
        Returns list with:
            - query, contexts, answer
            - evaluation scores
            - lane (understanding/incident/spam)
            - category (billing/leave/etc)
            - action (doc_update/doc_write)
            - target_article
            - diagnostics (surface diagnostics results)
        """
        analyzed = []
        
        for item in tqdm(evaluated_results, desc="Analyzing queries"):
            query = item["query"]
            evaluation = item.get("evaluation", {})
            bucket = evaluation.get("bucket", "partial")
            
            # Step 1: Classify lane
            lane_result = self.classify_lane(query)
            lane = lane_result.get("lane", "UNKNOWN")
            
            # Only categorize understanding queries
            if lane == "UNDERSTANDING" and bucket == "partial":
                # Run surface diagnostics
                diagnostics = self.run_diagnostics(
                    query=query,
                    contexts=item.get("contexts", []),
                    generated_answer=item.get("generated_answer", "")
                )
                
                cat_result = self.categorize(query)
                action_result = self.determine_action(
                    query=query,
                    contexts=item.get("contexts", []),
                    generated_answer=item.get("generated_answer", ""),
                    evaluation=evaluation,
                    diagnostics=diagnostics
                )
                
                result = {
                    **item,
                    "lane": lane,
                    "category": cat_result.get("category", "GENERAL"),
                    "topic": cat_result.get("topic", "unknown"),
                    "action": action_result.get("action", "DOC_UPDATE"),
                    "target_article": action_result.get("target_article", "Unknown"),
                    "gap": action_result.get("gap", ""),
                    "reason": action_result.get("reason", "")
                }
                
                # Add diagnostics if available
                if diagnostics:
                    result["surface_diagnostics"] = diagnostics
                
                analyzed.append(result)
            else:
                # Include but mark as non-actionable
                analyzed.append({
                    **item,
                    "lane": lane,
                    "category": "N/A",
                    "topic": "N/A",
                    "action": "N/A",
                    "target_article": "N/A",
                    "gap": "N/A",
                    "reason": lane_result.get("reason", "")
                })
        
        return analyzed
