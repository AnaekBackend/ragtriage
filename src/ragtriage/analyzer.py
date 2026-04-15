"""Analyze evaluated queries and generate actionable insights."""

import json
import os
from collections import defaultdict
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm


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

Return JSON:
{
    "action": "DOC_UPDATE" | "DOC_WRITE",
    "target_article": "specific article name to update or write",
    "gap": "what information is missing",
    "reason": "why this action was chosen"
}"""


class QueryAnalyzer:
    """Analyze queries and generate action items."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
    
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
    
    def determine_action(
        self,
        query: str,
        contexts: List[str],
        generated_answer: str,
        evaluation: Dict
    ) -> Dict:
        """Determine action for partial answer."""
        try:
            prompt = f"""Query: {query}

Retrieved Contexts:
{chr(10).join(f"- {ctx[:300]}..." for ctx in contexts) if contexts else "No contexts retrieved"}

Generated Answer: {generated_answer}

Evaluation: {evaluation.get('why_failed', 'No explanation')}

Determine the action needed."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ACTION_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "action": "DOC_UPDATE",
                "target_article": "Unknown",
                "gap": "Unable to determine",
                "reason": str(e)
            }
    
    def analyze_results(self, evaluated_results: List[Dict]) -> List[Dict]:
        """Analyze all evaluated results and produce action items.
        
        Returns list with:
            - query, contexts, answer
            - evaluation scores
            - lane (understanding/incident/spam)
            - category (billing/leave/etc)
            - action (doc_update/doc_write)
            - target_article
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
                cat_result = self.categorize(query)
                action_result = self.determine_action(
                    query=query,
                    contexts=item.get("contexts", []),
                    generated_answer=item.get("generated_answer", ""),
                    evaluation=evaluation
                )
                
                analyzed.append({
                    **item,
                    "lane": lane,
                    "category": cat_result.get("category", "GENERAL"),
                    "topic": cat_result.get("topic", "unknown"),
                    "action": action_result.get("action", "DOC_UPDATE"),
                    "target_article": action_result.get("target_article", "Unknown"),
                    "gap": action_result.get("gap", ""),
                    "reason": action_result.get("reason", "")
                })
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
