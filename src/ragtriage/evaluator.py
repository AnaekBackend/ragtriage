"""RAG evaluation with 5-dimension scoring."""

import json
import os
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


class RAGEvaluator:
    """Evaluate RAG responses across 5 dimensions."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize evaluator.
        
        Args:
            model: OpenAI model to use for evaluation
        """
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
    
    def evaluate_answer(
        self,
        query: str,
        contexts: List[str],
        generated_answer: str,
    ) -> Dict:
        """Evaluate a single RAG response.
        
        Returns dict with:
            - scores: Dict of 5 dimension scores (1-5)
            - overall_score: Average (1-5)
            - why_failed: Explanation if score < 3
            - bucket: well_answered, partial, or content_gap
        """
        # Construct evaluation prompt
        system_prompt = """You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.

Evaluate the answer across 5 dimensions (1-5 scale):
1. Correctness: Is the information accurate and true?
2. Completeness: Does it answer all parts of the question?
3. Context Usage: Does it use the retrieved contexts effectively?
4. Clarity: Is it clear and easy to understand?
5. Conciseness: Is it appropriately brief without omitting key info?

Return JSON with:
{
    "scores": {
        "correctness": 1-5,
        "completeness": 1-5,
        "context_usage": 1-5,
        "clarity": 1-5,
        "conciseness": 1-5
    },
    "overall_score": average,
    "bucket": "well_answered" (>=3) or "partial" (1-2 with contexts) or "content_gap" (1-2 without relevant contexts),
    "why_failed": "explanation if score < 3"
}"""

        user_prompt = f"""Query: {query}

Retrieved Contexts:
{chr(10).join(f"- {ctx[:500]}..." for ctx in contexts) if contexts else "No contexts retrieved"}

Generated Answer: {generated_answer}

Evaluate this RAG response."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Ensure all required fields
            if "scores" not in result:
                result["scores"] = {
                    "correctness": 1,
                    "completeness": 1,
                    "context_usage": 1,
                    "clarity": 1,
                    "conciseness": 1
                }
            
            if "overall_score" not in result:
                scores = result["scores"]
                result["overall_score"] = sum(scores.values()) / len(scores)
            
            return result
            
        except Exception as e:
            return {
                "scores": {
                    "correctness": 1,
                    "completeness": 1,
                    "context_usage": 1,
                    "clarity": 1,
                    "conciseness": 1
                },
                "overall_score": 1,
                "bucket": "error",
                "why_failed": f"Evaluation error: {str(e)}"
            }
    
    def evaluate_dataset(
        self,
        queries: List[Dict],
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """Evaluate a dataset of queries.
        
        Args:
            queries: List of dicts with 'query', 'contexts', 'generated_answer'
            output_path: Optional path to save results
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for item in tqdm(queries, desc="Evaluating queries"):
            result = self.evaluate_answer(
                query=item["query"],
                contexts=item.get("contexts", []),
                generated_answer=item.get("generated_answer", "")
            )
            
            # Combine with original data
            full_result = {
                **item,
                "evaluation": result
            }
            results.append(full_result)
            
            # Save incrementally
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
        
        return results
