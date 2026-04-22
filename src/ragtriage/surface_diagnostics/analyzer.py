"""Surface diagnostics analyzer for RAG quality assessment."""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class SurfaceDiagnostics:
    """Analyze RAG quality using only query, contexts, and answer.
    
    No vector DB access required. Uses embeddings and LLM checks to detect:
    - Coverage gaps (answer not supported by contexts)
    - Contradictions (answer conflicts with contexts)
    - Context relevance (retrieved docs don't match query)
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        coverage_threshold: float = 0.5,
        relevance_threshold: float = 0.4
    ):
        """Initialize diagnostics.
        
        Args:
            embedding_model: Sentence transformer for semantic similarity
            llm_model: OpenAI model for contradiction detection
            coverage_threshold: Below this = coverage gap
            relevance_threshold: Below this = irrelevant context
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.llm_model = llm_model
        self.coverage_threshold = coverage_threshold
        self.relevance_threshold = relevance_threshold
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required for contradiction detection")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using sentence transformer."""
        return self.embedding_model.encode(texts, convert_to_numpy=True)
    
    def chunk_text(self, text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def analyze_coverage(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict:
        """Calculate semantic coverage of answer by contexts.
        
        Returns coverage score (0-1) and identifies ungrounded segments.
        """
        if not contexts or not answer:
            return {
                "score": 0.0,
                "ungrounded_segments": [answer] if answer else [],
                "well_supported_segments": [],
                "explanation": "No contexts provided or empty answer"
            }
        
        # Chunk the answer
        answer_chunks = self.chunk_text(answer, chunk_size=50, overlap=10)
        
        # Flatten all contexts into chunks
        context_chunks = []
        for ctx in contexts:
            context_chunks.extend(self.chunk_text(ctx, chunk_size=100, overlap=20))
        
        if not context_chunks:
            return {
                "score": 0.0,
                "ungrounded_segments": answer_chunks,
                "well_supported_segments": [],
                "explanation": "No valid context chunks"
            }
        
        # Embed
        answer_embeds = self.embed(answer_chunks)
        context_embeds = self.embed(context_chunks)
        
        # For each answer chunk, find max similarity to any context chunk
        coverage_scores = []
        for a_embed in answer_embeds:
            similarities = np.dot(context_embeds, a_embed) / (
                np.linalg.norm(context_embeds, axis=1) * np.linalg.norm(a_embed)
            )
            max_sim = float(np.max(similarities))
            coverage_scores.append(max_sim)
        
        # Overall coverage is mean of chunk scores
        coverage_score = float(np.mean(coverage_scores))
        
        # Identify ungrounded segments (below threshold)
        ungrounded = [
            answer_chunks[i] for i, score in enumerate(coverage_scores)
            if score < self.coverage_threshold
        ]
        well_supported = [
            answer_chunks[i] for i, score in enumerate(coverage_scores)
            if score >= 0.7  # High confidence threshold
        ]
        
        return {
            "score": round(coverage_score, 2),
            "ungrounded_segments": ungrounded[:3],  # Limit for readability
            "well_supported_segments": well_supported[:3],
            "chunk_scores": [round(s, 2) for s in coverage_scores],
            "explanation": self._explain_coverage(coverage_score, len(ungrounded), len(answer_chunks))
        }
    
    def _explain_coverage(self, score: float, ungrounded: int, total: int) -> str:
        """Generate human-readable coverage explanation."""
        if score >= 0.7:
            return f"High coverage ({score:.0%}). Answer well-supported by contexts."
        elif score >= 0.4:
            return f"Partial coverage ({score:.0%}). {ungrounded}/{total} segments lack support."
        else:
            return f"Low coverage ({score:.0%}). Answer mostly unsupported by retrieved contexts."
    
    def score_context_relevance(
        self,
        query: str,
        contexts: List[str]
    ) -> Dict:
        """Score how relevant each context is to the query.
        
        Returns per-context relevance scores and aggregate metrics.
        """
        if not contexts:
            return {
                "scores": [],
                "avg_relevance": 0.0,
                "max_relevance": 0.0,
                "min_relevance": 0.0,
                "irrelevant_count": 0,
                "explanation": "No contexts provided"
            }
        
        # Embed query and contexts
        query_embed = self.embed([query])[0]
        
        # Use first 200 chars of each context for relevance check
        context_previews = [ctx[:200] for ctx in contexts]
        context_embeds = self.embed(context_previews)
        
        # Calculate cosine similarities
        scores = []
        for ctx_embed in context_embeds:
            sim = float(np.dot(query_embed, ctx_embed) / (
                np.linalg.norm(query_embed) * np.linalg.norm(ctx_embed)
            ))
            scores.append(round(sim, 2))
        
        irrelevant_count = sum(1 for s in scores if s < self.relevance_threshold)
        
        return {
            "scores": scores,
            "avg_relevance": round(float(np.mean(scores)), 2),
            "max_relevance": round(float(np.max(scores)), 2),
            "min_relevance": round(float(np.min(scores)), 2),
            "irrelevant_count": irrelevant_count,
            "explanation": self._explain_relevance(scores, irrelevant_count)
        }
    
    def _explain_relevance(self, scores: List[float], irrelevant: int) -> str:
        """Generate human-readable relevance explanation."""
        avg = np.mean(scores) if scores else 0
        
        if avg >= 0.7:
            return f"High relevance (avg {avg:.2f}). Retrieved contexts match query well."
        elif avg >= 0.4:
            if irrelevant > 0:
                return f"Mixed relevance (avg {avg:.2f}). {irrelevant} context(s) appear off-topic."
            return f"Moderate relevance (avg {avg:.2f}). Contexts partially match query."
        else:
            return f"Low relevance (avg {avg:.2f}). Retrieved contexts may not answer query."
    
    def detect_contradictions(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict:
        """Detect if answer contradicts any context.
        
        Uses LLM-based NLI (Natural Language Inference) check.
        Expensive, so only run on promising candidates.
        """
        if not contexts or not answer:
            return {
                "contradiction_detected": False,
                "contradictions": [],
                "explanation": "No answer or contexts to check"
            }
        
        contradictions = []
        
        # Check against each context
        for i, context in enumerate(contexts):
            # Quick embedding-based pre-filter: if answer and context are 
            # semantically distant, skip expensive LLM check
            answer_embed = self.embed([answer[:200]])[0]
            ctx_embed = self.embed([context[:200]])[0]
            sim = float(np.dot(answer_embed, ctx_embed) / (
                np.linalg.norm(answer_embed) * np.linalg.norm(ctx_embed)
            ))
            
            # Only check for contradictions if there's some semantic overlap
            if sim < 0.3:
                continue
            
            # LLM-based contradiction check
            result = self._check_contradiction_llm(context, answer)
            
            if result.get("relation") == "contradiction" and result.get("confidence", 0) > 0.7:
                contradictions.append({
                    "context_index": i,
                    "context_snippet": context[:150] + "...",
                    "answer_claim": result.get("answer_claim", ""),
                    "context_claim": result.get("context_claim", ""),
                    "explanation": result.get("explanation", ""),
                    "confidence": result.get("confidence", 0)
                })
        
        return {
            "contradiction_detected": len(contradictions) > 0,
            "contradictions": contradictions,
            "explanation": self._explain_contradictions(contradictions)
        }
    
    def _check_contradiction_llm(self, context: str, answer: str) -> Dict:
        """Use LLM to check for contradiction between context and answer."""
        prompt = f"""Given a CONTEXT and an ANSWER, determine their relationship.

CONTEXT: {context[:500]}

ANSWER: {answer[:300]}

Determine if the ANSWER contradicts the CONTEXT.
- Contradiction: ANSWER states X, CONTEXT explicitly states not-X or incompatible fact
- Neutral: ANSWER adds info not in CONTEXT (not contradiction, just extension)  
- Entailment: ANSWER is supported by CONTEXT

Return JSON:
{{
    "relation": "contradiction" | "neutral" | "entailment",
    "confidence": 0.0-1.0,
    "answer_claim": "the specific claim in answer being evaluated",
    "context_claim": "the specific claim in context that relates",
    "explanation": "brief reasoning"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise NLI (Natural Language Inference) evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "relation": "neutral",
                "confidence": 0.0,
                "explanation": f"Error during check: {str(e)}"
            }
    
    def _explain_contradictions(self, contradictions: List[Dict]) -> str:
        """Generate human-readable contradiction explanation."""
        if not contradictions:
            return "No contradictions detected between answer and contexts."
        elif len(contradictions) == 1:
            return f"1 contradiction detected. Answer conflicts with retrieved context."
        else:
            return f"{len(contradictions)} contradictions detected. Answer conflicts with {len(contradictions)} contexts."
    
    def full_diagnostic(
        self,
        query: str,
        contexts: List[str],
        answer: str
    ) -> Dict:
        """Run full surface diagnostic.
        
        Returns comprehensive analysis of RAG quality without vector DB access.
        """
        # Run all diagnostics
        coverage = self.analyze_coverage(answer, contexts)
        relevance = self.score_context_relevance(query, contexts)
        
        # Only run contradiction check if there's some signal
        # (coverage < 0.9 but > 0.2 suggests potential issues)
        if 0.2 < coverage["score"] < 0.9:
            contradictions = self.detect_contradictions(answer, contexts)
        else:
            contradictions = {
                "contradiction_detected": False,
                "contradictions": [],
                "explanation": "Skipped (coverage too low or too high for contradiction check)"
            }
        
        # Generate overall diagnosis
        diagnosis = self._generate_diagnosis(coverage, relevance, contradictions)
        
        return {
            "coverage": coverage,
            "context_relevance": relevance,
            "contradictions": contradictions,
            "overall_diagnosis": diagnosis
        }
    
    def _generate_diagnosis(
        self,
        coverage: Dict,
        relevance: Dict,
        contradictions: Dict
    ) -> Dict:
        """Generate overall diagnosis from individual signals."""
        coverage_score = coverage["score"]
        avg_relevance = relevance["avg_relevance"]
        has_contradiction = contradictions["contradiction_detected"]
        irrelevant_count = relevance["irrelevant_count"]
        
        # Determine primary issue
        if has_contradiction:
            primary_issue = "contradiction"
            confidence = 0.9
            recommended_action = "DOC_UPDATE"
            explanation = "Answer contradicts retrieved context. Document or answer needs correction."
        elif coverage_score < 0.4 and avg_relevance < 0.4:
            primary_issue = "retrieval_failure"
            confidence = 0.85
            recommended_action = "DOC_WRITE"
            explanation = "Low coverage + low relevance. Retrieved contexts don't contain answer. New doc likely needed."
        elif coverage_score < 0.4 and avg_relevance >= 0.5:
            primary_issue = "poor_generation"
            confidence = 0.75
            recommended_action = "DOC_UPDATE"
            explanation = "Relevant contexts retrieved but answer doesn't use them. Check LLM prompt or doc clarity."
        elif irrelevant_count >= 2:
            primary_issue = "noisy_retrieval"
            confidence = 0.7
            recommended_action = "DOC_UPDATE"
            explanation = f"{irrelevant_count} irrelevant contexts retrieved. Tune retrieval or update docs."
        elif coverage_score >= 0.7:
            primary_issue = "good"
            confidence = 0.8
            recommended_action = "NONE"
            explanation = "Good coverage and relevance. Answer well-supported by contexts."
        else:
            primary_issue = "partial_coverage"
            confidence = 0.6
            recommended_action = "DOC_UPDATE"
            explanation = "Partial coverage. Some answer segments unsupported by contexts."
        
        return {
            "primary_issue": primary_issue,
            "confidence": confidence,
            "recommended_action": recommended_action,
            "explanation": explanation,
            "signals": {
                "coverage_score": coverage_score,
                "avg_relevance": avg_relevance,
                "has_contradiction": has_contradiction,
                "irrelevant_contexts": irrelevant_count
            }
        }
