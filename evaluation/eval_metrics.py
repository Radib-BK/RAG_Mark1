"""
RAG System Evaluation Metrics

This module provides focused evaluation for RAG systems:
- Groundedness: Is the answer supported by retrieved context?
- Relevance: Does the system fetch appropriate documents?
- Cosine similarity for semantic evaluation
- Simple metrics for practical evaluation
- Test set management for Bengali literature content
"""

import os
import json
import math
import statistics
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# Optional imports with fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Essential imports
try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")
    logger.warning("ROUGE not available. Install with: pip install rouge-score")

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
@dataclass
class RAGEvaluationResult:
    """
    Simple evaluation result for RAG system
    """
    question_id: str
    question: str
    predicted_answer: str
    ground_truth_answer: str
    retrieved_context: str
    
    # Core RAG metrics
    groundedness_score: float = 0.0  # Is answer supported by context?
    relevance_score: float = 0.0     # Are retrieved docs relevant to question?
    semantic_similarity: float = 0.0  # Cosine similarity with ground truth
    context_utilization: float = 0.0  # How well context is used
    
    # Additional metrics
    answer_length: int = 0
    context_length: int = 0
    response_time_ms: int = 0
    
    # Human evaluation (optional)
    human_groundedness: Optional[float] = None
    human_relevance: Optional[float] = None
    human_quality: Optional[float] = None

@dataclass
class RAGEvaluationSummary:
    """
    Summary of RAG evaluation results
    """
    total_questions: int
    
    # Average scores
    avg_groundedness: float
    avg_relevance: float  
    avg_semantic_similarity: float
    avg_context_utilization: float
    avg_response_time_ms: float
    
    # Score distributions
    groundedness_distribution: Dict[str, int]  # e.g., {"good": 5, "fair": 3, "poor": 2}
    relevance_distribution: Dict[str, int]
    
    # Performance by question type
    performance_by_type: Dict[str, Dict[str, float]]
    
    evaluation_timestamp: str

class SimpleRAGEvaluator:
    """
    Simple, practical RAG system evaluator focusing on core metrics
    """
    
    def __init__(self, embedder=None):
        """
        Initialize evaluator
        
        Args:
            embedder: Embedding model for semantic similarity
        """
        self.embedder = embedder
        logger.info("Initialized SimpleRAGEvaluator")
    
    def evaluate_groundedness(self, answer: str, context: str) -> float:
        """
        Evaluate if answer is grounded in the provided context
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Groundedness score between 0 and 1
        """
        if not answer or not context:
            return 0.0
        
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Check if answer contains information from context
        answer_words = set(answer_lower.split())
        context_words = set(context_lower.split())
        
        # Remove common Bengali/English stop words
        stop_words = {
            'এই', 'সেই', 'এটি', 'যে', 'তার', 'হয়', 'করে', 'থেকে', 'দিয়ে', 'সাথে',
            'the', 'is', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }
        
        answer_content = answer_words - stop_words
        context_content = context_words - stop_words
        
        if not answer_content:
            return 0.0
        
        # Calculate overlap
        overlap = len(answer_content & context_content)
        answer_coverage = overlap / len(answer_content) if answer_content else 0.0
        
        # Check for specific patterns that indicate grounding
        grounding_bonus = 0.0
        
        # Look for direct quotes or specific facts
        if any(phrase in context_lower for phrase in answer_lower.split('.') if len(phrase) > 10):
            grounding_bonus = 0.2
        
        # Check for number/name preservation
        import re
        answer_numbers = set(re.findall(r'\d+', answer))
        context_numbers = set(re.findall(r'\d+', context))
        if answer_numbers and answer_numbers.issubset(context_numbers):
            grounding_bonus += 0.1
        
        groundedness = min(answer_coverage + grounding_bonus, 1.0)
        return groundedness
    
    def evaluate_relevance(self, question: str, retrieved_context: str) -> float:
        """
        Evaluate relevance of retrieved context to the question
        
        Args:
            question: Original question
            retrieved_context: Retrieved context
            
        Returns:
            Relevance score between 0 and 1
        """
        if not question or not retrieved_context:
            return 0.0
        
        question_lower = question.lower()
        context_lower = retrieved_context.lower()
        
        # Extract key terms from question
        question_words = set(question_lower.split())
        context_words = set(context_lower.split())
        
        # Remove question words (কী, কে, কোথায়, what, who, where, etc.)
        question_stopwords = {
            'কী', 'কি', 'কে', 'কার', 'কোথায়', 'কখন', 'কেন', 'কীভাবে',
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose',
            'কত', 'কতো', 'বছর', 'years', 'old', 'age'  # Don't remove these key terms
        }
        
        # Don't remove all question words - keep content words
        question_content = question_words - {'কী', 'কি', 'কে', 'কার', 'কোথায়', 'কখন', 'কেন', 'কীভাবে',
                                           'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose'}
        
        if not question_content:
            return 0.5  # Neutral score for empty question content
        
        # Calculate how many question terms appear in context
        matching_terms = len(question_content & context_words)
        term_coverage = matching_terms / len(question_content) if question_content else 0.0
        
        # Enhanced relevance patterns for Bengali content
        relevance_bonus = 0.0
        
        # Check if context contains answer-like patterns
        if any(word in context_lower for word in ['উত্তর', 'answer', 'because', 'কারণ']):
            relevance_bonus += 0.1
        
        # Check for topic consistency - be more lenient
        if len(question_content & context_words) >= 1:  # At least one match
            relevance_bonus += 0.2
        
        # Check for name/entity matching (like "অনুপম")
        import re
        question_entities = re.findall(r'[অ-ৱ]+', question)  # Bengali words
        context_entities = re.findall(r'[অ-ৱ]+', retrieved_context)
        
        entity_matches = 0
        for q_entity in question_entities:
            if len(q_entity) > 2 and q_entity in context_entities:  # Avoid short words
                entity_matches += 1
        
        if entity_matches > 0:
            relevance_bonus += min(0.3, entity_matches * 0.1)
        
        relevance = min(term_coverage + relevance_bonus, 1.0)
        return relevance
    
    def compute_semantic_similarity(self, answer: str, ground_truth: str) -> float:
        """
        Compute semantic similarity using embeddings or enhanced text overlap
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Similarity score between 0 and 1
        """
        if not answer or not ground_truth:
            return 0.0
        
        # If embedder is available, use semantic similarity
        if self.embedder:
            try:
                answer_embedding = self.embedder.encode_single(answer)
                truth_embedding = self.embedder.encode_single(ground_truth)
                return self.embedder.compute_similarity(answer_embedding, truth_embedding)
            except Exception as e:
                logger.warning(f"Embedding similarity failed: {e}")
        
        # Enhanced fallback for Bengali text
        answer_lower = answer.lower()
        truth_lower = ground_truth.lower()
        
        # Extract meaningful words (remove common words)
        answer_words = set(answer_lower.split())
        truth_words = set(truth_lower.split())
        
        # Remove very common Bengali words that don't add meaning
        common_words = {'এর', 'এই', 'সেই', 'তার', 'যে', 'হয়', 'করে', 'একটি', 'একজন', 'বছর', 'year', 'years'}
        answer_content = answer_words - common_words
        truth_content = truth_words - common_words
        
        if not answer_content or not truth_content:
            # If no content words, check for exact substring match
            if truth_lower in answer_lower or answer_lower in truth_lower:
                return 0.8
            return 0.0
        
        # Calculate enhanced similarity
        intersection = len(answer_content & truth_content)
        union = len(answer_content | truth_content)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Bonus for key information preservation
        bonus = 0.0
        
        # Check for number preservation (Bengali numerals and words)
        import re
        answer_numbers = re.findall(r'[০-৯]+|[0-9]+|সাতাশ|আঠাশ|তিরিশ|চল্লিশ|পঞ্চাশ|পনেরো|ষোল|সতেরো|আঠারো|উনিশ|বিশ|২৭|১৫', answer_lower)
        truth_numbers = re.findall(r'[০-৯]+|[0-9]+|সাতাশ|আঠাশ|তিরিশ|চল্লিশ|পঞ্চাশ|পনেরো|ষোল|সতেরো|আঠারো|উনিশ|বিশ|২৭|১৫', truth_lower)
        
        if answer_numbers and truth_numbers:
            if any(num in truth_numbers for num in answer_numbers):
                bonus += 0.3
        
        # Check for name/entity preservation
        answer_entities = re.findall(r'[অ-ৱ]{3,}', answer)  # Bengali words 3+ chars
        truth_entities = re.findall(r'[অ-ৱ]{3,}', ground_truth)
        
        entity_matches = sum(1 for entity in truth_entities if entity in answer_entities)
        if entity_matches > 0:
            bonus += min(0.2, entity_matches * 0.1)
        
        return min(jaccard + bonus, 1.0)
    
    def evaluate_context_utilization(self, answer: str, context: str) -> float:
        """
        Evaluate how well the context is utilized in the answer
        
        Args:
            answer: Generated answer
            context: Available context
            
        Returns:
            Utilization score between 0 and 1
        """
        if not answer or not context:
            return 0.0
        
        # Check if answer is too generic (doesn't use specific context)
        generic_phrases = [
            'আমি জানি না', 'নিশ্চিত নই', 'তথ্য নেই',
            "i don't know", "not sure", "no information"
        ]
        
        if any(phrase in answer.lower() for phrase in generic_phrases):
            return 0.1
        
        # Check for specific information usage
        context_length = len(context.split())
        answer_length = len(answer.split())
        
        # Penalize very short answers when context is rich
        if context_length > 50 and answer_length < 5:
            return 0.2
        
        # Reward answers that use specific facts from context
        groundedness = self.evaluate_groundedness(answer, context)
        
        # Additional check for information density
        info_density = min(answer_length / max(context_length * 0.1, 1), 1.0)
        
        utilization = (groundedness * 0.7) + (info_density * 0.3)
        return min(utilization, 1.0)
    
    def evaluate_single_question(self, 
                                question: str,
                                predicted_answer: str,
                                ground_truth_answer: str,
                                retrieved_context: str,
                                question_id: str = "",
                                response_time_ms: int = 0) -> RAGEvaluationResult:
        """
        Evaluate a single question-answer pair using focused RAG metrics
        
        Args:
            question: Input question
            predicted_answer: System's generated answer
            ground_truth_answer: Expected correct answer
            retrieved_context: Context retrieved by the system
            question_id: Unique identifier for the question
            response_time_ms: Response time in milliseconds
            
        Returns:
            RAGEvaluationResult with all computed metrics
        """
        logger.debug(f"Evaluating question: {question_id}")
        
        # Core RAG metrics
        groundedness = self.evaluate_groundedness(predicted_answer, retrieved_context)
        relevance = self.evaluate_relevance(question, retrieved_context)
        semantic_similarity = self.compute_semantic_similarity(predicted_answer, ground_truth_answer)
        context_utilization = self.evaluate_context_utilization(predicted_answer, retrieved_context)
        
        return RAGEvaluationResult(
            question_id=question_id,
            question=question,
            predicted_answer=predicted_answer,
            ground_truth_answer=ground_truth_answer,
            retrieved_context=retrieved_context,
            groundedness_score=groundedness,
            relevance_score=relevance,
            semantic_similarity=semantic_similarity,
            context_utilization=context_utilization,
            answer_length=len(predicted_answer.split()),
            context_length=len(retrieved_context.split()),
            response_time_ms=response_time_ms
        )
    
    def evaluate_test_set(self, 
                         test_cases: List[Dict[str, Any]], 
                         rag_chain,
                         output_file: Optional[str] = None) -> RAGEvaluationSummary:
        """
        Evaluate a complete test set
        
        Args:
            test_cases: List of test case dictionaries
            rag_chain: RAG system to evaluate
            output_file: Optional file to save results
            
        Returns:
            RAGEvaluationSummary with overall performance metrics
        """
        logger.info(f"Starting evaluation of {len(test_cases)} test cases")
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                question = test_case["question"]
                ground_truth = test_case["answer"]
                question_id = test_case.get("id", f"test_{i+1}")
                
                # Get system response
                start_time = datetime.now()
                response = rag_chain.ask(question)
                response_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                # Extract context from response
                context = ""
                if hasattr(response, 'source_chunks') and response.source_chunks:
                    context = "\n".join([chunk.get('content', '') for chunk in response.source_chunks])
                elif hasattr(response, 'context'):
                    context = response.context
                
                # Evaluate this question
                result = self.evaluate_single_question(
                    question=question,
                    predicted_answer=response.answer if hasattr(response, 'answer') else str(response),
                    ground_truth_answer=ground_truth,
                    retrieved_context=context,
                    question_id=question_id,
                    response_time_ms=response_time
                )
                
                results.append(result)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Evaluated {i+1}/{len(test_cases)} questions")
                    
            except Exception as e:
                logger.error(f"Error evaluating question {i+1}: {e}")
                continue
        
        # Compute summary
        summary = self._compute_summary(results)
        
        # Save results if requested
        if output_file:
            self._save_results(results, summary, output_file)
        
        logger.info(f"Evaluation completed: {len(results)} successful evaluations")
        return summary
    
    def _compute_summary(self, results: List[RAGEvaluationResult]) -> RAGEvaluationSummary:
        """Compute summary statistics from evaluation results"""
        
        if not results:
            return RAGEvaluationSummary(
                total_questions=0,
                avg_groundedness=0.0,
                avg_relevance=0.0,
                avg_semantic_similarity=0.0,
                avg_context_utilization=0.0,
                avg_response_time_ms=0.0,
                groundedness_distribution={"poor": 0, "fair": 0, "good": 0},
                relevance_distribution={"poor": 0, "fair": 0, "good": 0},
                performance_by_type={},
                evaluation_timestamp=datetime.now().isoformat()
            )
        
        # Calculate averages
        avg_groundedness = statistics.mean([r.groundedness_score for r in results])
        avg_relevance = statistics.mean([r.relevance_score for r in results])
        avg_semantic_similarity = statistics.mean([r.semantic_similarity for r in results])
        avg_context_utilization = statistics.mean([r.context_utilization for r in results])
        avg_response_time_ms = statistics.mean([r.response_time_ms for r in results])
        
        # Score distributions
        groundedness_dist = {"poor": 0, "fair": 0, "good": 0}
        relevance_dist = {"poor": 0, "fair": 0, "good": 0}
        
        for result in results:
            # Categorize groundedness
            if result.groundedness_score >= 0.7:
                groundedness_dist["good"] += 1
            elif result.groundedness_score >= 0.4:
                groundedness_dist["fair"] += 1
            else:
                groundedness_dist["poor"] += 1
            
            # Categorize relevance
            if result.relevance_score >= 0.7:
                relevance_dist["good"] += 1
            elif result.relevance_score >= 0.4:
                relevance_dist["fair"] += 1
            else:
                relevance_dist["poor"] += 1
        
        # Analyze by question type (basic categorization)
        performance_by_type = {}
        for result in results:
            question_type = self._categorize_question(result.question)
            if question_type not in performance_by_type:
                performance_by_type[question_type] = []
            performance_by_type[question_type].append(result)
        
        # Compute averages by type
        for question_type, type_results in performance_by_type.items():
            performance_by_type[question_type] = {
                "count": len(type_results),
                "avg_groundedness": statistics.mean([r.groundedness_score for r in type_results]),
                "avg_relevance": statistics.mean([r.relevance_score for r in type_results]),
                "avg_semantic_similarity": statistics.mean([r.semantic_similarity for r in type_results])
            }
        
        return RAGEvaluationSummary(
            total_questions=len(results),
            avg_groundedness=avg_groundedness,
            avg_relevance=avg_relevance,
            avg_semantic_similarity=avg_semantic_similarity,
            avg_context_utilization=avg_context_utilization,
            avg_response_time_ms=avg_response_time_ms,
            groundedness_distribution=groundedness_dist,
            relevance_distribution=relevance_dist,
            performance_by_type=performance_by_type,
            evaluation_timestamp=datetime.now().isoformat()
        )
    
    def _categorize_question(self, question: str) -> str:
        """Categorize question type for analysis"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['কে', 'who']):
            return "who_questions"
        elif any(word in question_lower for word in ['কী', 'কি', 'what']):
            return "what_questions"
        elif any(word in question_lower for word in ['কেন', 'why']):
            return "why_questions"
        elif any(word in question_lower for word in ['কখন', 'when']):
            return "when_questions"
        elif any(word in question_lower for word in ['কোথায়', 'where']):
            return "where_questions"
        elif any(word in question_lower for word in ['কীভাবে', 'how']):
            return "how_questions"
        else:
            return "other_questions"
    
    def _save_results(self, 
                     results: List[RAGEvaluationResult], 
                     summary: RAGEvaluationSummary, 
                     output_file: str):
        """Save evaluation results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionaries for JSON serialization
        results_dict = [asdict(result) for result in results]
        summary_dict = asdict(summary)
        
        evaluation_data = {
            "summary": summary_dict,
            "detailed_results": results_dict,
            "metadata": {
                "total_questions": len(results),
                "evaluation_date": datetime.now().isoformat(),
                "evaluator_type": "SimpleRAGEvaluator"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def print_evaluation_report(self, summary: RAGEvaluationSummary):
        """Print a formatted evaluation report"""
        print("\n" + "="*60)
        print("🔍 RAG SYSTEM EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Questions: {summary.total_questions}")
        print(f"   Average Response Time: {summary.avg_response_time_ms:.1f}ms")
        
        print(f"\n🎯 Core RAG Metrics:")
        print(f"   Groundedness:        {summary.avg_groundedness:.3f} (Is answer supported by context?)")
        print(f"   Relevance:          {summary.avg_relevance:.3f} (Are retrieved docs relevant?)")
        print(f"   Semantic Similarity: {summary.avg_semantic_similarity:.3f} (Answer quality)")
        print(f"   Context Utilization: {summary.avg_context_utilization:.3f} (How well context is used)")
        
        print(f"\n📈 Score Distributions:")
        print(f"   Groundedness -> Good: {summary.groundedness_distribution['good']}, "
              f"Fair: {summary.groundedness_distribution['fair']}, "
              f"Poor: {summary.groundedness_distribution['poor']}")
        print(f"   Relevance    -> Good: {summary.relevance_distribution['good']}, "
              f"Fair: {summary.relevance_distribution['fair']}, "
              f"Poor: {summary.relevance_distribution['poor']}")
        
        if summary.performance_by_type:
            print(f"\n🔍 Performance by Question Type:")
            for q_type, metrics in summary.performance_by_type.items():
                print(f"   {q_type.replace('_', ' ').title()}: "
                      f"Count={metrics['count']}, "
                      f"Ground={metrics['avg_groundedness']:.2f}, "
                      f"Rel={metrics['avg_relevance']:.2f}")
        
        print("\n" + "="*60)

# Test data creation function for Bengali literature
def create_bangla_literature_test_set() -> List[Dict[str, Any]]:
    """Create comprehensive test set for Bengali literature RAG evaluation"""
    return [
        {
            "id": "test_001",
            "question": "অনুপমের বয়স কত বছর?",
            "answer": "২৭ বছর",
            "type": "factual",
            "expected_context_keywords": ["অনুপম", "২৭", "বয়স", "বছর"]
        },
        {
            "id": "test_002", 
            "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "answer": "শুম্ভুনাথ",
            "type": "character_reference",
            "expected_context_keywords": ["শুম্ভুনাথ", "সুপুরুষ"]
        },
        {
            "id": "test_003",
            "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "answer": "মামাকে",
            "type": "character_reference",
            "expected_context_keywords": ["মামা", "ভাগ্য", "দেবতা", "অনুপম"]
        },
        {
            "id": "test_004",
            "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "answer": "১৫ বছর",
            "type": "factual",
            "expected_context_keywords": ["কল্যাণী", "১৫", "বয়স", "বিয়ে"]
        }
    ]

def run_evaluation_demo():
    """Demo function to show how to use the evaluator with realistic examples"""
    logger.info("🚀 Starting RAG Evaluation Demo")
    
    # Create evaluator (without embedder for demo)
    evaluator = SimpleRAGEvaluator()
    
    # Create sample test data
    test_cases = create_bangla_literature_test_set()
    
    print(f"\n📝 Demo Test Set: {len(test_cases)} questions")
    for case in test_cases[:3]:  # Show first 3
        print(f"   {case['id']}: {case['question']}")
    
    print(f"\n🔍 Running Sample Evaluations:")
    print("=" * 60)
    
    # Test 1: Good answer with relevant context
    print(f"\n1️⃣ Test Case 1 - Good Performance Expected:")
    sample_result_1 = evaluator.evaluate_single_question(
        question="অনুপমের বয়স কত বছর?",
        predicted_answer="অনুপমের বয়স ২৭ বছর।",
        ground_truth_answer="২৭ বছর",
        retrieved_context="অনুপম ২৭ বছর বয়সী একজন যুবক। তার মামা তার সব কাজের দায়িত্ব নিয়ন্ত্রণ করেন। অনুপমের বয়স নিয়ে গল্পে স্পষ্ট উল্লেখ রয়েছে।",
        question_id="demo_001"
    )
    
    print(f"   Question: {sample_result_1.question}")
    print(f"   Predicted: {sample_result_1.predicted_answer}")
    print(f"   Expected: {sample_result_1.ground_truth_answer}")
    print(f"   📊 Metrics:")
    print(f"      Groundedness: {sample_result_1.groundedness_score:.3f}")
    print(f"      Relevance: {sample_result_1.relevance_score:.3f}")
    print(f"      Semantic Similarity: {sample_result_1.semantic_similarity:.3f}")
    print(f"      Context Utilization: {sample_result_1.context_utilization:.3f}")
    
    # Test 2: Character reference question
    print(f"\n2️⃣ Test Case 2 - Character Reference:")
    sample_result_2 = evaluator.evaluate_single_question(
        question="অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        predicted_answer="শুম্ভুনাথকে সুপুরুষ বলা হয়েছে।",
        ground_truth_answer="শুম্ভুনাথ",
        retrieved_context="অনুপমের ভাষায় শুম্ভুনাথ একজন সুপুরুষ। তিনি সুদর্শন এবং ভদ্র স্বভাবের লোক। গল্পে শুম্ভুনাথের চরিত্র নিয়ে বিস্তৃত আলোচনা রয়েছে।",
        question_id="demo_002"
    )
    
    print(f"   Question: {sample_result_2.question}")
    print(f"   Predicted: {sample_result_2.predicted_answer}")
    print(f"   Expected: {sample_result_2.ground_truth_answer}")
    print(f"   � Metrics:")
    print(f"      Groundedness: {sample_result_2.groundedness_score:.3f}")
    print(f"      Relevance: {sample_result_2.relevance_score:.3f}")
    print(f"      Semantic Similarity: {sample_result_2.semantic_similarity:.3f}")
    print(f"      Context Utilization: {sample_result_2.context_utilization:.3f}")
    
    # Test 3: Poor context example
    print(f"\n3️⃣ Test Case 3 - Poor Context (Expected Low Scores):")
    sample_result_3 = evaluator.evaluate_single_question(
        question="অনুপমের বয়স কত বছর?",
        predicted_answer="আমি নিশ্চিত নই।",
        ground_truth_answer="২৭ বছর",
        retrieved_context="গল্পটি রবীন্দ্রনাথ ঠাকুরের লেখা। এটি একটি সামাজিক গল্প।",
        question_id="demo_003"
    )
    
    print(f"   Question: {sample_result_3.question}")
    print(f"   Predicted: {sample_result_3.predicted_answer}")
    print(f"   Expected: {sample_result_3.ground_truth_answer}")
    print(f"   📊 Metrics:")
    print(f"      Groundedness: {sample_result_3.groundedness_score:.3f}")
    print(f"      Relevance: {sample_result_3.relevance_score:.3f}")
    print(f"      Semantic Similarity: {sample_result_3.semantic_similarity:.3f}")
    print(f"      Context Utilization: {sample_result_3.context_utilization:.3f}")
    
    # Show interpretation guide
    print(f"\n📈 Score Interpretation Guide:")
    print(f"   🟢 Good (0.7-1.0): High quality")
    print(f"   🟡 Fair (0.4-0.7): Moderate quality") 
    print(f"   🔴 Poor (0.0-0.4): Needs improvement")
    
    print(f"\n📋 Available Test Questions:")
    for i, case in enumerate(test_cases, 1):
        print(f"   {i}. {case['question']} → {case['answer']}")
    
    logger.info("✅ Evaluation demo completed!")
    print(f"\n💡 To run full evaluation with your RAG system:")
    print(f"   python run_evaluation.py")
    print(f"   or")
    print(f"   python evaluation\\eval_metrics.py")

if __name__ == "__main__":
    # Run demo
    run_evaluation_demo()
    
    # Example of full evaluation (commented out - requires actual RAG chain)
    """
    # Full evaluation example:
    
    # Setup
    sys.path.append(str(Path(__file__).parent.parent))
    from embeddings.embed_model import create_embedder
    from generation.rag_chain import create_rag_chain
    
    # Create components
    embedder = create_embedder()
    rag_chain = create_rag_chain()
    evaluator = SimpleRAGEvaluator(embedder=embedder)
    
    # Run evaluation
    test_cases = create_bangla_literature_test_set()
    summary = evaluator.evaluate_test_set(
        test_cases=test_cases,
        rag_chain=rag_chain,
        output_file="evaluation_results.json"
    )
    
    # Print report
    evaluator.print_evaluation_report(summary)
    """ 