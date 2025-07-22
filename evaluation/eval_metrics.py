"""
Evaluation Metrics Module for Multilingual RAG System

This module provides:
- ROUGE metrics for answer quality
- BERTScore for semantic similarity  
- Cosine similarity for retrieval evaluation
- Custom metrics for multilingual content
- Ground truth test set management
- Automated evaluation pipeline
"""

import os
import json
import math
import statistics
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

import numpy as np
from loguru import logger

# Optional imports with fallbacks
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    logger.warning("ROUGE not available. Install with: pip install rouge-score")

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    logger.warning("BERTScore not available. Install with: pip install bert-score")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available for cosine similarity")

@dataclass
class EvaluationResult:
    """
    Data class for storing evaluation results
    """
    question_id: str
    question: str
    predicted_answer: str
    ground_truth_answer: str
    language: str
    
    # Retrieval metrics
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_f1: float = 0.0
    mean_reciprocal_rank: float = 0.0
    
    # Generation metrics
    rouge_1_f: float = 0.0
    rouge_2_f: float = 0.0
    rouge_l_f: float = 0.0
    bert_score_f1: float = 0.0
    semantic_similarity: float = 0.0
    
    # Custom metrics
    answer_relevance: float = 0.0
    factual_accuracy: float = 0.0
    language_consistency: float = 0.0
    confidence_score: float = 0.0
    
    # Metadata
    response_time_ms: int = 0
    source_chunks_used: int = 0
    model_used: str = ""

@dataclass  
class EvaluationSummary:
    """
    Summary of evaluation across multiple test cases
    """
    total_questions: int
    languages: List[str]
    
    # Average scores
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_retrieval_f1: float
    avg_mrr: float
    
    avg_rouge_1: float
    avg_rouge_2: float  
    avg_rouge_l: float
    avg_bert_score: float
    avg_semantic_similarity: float
    
    avg_answer_relevance: float
    avg_factual_accuracy: float
    avg_language_consistency: float
    avg_confidence: float
    
    avg_response_time_ms: float
    
    # Per-language breakdown
    language_scores: Dict[str, Dict[str, float]]
    
    # Timestamp
    evaluation_timestamp: str

class RAGEvaluator:
    """
    Comprehensive evaluator for RAG system performance
    """
    
    def __init__(self, 
                 embedder=None,
                 use_rouge: bool = True,
                 use_bertscore: bool = True):
        """
        Initialize RAG evaluator
        
        Args:
            embedder: Embedder for semantic similarity computation
            use_rouge: Whether to compute ROUGE scores
            use_bertscore: Whether to compute BERTScore
        """
        self.embedder = embedder
        self.use_rouge = use_rouge and HAS_ROUGE
        self.use_bertscore = use_bertscore and HAS_BERTSCORE
        
        # Initialize ROUGE scorer
        if self.use_rouge:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        
        logger.info(f"Initialized RAGEvaluator (ROUGE: {self.use_rouge}, BERTScore: {self.use_bertscore})")
    
    def compute_rouge_scores(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not self.use_rouge:
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, predicted)
            return {
                "rouge1_f": scores['rouge1'].fmeasure,
                "rouge2_f": scores['rouge2'].fmeasure,
                "rougeL_f": scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error computing ROUGE scores: {e}")
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    
    def compute_bert_score(self, predicted: str, reference: str, lang: str = "en") -> float:
        """
        Compute BERTScore
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            lang: Language code
            
        Returns:
            BERTScore F1
        """
        if not self.use_bertscore:
            return 0.0
        
        try:
            # BERTScore expects lists
            P, R, F1 = bert_score([predicted], [reference], lang=lang, verbose=False)
            return float(F1[0])
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            return 0.0
    
    def compute_semantic_similarity(self, predicted: str, reference: str) -> float:
        """
        Compute semantic similarity using embeddings
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Cosine similarity score
        """
        if not self.embedder:
            return 0.0
        
        try:
            pred_embedding = self.embedder.encode_single(predicted)
            ref_embedding = self.embedder.encode_single(reference)
            
            return self.embedder.compute_similarity(pred_embedding, ref_embedding)
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def compute_retrieval_metrics(self, 
                                 retrieved_chunks: List[Dict[str, Any]], 
                                 relevant_chunk_ids: List[str]) -> Dict[str, float]:
        """
        Compute retrieval evaluation metrics
        
        Args:
            retrieved_chunks: List of retrieved chunks with metadata
            relevant_chunk_ids: List of relevant chunk IDs for ground truth
            
        Returns:
            Dictionary with retrieval metrics
        """
        if not retrieved_chunks or not relevant_chunk_ids:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mrr": 0.0
            }
        
        # Extract retrieved chunk IDs
        retrieved_ids = [chunk.get('chunk_id', '') for chunk in retrieved_chunks]
        relevant_set = set(relevant_chunk_ids)
        retrieved_set = set(retrieved_ids)
        
        # Precision and Recall
        true_positives = len(relevant_set & retrieved_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(relevant_set) if relevant_set else 0.0
        
        # F1 Score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Mean Reciprocal Rank
        mrr = 0.0
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr
        }
    
    def evaluate_answer_relevance(self, question: str, answer: str, context: str = "") -> float:
        """
        Evaluate answer relevance to question
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context (optional)
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple heuristic-based relevance scoring
        score = 0.0
        
        # Check if answer is not empty or generic
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        # Check for "I don't know" type responses
        dont_know_phrases = [
            "don't know", "not sure", "couldn't find", "no information",
            "জানি না", "নিশ্চিত নই", "পাইনি", "তথ্য নেই"
        ]
        
        if any(phrase in answer.lower() for phrase in dont_know_phrases):
            score = 0.2  # Partial credit for honest uncertainty
        else:
            score = 0.7  # Base score for providing an answer
        
        # Bonus for using context-specific information
        if context and len(context) > 0:
            # Simple overlap check
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            
            # Check if answer contains information from context
            context_overlap = len(answer_words & context_words) / len(context_words) if context_words else 0
            score += min(0.3, context_overlap * 0.3)
        
        return min(score, 1.0)
    
    def evaluate_factual_accuracy(self, 
                                 answer: str, 
                                 ground_truth: str,
                                 question: str = "") -> float:
        """
        Evaluate factual accuracy of answer
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            question: Original question (for context)
            
        Returns:
            Accuracy score between 0 and 1
        """
        if not answer or not ground_truth:
            return 0.0
        
        # Compute semantic similarity as proxy for factual accuracy
        semantic_sim = self.compute_semantic_similarity(answer, ground_truth)
        
        # Compute ROUGE-L as another proxy
        if self.use_rouge:
            rouge_scores = self.compute_rouge_scores(answer, ground_truth)
            rouge_l = rouge_scores.get("rougeL_f", 0.0)
        else:
            rouge_l = 0.0
        
        # Combine metrics
        accuracy = (semantic_sim * 0.7 + rouge_l * 0.3)
        
        return min(accuracy, 1.0)
    
    def evaluate_language_consistency(self, question: str, answer: str, expected_lang: str) -> float:
        """
        Evaluate language consistency between question and answer
        
        Args:
            question: Original question
            answer: Generated answer
            expected_lang: Expected language
            
        Returns:
            Consistency score between 0 and 1
        """
        try:
            from langdetect import detect
            
            # Detect answer language
            detected_lang = detect(answer) if len(answer) > 10 else expected_lang
            
            # Check consistency
            if expected_lang == 'mixed':
                return 1.0  # Mixed language is always consistent
            elif expected_lang == detected_lang:
                return 1.0
            elif expected_lang == 'bn' and detected_lang in ['bn', 'hi', 'ur']:
                return 0.8  # Close languages
            elif expected_lang == 'en' and detected_lang in ['en']:
                return 1.0
            else:
                return 0.3  # Different language family
                
        except Exception as e:
            logger.debug(f"Language detection error: {e}")
            return 0.5  # Neutral score if detection fails
    
    def evaluate_single_question(self, 
                                question: str,
                                predicted_answer: str,
                                ground_truth_answer: str,
                                retrieved_chunks: List[Dict[str, Any]],
                                relevant_chunk_ids: List[str],
                                question_id: str = "",
                                language: str = "en",
                                confidence_score: float = 0.0,
                                response_time_ms: int = 0,
                                model_used: str = "") -> EvaluationResult:
        """
        Evaluate a single question-answer pair
        
        Args:
            question: Input question
            predicted_answer: System's answer
            ground_truth_answer: Expected answer
            retrieved_chunks: Retrieved context chunks
            relevant_chunk_ids: Relevant chunk IDs for ground truth
            question_id: Unique question ID
            language: Question language
            confidence_score: System confidence
            response_time_ms: Response time
            model_used: Model identifier
            
        Returns:
            EvaluationResult object
        """
        logger.debug(f"Evaluating question: {question_id}")
        
        # Compute retrieval metrics
        retrieval_metrics = self.compute_retrieval_metrics(retrieved_chunks, relevant_chunk_ids)
        
        # Compute generation metrics
        rouge_scores = self.compute_rouge_scores(predicted_answer, ground_truth_answer)
        bert_score_f1 = self.compute_bert_score(predicted_answer, ground_truth_answer, language)
        semantic_sim = self.compute_semantic_similarity(predicted_answer, ground_truth_answer)
        
        # Compute custom metrics
        context_text = "\n".join([chunk.get('content', '') for chunk in retrieved_chunks])
        answer_relevance = self.evaluate_answer_relevance(question, predicted_answer, context_text)
        factual_accuracy = self.evaluate_factual_accuracy(predicted_answer, ground_truth_answer, question)
        language_consistency = self.evaluate_language_consistency(question, predicted_answer, language)
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            predicted_answer=predicted_answer,
            ground_truth_answer=ground_truth_answer,
            language=language,
            
            # Retrieval metrics
            retrieval_precision=retrieval_metrics["precision"],
            retrieval_recall=retrieval_metrics["recall"],
            retrieval_f1=retrieval_metrics["f1"],
            mean_reciprocal_rank=retrieval_metrics["mrr"],
            
            # Generation metrics
            rouge_1_f=rouge_scores["rouge1_f"],
            rouge_2_f=rouge_scores["rouge2_f"],
            rouge_l_f=rouge_scores["rougeL_f"],
            bert_score_f1=bert_score_f1,
            semantic_similarity=semantic_sim,
            
            # Custom metrics
            answer_relevance=answer_relevance,
            factual_accuracy=factual_accuracy,
            language_consistency=language_consistency,
            confidence_score=confidence_score,
            
            # Metadata
            response_time_ms=response_time_ms,
            source_chunks_used=len(retrieved_chunks),
            model_used=model_used
        )
    
    def evaluate_test_set(self, 
                         test_cases: List[Dict[str, Any]], 
                         rag_chain,
                         output_file: Optional[str] = None) -> EvaluationSummary:
        """
        Evaluate complete test set
        
        Args:
            test_cases: List of test case dictionaries
            rag_chain: RAG chain instance for generating answers
            output_file: Optional file to save detailed results
            
        Returns:
            EvaluationSummary object
        """
        logger.info(f"Evaluating {len(test_cases)} test cases")
        
        results = []
        languages = set()
        
        for i, test_case in enumerate(test_cases):
            try:
                question = test_case["question"]
                ground_truth = test_case["answer"]
                language = test_case.get("language", "en")
                question_id = test_case.get("id", f"test_{i}")
                relevant_chunks = test_case.get("relevant_chunks", [])
                
                languages.add(language)
                
                # Get system response
                start_time = datetime.now()
                response = rag_chain.ask(question)
                response_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                # Evaluate
                result = self.evaluate_single_question(
                    question=question,
                    predicted_answer=response.answer,
                    ground_truth_answer=ground_truth,
                    retrieved_chunks=response.source_chunks,
                    relevant_chunk_ids=relevant_chunks,
                    question_id=question_id,
                    language=language,
                    confidence_score=response.confidence_score,
                    response_time_ms=response_time,
                    model_used=response.model_used
                )
                
                results.append(result)
                
                if i % 10 == 0:
                    logger.info(f"Evaluated {i+1}/{len(test_cases)} questions")
                    
            except Exception as e:
                logger.error(f"Error evaluating question {i}: {e}")
                continue
        
        # Compute summary
        summary = self._compute_evaluation_summary(results, list(languages))
        
        # Save detailed results if requested
        if output_file:
            self.save_evaluation_results(results, summary, output_file)
        
        logger.info(f"Evaluation completed: {len(results)} successful evaluations")
        return summary
    
    def _compute_evaluation_summary(self, 
                                   results: List[EvaluationResult], 
                                   languages: List[str]) -> EvaluationSummary:
        """Compute summary statistics from evaluation results"""
        
        if not results:
            return EvaluationSummary(
                total_questions=0,
                languages=[],
                avg_retrieval_precision=0.0,
                avg_retrieval_recall=0.0,
                avg_retrieval_f1=0.0,
                avg_mrr=0.0,
                avg_rouge_1=0.0,
                avg_rouge_2=0.0,
                avg_rouge_l=0.0,
                avg_bert_score=0.0,
                avg_semantic_similarity=0.0,
                avg_answer_relevance=0.0,
                avg_factual_accuracy=0.0,
                avg_language_consistency=0.0,
                avg_confidence=0.0,
                avg_response_time_ms=0.0,
                language_scores={},
                evaluation_timestamp=datetime.now().isoformat()
            )
        
        # Overall averages
        avg_retrieval_precision = statistics.mean([r.retrieval_precision for r in results])
        avg_retrieval_recall = statistics.mean([r.retrieval_recall for r in results])
        avg_retrieval_f1 = statistics.mean([r.retrieval_f1 for r in results])
        avg_mrr = statistics.mean([r.mean_reciprocal_rank for r in results])
        
        avg_rouge_1 = statistics.mean([r.rouge_1_f for r in results])
        avg_rouge_2 = statistics.mean([r.rouge_2_f for r in results])
        avg_rouge_l = statistics.mean([r.rouge_l_f for r in results])
        avg_bert_score = statistics.mean([r.bert_score_f1 for r in results])
        avg_semantic_similarity = statistics.mean([r.semantic_similarity for r in results])
        
        avg_answer_relevance = statistics.mean([r.answer_relevance for r in results])
        avg_factual_accuracy = statistics.mean([r.factual_accuracy for r in results])
        avg_language_consistency = statistics.mean([r.language_consistency for r in results])
        avg_confidence = statistics.mean([r.confidence_score for r in results])
        avg_response_time_ms = statistics.mean([r.response_time_ms for r in results])
        
        # Per-language breakdown
        language_scores = {}
        for lang in languages:
            lang_results = [r for r in results if r.language == lang]
            if lang_results:
                language_scores[lang] = {
                    "count": len(lang_results),
                    "avg_rouge_l": statistics.mean([r.rouge_l_f for r in lang_results]),
                    "avg_semantic_similarity": statistics.mean([r.semantic_similarity for r in lang_results]),
                    "avg_answer_relevance": statistics.mean([r.answer_relevance for r in lang_results]),
                    "avg_factual_accuracy": statistics.mean([r.factual_accuracy for r in lang_results]),
                    "avg_language_consistency": statistics.mean([r.language_consistency for r in lang_results])
                }
        
        return EvaluationSummary(
            total_questions=len(results),
            languages=languages,
            avg_retrieval_precision=avg_retrieval_precision,
            avg_retrieval_recall=avg_retrieval_recall,
            avg_retrieval_f1=avg_retrieval_f1,
            avg_mrr=avg_mrr,
            avg_rouge_1=avg_rouge_1,
            avg_rouge_2=avg_rouge_2,
            avg_rouge_l=avg_rouge_l,
            avg_bert_score=avg_bert_score,
            avg_semantic_similarity=avg_semantic_similarity,
            avg_answer_relevance=avg_answer_relevance,
            avg_factual_accuracy=avg_factual_accuracy,
            avg_language_consistency=avg_language_consistency,
            avg_confidence=avg_confidence,
            avg_response_time_ms=avg_response_time_ms,
            language_scores=language_scores,
            evaluation_timestamp=datetime.now().isoformat()
        )
    
    def save_evaluation_results(self, 
                               results: List[EvaluationResult], 
                               summary: EvaluationSummary, 
                               output_file: str):
        """Save evaluation results to file"""
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
                "evaluator_config": {
                    "use_rouge": self.use_rouge,
                    "use_bertscore": self.use_bertscore,
                    "has_embedder": self.embedder is not None
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {output_path}")

def create_hsc_bangla_test_set() -> List[Dict[str, Any]]:
    """Create HSC Bangla literature test set based on actual textbook content"""
    return [
        {
            "id": "test_001",
            "question": "অনুপমের বয়স কত বছর?",
            "answer": "অনুপমের বয়স সাতাশ বছর।",
            "language": "bn",
            "relevant_chunks": ["page_1_para_1", "page_1_mcq_1"]
        },
        {
            "id": "test_002",
            "question": "'অপরিচিতা' গল্পে অনুপমের চাচার সাথে কোন চরিত্রের মিল রয়েছে?",
            "answer": "'অপরিচিতা' গল্পে অনুপমের চাচার সাথে মামার চরিত্রের মিল রয়েছে।",
            "language": "bn", 
            "relevant_chunks": ["page_1_mcq_3", "page_2_paragraph_1"]
        },
        {
            "id": "test_003",
            "question": "What is the main theme of 'Aparichita' story?",
            "answer": "The main theme of 'Aparichita' is the critique of dowry system and arranged marriage practices in Bengali society.",
            "language": "en",
            "relevant_chunks": ["page_1_definition_1", "page_3_paragraph_2"]
        },
        {
            "id": "test_004",
            "question": "অনুপমের মামার চরিত্রের বৈশিষ্ট্য কী?",
            "answer": "অনুপমের মামা একজন লোভী, কূটবুদ্ধিসম্পন্ন এবং স্বার্থপর ব্যক্তি যিনি পরিবারের সব দায়িত্ব নিয়ন্ত্রণ করেন।",
            "language": "bn",
            "relevant_chunks": ["page_2_paragraph_2", "page_3_paragraph_1"]
        },
        {
            "id": "test_005",
            "question": "গল্পে 'ফল্গুর বালির মতো' বলতে কী বোঝানো হয়েছে?",
            "answer": "ফল্গুর বালির মতো বলতে বোঝানো হয়েছে যে, মামা বাইরে থেকে সাধারণ মনে হলেও ভেতরে সংসারের সব বিষয় নিয়ন্ত্রণ করেন।",
            "language": "bn",
            "relevant_chunks": ["page_2_definition_1", "page_2_paragraph_3"]
        },
        {
            "id": "test_006", 
            "question": "Who is Harish in the story and what role does he play?",
            "answer": "Harish is Anupam's friend who brings the marriage proposal and acts as a mediator between the two families.",
            "language": "en",
            "relevant_chunks": ["page_3_paragraph_1", "page_4_paragraph_1"]
        },
        {
            "id": "test_007",
            "question": "বিবাহের ক্ষেত্রে অনুপমের মামার দৃষ্টিভঙ্গি কী ছিল?",
            "answer": "বিবাহের ক্ষেত্রে মামার দৃষ্টিভঙ্গি ছিল যে মেয়ের পরিবার ধনী হতে হবে এবং যৌতুক দিতে সক্ষম হতে হবে।",
            "language": "bn",
            "relevant_chunks": ["page_4_paragraph_2", "page_5_paragraph_1"]
        },
        {
            "id": "test_008",
            "question": "What does the gold testing scene reveal about the character dynamics?",
            "answer": "The gold testing scene reveals the materialistic nature of the groom's family and their lack of trust, which ultimately leads to the breakdown of the marriage proposal.",
            "language": "en",
            "relevant_chunks": ["page_8_paragraph_1", "page_9_paragraph_1"]
        },
        {
            "id": "test_009",
            "question": "শম্ভুনাথ বাবু কেন স্বর্ণ পরীক্ষা করতে চেয়েছিলেন?",
            "answer": "শম্ভুনাথ বাবু স্বর্ণ পরীক্ষা করতে চেয়েছিলেন কারণ তিনি নিশ্চিত হতে চেয়েছিলেন যে দেওয়া স্বর্ণালংকার খাঁটি কিনা।",
            "language": "bn", 
            "relevant_chunks": ["page_8_paragraph_2", "page_9_paragraph_2"]
        },
        {
            "id": "test_010",
            "question": "'গণ্ডূষ' শব্দের অর্থ কী এবং গল্পে এর ব্যবহার কেমন?",
            "answer": "'গণ্ডূষ' শব্দের অর্থ এক মুখ বা এক কোষ জল। গল্পে এর ব্যবহার করে বোঝানো হয়েছে যে মামার কাছে খুব সামান্য পরিমাণ জলও পাওয়া যায় না।",
            "language": "bn",
            "relevant_chunks": ["page_2_definition_2", "page_2_paragraph_4"]
        }
    ]

def create_sample_test_set() -> List[Dict[str, Any]]:
    """Create a sample test set for demonstration - keeping for backward compatibility"""
    return create_hsc_bangla_test_set()

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    
    # Add parent directories to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from embeddings.embed_model import create_embedder
        from retrieval.vector_store import create_vector_store
        from generation.rag_chain import create_rag_chain
        
        # Create sample evaluator
        logger.info("Creating sample evaluator...")
        embedder = create_embedder()
        evaluator = RAGEvaluator(embedder=embedder)
        
        # Test individual metrics
        predicted = "বাংলাদেশের রাজধানী হলো ঢাকা।"
        reference = "বাংলাদেশের রাজধানী ঢাকা।"
        
        rouge_scores = evaluator.compute_rouge_scores(predicted, reference)
        semantic_sim = evaluator.compute_semantic_similarity(predicted, reference)
        bert_score = evaluator.compute_bert_score(predicted, reference, "bn")
        
        logger.info(f"ROUGE scores: {rouge_scores}")
        logger.info(f"Semantic similarity: {semantic_sim:.4f}")
        logger.info(f"BERTScore: {bert_score:.4f}")
        
        # Test retrieval metrics
        retrieved_chunks = [
            {"chunk_id": "chunk_1", "content": "Sample content"},
            {"chunk_id": "chunk_2", "content": "More content"}
        ]
        relevant_chunks = ["chunk_1", "chunk_3"]
        
        retrieval_metrics = evaluator.compute_retrieval_metrics(retrieved_chunks, relevant_chunks)
        logger.info(f"Retrieval metrics: {retrieval_metrics}")
        
        # Create sample test set
        test_set = create_sample_test_set()
        logger.info(f"Created sample test set with {len(test_set)} questions")
        
        # Save sample test set
        with open("sample_test_set.json", "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2, ensure_ascii=False)
        logger.info("Sample test set saved to sample_test_set.json")
        
    except ImportError as e:
        logger.warning(f"Could not import dependencies: {e}")
        
        # Basic test without dependencies
        logger.info("Running basic evaluator test...")
        evaluator = RAGEvaluator()
        
        # Test basic metrics
        result = evaluator.evaluate_answer_relevance(
            "What is the capital?", 
            "The capital is Dhaka",
            "Bangladesh capital city information"
        )
        logger.info(f"Answer relevance: {result:.4f}")
        
        logger.info("Basic evaluation tests completed") 