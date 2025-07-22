"""
RAG Chain Implementation for Multilingual Educational System

This module provides:
- LangChain-based RAG pipeline with Ollama integration
- Language detection and automatic LLM switching
- Memory management for conversation history
- Custom prompt templates for educational content
- Support for both Bangla (bongLlama) and English (mistral-instruct) models
"""

import os
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import Ollama
from loguru import logger
from langdetect import detect, LangDetectError

@dataclass
class RAGResponse:
    """
    Data class for RAG response with metadata
    """
    answer: str
    source_chunks: List[Dict[str, Any]]
    query_language: str
    model_used: str
    confidence_score: float
    total_tokens: Optional[int] = None

class LanguageDetector:
    """
    Enhanced language detection for query routing
    """
    
    def __init__(self):
        self.bangla_keywords = [
            'কি', 'কী', 'কে', 'কোথায়', 'কেন', 'কিভাবে', 'কখন', 
            'বাংলা', 'বাংলাদেশ', 'ঢাকা', 'এর', 'এবং', 'তার', 'যে'
        ]
        self.english_keywords = [
            'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'the', 'and'
        ]
    
    def detect_language(self, text: str) -> str:
        """
        Detect language with enhanced logic
        
        Args:
            text: Input text
            
        Returns:
            Language code ('bn', 'en', 'mixed')
        """
        if not text or len(text.strip()) < 3:
            return 'en'  # Default to English
        
        text_lower = text.lower()
        
        # Check for Bangla characters
        bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text))
        
        if total_chars == 0:
            return 'en'
        
        bangla_ratio = bangla_chars / total_chars
        
        # Check for language-specific keywords
        bangla_keyword_count = sum(1 for keyword in self.bangla_keywords if keyword in text_lower)
        english_keyword_count = sum(1 for keyword in self.english_keywords if keyword in text_lower)
        
        # Decision logic
        if bangla_ratio > 0.5 or bangla_keyword_count > english_keyword_count:
            return 'bn'
        elif bangla_ratio > 0.2:
            return 'mixed'
        else:
            # Try langdetect for confirmation
            try:
                detected = detect(text)
                if detected == 'bn':
                    return 'bn'
            except LangDetectError:
                pass
            
            return 'en'

class OllamaLLMManager:
    """
    Manager for multiple Ollama models
    """
    
    def __init__(self, 
                 bangla_model: str = "bongllama",
                 english_model: str = "mistral:instruct",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1):
        """
        Initialize Ollama LLM manager
        
        Args:
            bangla_model: Model name for Bangla text
            english_model: Model name for English text
            base_url: Ollama API base URL
            temperature: Generation temperature
        """
        self.bangla_model = bangla_model
        self.english_model = english_model
        self.base_url = base_url
        self.temperature = temperature
        
        # Initialize LLMs
        self.llms = {}
        self._init_llms()
        
        logger.info(f"Initialized Ollama LLMs - Bangla: {bangla_model}, English: {english_model}")
    
    def _init_llms(self):
        """Initialize Ollama LLM instances"""
        try:
            self.llms['bn'] = Ollama(
                model=self.bangla_model,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
            self.llms['en'] = Ollama(
                model=self.english_model,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
            self.llms['mixed'] = self.llms['bn']  # Use Bangla model for mixed content
            
        except Exception as e:
            logger.error(f"Error initializing Ollama LLMs: {str(e)}")
            raise
    
    def get_llm(self, language: str) -> Ollama:
        """
        Get appropriate LLM for language
        
        Args:
            language: Language code
            
        Returns:
            Ollama LLM instance
        """
        return self.llms.get(language, self.llms['en'])
    
    def test_connection(self) -> Dict[str, bool]:
        """
        Test connection to Ollama models
        
        Returns:
            Dictionary with connection status
        """
        results = {}
        
        for lang, llm in self.llms.items():
            try:
                # Simple test query
                response = llm.invoke("Hello")
                results[lang] = bool(response)
            except Exception as e:
                logger.error(f"Connection test failed for {lang} model: {str(e)}")
                results[lang] = False
        
        return results

class RAGPromptManager:
    """
    Manager for RAG prompt templates
    """
    
    def __init__(self):
        """Initialize prompt templates"""
        self.templates = self._create_templates()
    
    def _create_templates(self) -> Dict[str, PromptTemplate]:
        """Create language-specific prompt templates"""
        
        # English template
        english_template = """Use the following context from the HSC textbook to answer the user's question.
If you cannot find the answer in the provided context, say: "I'm sorry, I couldn't find the answer in the book."

Context:
{context}

Question: {question}

Answer: """

        # Bangla template  
        bangla_template = """নিম্নলিখিত HSC পাঠ্যবইয়ের প্রসঙ্গ ব্যবহার করে ব্যবহারকারীর প্রশ্নের উত্তর দিন।
যদি আপনি প্রদত্ত প্রসঙ্গে উত্তর খুঁজে না পান, তাহলে বলুন: "দুঃখিত, আমি বইয়ে উত্তর খুঁজে পাইনি।"

প্রসঙ্গ:
{context}

প্রশ্ন: {question}

উত্তর: """

        # Mixed language template (primarily Bangla with English fallback)
        mixed_template = """নিম্নলিখিত HSC পাঠ্যবইয়ের প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর দিন। Use the context from the HSC textbook to answer the question.
যদি উত্তর না পান তাহলে বলুন: "দুঃখিত, আমি বইয়ে উত্তর খুঁজে পাইনি।" / If you cannot find the answer, say: "I'm sorry, I couldn't find the answer in the book."

Context/প্রসঙ্গ:
{context}

Question/প্রশ্ন: {question}

Answer/উত্তর: """

        return {
            'en': PromptTemplate(
                template=english_template,
                input_variables=['context', 'question']
            ),
            'bn': PromptTemplate(
                template=bangla_template,
                input_variables=['context', 'question']
            ),
            'mixed': PromptTemplate(
                template=mixed_template,
                input_variables=['context', 'question']
            )
        }
    
    def get_prompt(self, language: str) -> PromptTemplate:
        """Get prompt template for language"""
        return self.templates.get(language, self.templates['en'])

class MultilingualRAGChain:
    """
    Main RAG chain with multilingual support
    """
    
    def __init__(self,
                 vector_store,
                 embedder,
                 bangla_model: str = "bongllama",
                 english_model: str = "mistral:instruct",
                 ollama_base_url: str = "http://localhost:11434",
                 memory_window: int = 10,
                 temperature: float = 0.1):
        """
        Initialize multilingual RAG chain
        
        Args:
            vector_store: Vector store instance
            embedder: Embedder instance
            bangla_model: Bangla model name
            english_model: English model name
            ollama_base_url: Ollama base URL
            memory_window: Memory window size
            temperature: Generation temperature
        """
        self.vector_store = vector_store
        self.embedder = embedder
        
        # Initialize components
        self.language_detector = LanguageDetector()
        self.llm_manager = OllamaLLMManager(
            bangla_model=bangla_model,
            english_model=english_model,
            base_url=ollama_base_url,
            temperature=temperature
        )
        self.prompt_manager = RAGPromptManager()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history"
        )
        
        logger.info("Initialized MultilingualRAGChain")
    
    def _retrieve_context(self, 
                         query: str, 
                         k: int = 5, 
                         threshold: float = 0.3) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant context from vector store
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            threshold: Similarity threshold
            
        Returns:
            Tuple of (context_text, source_chunks)
        """
        try:
            # Search vector store
            results = self.vector_store.search_by_text(
                query, 
                self.embedder, 
                k=k, 
                threshold=threshold
            )
            
            if not results:
                return "", []
            
            # Format context
            context_parts = []
            source_chunks = []
            
            for result in results:
                context_parts.append(result.content)
                source_chunks.append({
                    'chunk_id': result.chunk_id,
                    'content': result.content,
                    'score': result.score,
                    'metadata': result.metadata
                })
            
            context_text = "\n\n".join(context_parts)
            
            logger.debug(f"Retrieved {len(results)} chunks for query")
            return context_text, source_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "", []
    
    def _format_context_with_memory(self, context: str, query: str) -> str:
        """
        Format context with conversation memory
        
        Args:
            context: Retrieved context
            query: Current query
            
        Returns:
            Formatted context with memory
        """
        # Get recent conversation history
        memory_context = ""
        if hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages[-4:]  # Last 2 exchanges
            if messages:
                memory_context = "\n\nRecent conversation:\n"
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        memory_context += f"Human: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        memory_context += f"Assistant: {msg.content}\n"
        
        return context + memory_context
    
    def _calculate_confidence(self, 
                            source_chunks: List[Dict], 
                            query: str) -> float:
        """
        Calculate confidence score for the response
        
        Args:
            source_chunks: Retrieved source chunks
            query: User query
            
        Returns:
            Confidence score between 0 and 1
        """
        if not source_chunks:
            return 0.0
        
        # Average similarity score
        avg_score = sum(chunk['score'] for chunk in source_chunks) / len(source_chunks)
        
        # Number of sources factor
        num_sources_factor = min(len(source_chunks) / 3.0, 1.0)
        
        # Length factor (longer context generally better)
        total_context_length = sum(len(chunk['content']) for chunk in source_chunks)
        length_factor = min(total_context_length / 1000.0, 1.0)
        
        # Combined confidence
        confidence = (avg_score * 0.6 + num_sources_factor * 0.2 + length_factor * 0.2)
        
        return min(confidence, 1.0)
    
    def ask(self, 
           question: str, 
           max_context_chunks: int = 5,
           similarity_threshold: float = 0.3,
           include_sources: bool = True) -> RAGResponse:
        """
        Main method to ask questions to the RAG system
        
        Args:
            question: User question
            max_context_chunks: Maximum chunks to retrieve
            similarity_threshold: Minimum similarity threshold
            include_sources: Whether to include source information
            
        Returns:
            RAGResponse object
        """
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Detect language
            language = self.language_detector.detect_language(question)
            logger.debug(f"Detected language: {language}")
            
            # Retrieve context
            context, source_chunks = self._retrieve_context(
                question, 
                k=max_context_chunks, 
                threshold=similarity_threshold
            )
            
            if not context:
                return RAGResponse(
                    answer="I'm sorry, I couldn't find the answer in the book.",
                    source_chunks=[],
                    query_language=language,
                    model_used="none",
                    confidence_score=0.0
                )
            
            # Format context with memory
            formatted_context = self._format_context_with_memory(context, question)
            
            # Get appropriate LLM and prompt
            llm = self.llm_manager.get_llm(language)
            prompt = self.prompt_manager.get_prompt(language)
            
            # Create RAG chain
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Generate response
            response = rag_chain.invoke({
                "context": formatted_context,
                "question": question
            })
            
            # Calculate confidence
            confidence = self._calculate_confidence(source_chunks, question)
            
            # Update memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(response)
            
            # Create response object
            rag_response = RAGResponse(
                answer=response.strip(),
                source_chunks=source_chunks if include_sources else [],
                query_language=language,
                model_used=f"{language}:{llm.model}",
                confidence_score=confidence
            )
            
            logger.info(f"Generated response with confidence: {confidence:.3f}")
            return rag_response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your question.",
                source_chunks=[],
                query_language=language if 'language' in locals() else 'en',
                model_used="error",
                confidence_score=0.0
            )
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history
        
        Returns:
            List of conversation exchanges
        """
        history = []
        if hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append({
                        'question': messages[i].content,
                        'answer': messages[i + 1].content
                    })
        
        return history
    
    def test_components(self) -> Dict[str, Any]:
        """
        Test all components of the RAG system
        
        Returns:
            Dictionary with test results
        """
        results = {
            'ollama_connection': self.llm_manager.test_connection(),
            'vector_store_stats': self.vector_store.get_statistics(),
            'embedder_test': False,
            'memory_test': False
        }
        
        # Test embedder
        try:
            test_embedding = self.embedder.encode_query("test")
            results['embedder_test'] = len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embedder test failed: {e}")
        
        # Test memory
        try:
            self.memory.chat_memory.add_user_message("test")
            self.memory.chat_memory.add_ai_message("test response")
            results['memory_test'] = True
            self.memory.clear()
        except Exception as e:
            logger.error(f"Memory test failed: {e}")
        
        return results

def create_rag_chain(vector_store,
                    embedder,
                    bangla_model: str = "bongllama",
                    english_model: str = "mistral:instruct",
                    ollama_base_url: str = "http://localhost:11434") -> MultilingualRAGChain:
    """
    Factory function to create RAG chain
    
    Args:
        vector_store: Vector store instance
        embedder: Embedder instance
        bangla_model: Bangla model name
        english_model: English model name
        ollama_base_url: Ollama base URL
        
    Returns:
        MultilingualRAGChain instance
    """
    return MultilingualRAGChain(
        vector_store=vector_store,
        embedder=embedder,
        bangla_model=bangla_model,
        english_model=english_model,
        ollama_base_url=ollama_base_url
    )

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    
    # Add parent directories to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from embeddings.embed_model import create_embedder
        from retrieval.vector_store import create_vector_store, build_vector_store_from_embeddings
        
        # Test with simple setup
        logger.info("Testing RAG chain components...")
        
        # Create embedder
        embedder = create_embedder(cache_dir="../cache/embeddings")
        
        # Create sample vector store
        embedding_dim = embedder.get_embedding_dimension()
        vector_store = create_vector_store(
            embedding_dimension=embedding_dim,
            index_type="flat",
            metric="cosine"
        )
        
        # Add sample data
        sample_texts = [
            "বাংলাদেশের রাজধানী ঢাকা।",
            "The capital of Bangladesh is Dhaka.",
            "এইচএসসি পরীক্ষা বাংলাদেশের একটি গুরুত্বপূর্ণ পরীক্ষা।"
        ]
        
        sample_ids = [f"sample_{i}" for i in range(len(sample_texts))]
        sample_metadata = [{"source": "test", "index": i} for i in range(len(sample_texts))]
        sample_embeddings = embedder.encode_batch(sample_texts)
        
        vector_store.add_embeddings(
            sample_embeddings, sample_ids, sample_texts, sample_metadata
        )
        
        # Create RAG chain
        rag_chain = create_rag_chain(
            vector_store=vector_store,
            embedder=embedder,
            bangla_model="bongllama",
            english_model="mistral:instruct"
        )
        
        # Test components
        test_results = rag_chain.test_components()
        logger.info(f"Component test results: {test_results}")
        
        # Test questions (if Ollama is running)
        if any(test_results['ollama_connection'].values()):
            test_questions = [
                "বাংলাদেশের রাজধানী কী?",
                "What is the capital of Bangladesh?",
                "Tell me about HSC exams"
            ]
            
            for question in test_questions:
                logger.info(f"\nTesting question: {question}")
                try:
                    response = rag_chain.ask(question)
                    logger.info(f"Response: {response.answer[:200]}...")
                    logger.info(f"Language: {response.query_language}")
                    logger.info(f"Model: {response.model_used}")
                    logger.info(f"Confidence: {response.confidence_score:.3f}")
                except Exception as e:
                    logger.error(f"Error: {str(e)}")
        else:
            logger.warning("Ollama not available, skipping question tests")
        
    except ImportError as e:
        logger.warning(f"Could not import dependencies: {e}")
        
        # Test individual components
        logger.info("Testing language detection...")
        detector = LanguageDetector()
        
        test_texts = [
            "Hello, how are you?",
            "আপনি কেমন আছেন?",
            "বাংলাদেশের capital কী?"
        ]
        
        for text in test_texts:
            lang = detector.detect_language(text)
            logger.info(f"'{text}' -> {lang}")
        
        logger.info("Basic component tests completed") 