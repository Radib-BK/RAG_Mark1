"""
Multilingual Embedding Model Module for RAG System

This module provides:
- Multilingual text embedding using intfloat/multilingual-e5-small
- Batch processing for efficient embedding generation
- Text normalization and preprocessing for embeddings
- Support for both Bangla and English text
- Caching for improved performance
"""

import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
import json

class MultilingualEmbedder:
    """
    Multilingual text embedder using Sentence Transformers
    """
    
    def __init__(self, 
                 model_name: str = "intfloat/multilingual-e5-small",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize multilingual embedder
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cpu', 'cuda', 'mps')
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing embedder with model: {model_name} on device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.load_cache()
    
    def load_cache(self):
        """Load cached embeddings from disk"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
                self.embedding_cache = {}
    
    def save_cache(self):
        """Save embeddings cache to disk"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # For E5 models, add instruction prefix for better performance
        if not text.startswith("query:") and not text.startswith("passage:"):
            # Use "passage:" for document chunks, "query:" for questions
            text = f"passage: {text}"
        
        return text
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode single text into embedding vector
        
        Args:
            text: Text to encode
            normalize: Whether to normalize the embedding
            
        Returns:
            Embedding vector
        """
        if not text:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        try:
            # Generate embedding
            embedding = self.model.encode([processed_text], 
                                        normalize_embeddings=normalize,
                                        show_progress_bar=False)[0]
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def encode_batch(self, texts: List[str], 
                    batch_size: int = 32,
                    normalize: bool = True,
                    show_progress: bool = True) -> np.ndarray:
        """
        Encode batch of texts into embedding vectors
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")
        
        embeddings = []
        cached_count = 0
        to_encode = []
        to_encode_indices = []
        
        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            if not text:
                embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))
                continue
                
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                cached_count += 1
            else:
                embeddings.append(None)  # Placeholder
                to_encode.append(self.preprocess_text(text))
                to_encode_indices.append(i)
        
        logger.info(f"Found {cached_count} cached embeddings, encoding {len(to_encode)} new texts")
        
        # Encode new texts in batches
        if to_encode:
            try:
                new_embeddings = self.model.encode(to_encode,
                                                 batch_size=batch_size,
                                                 normalize_embeddings=normalize,
                                                 show_progress_bar=show_progress)
                
                # Update embeddings array and cache
                for i, embedding in enumerate(new_embeddings):
                    original_index = to_encode_indices[i]
                    embeddings[original_index] = embedding
                    
                    # Cache the result
                    cache_key = self._get_cache_key(texts[original_index])
                    self.embedding_cache[cache_key] = embedding
                
            except Exception as e:
                logger.error(f"Error in batch encoding: {str(e)}")
                # Fill with zeros for failed encodings
                for i in to_encode_indices:
                    if embeddings[i] is None:
                        embeddings[i] = np.zeros(self.model.get_sentence_embedding_dimension())
        
        return np.array(embeddings)
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode query text (with query prefix for E5 models)
        
        Args:
            query: Query text
            normalize: Whether to normalize embedding
            
        Returns:
            Query embedding vector
        """
        if not query:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # For queries, use "query:" prefix
        query_text = f"query: {query.strip()}"
        
        try:
            embedding = self.model.encode([query_text],
                                        normalize_embeddings=normalize,
                                        show_progress_bar=False)[0]
            return embedding
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize vectors if not already normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Embedding cache cleared")

class TextChunkEmbedder:
    """
    Specialized embedder for text chunks with metadata
    """
    
    def __init__(self, embedder: MultilingualEmbedder):
        """
        Initialize chunk embedder
        
        Args:
            embedder: MultilingualEmbedder instance
        """
        self.embedder = embedder
    
    def embed_chunks(self, chunks: List[Any], 
                    batch_size: int = 32,
                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Embed text chunks with metadata
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for embedding
            save_path: Optional path to save embeddings
            
        Returns:
            Dictionary with embeddings and metadata
        """
        logger.info(f"Embedding {len(chunks)} text chunks")
        
        # Extract texts and metadata
        texts = [chunk.content for chunk in chunks]
        chunk_metadata = []
        
        for chunk in chunks:
            metadata = {
                'chunk_id': chunk.chunk_id,
                'source_page': chunk.source_page,
                'language': chunk.language,
                'content_type': chunk.content_type,
                'word_count': chunk.word_count,
                'char_count': chunk.char_count
            }
            chunk_metadata.append(metadata)
        
        # Generate embeddings
        embeddings = self.embedder.encode_batch(texts, batch_size=batch_size)
        
        # Create result dictionary
        result = {
            'embeddings': embeddings,
            'metadata': chunk_metadata,
            'texts': texts,
            'embedding_dimension': self.embedder.get_embedding_dimension(),
            'model_name': self.embedder.model_name,
            'total_chunks': len(chunks)
        }
        
        # Save if requested
        if save_path:
            self.save_embeddings(result, save_path)
        
        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return result
    
    def save_embeddings(self, embedding_data: Dict[str, Any], save_path: str):
        """
        Save embeddings to disk
        
        Args:
            embedding_data: Dictionary containing embeddings and metadata
            save_path: Path to save the data
        """
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save as numpy compressed format
            np.savez_compressed(
                save_file.with_suffix('.npz'),
                embeddings=embedding_data['embeddings'],
                texts=np.array(embedding_data['texts'], dtype=object),
                metadata=np.array(embedding_data['metadata'], dtype=object)
            )
            
            # Save metadata as JSON for easy reading
            metadata_file = save_file.with_suffix('.json')
            metadata = {
                'embedding_dimension': embedding_data['embedding_dimension'],
                'model_name': embedding_data['model_name'],
                'total_chunks': embedding_data['total_chunks'],
                'metadata': embedding_data['metadata']
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Embeddings saved to {save_file.with_suffix('.npz')}")
            logger.info(f"Metadata saved to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    def load_embeddings(self, load_path: str) -> Dict[str, Any]:
        """
        Load embeddings from disk
        
        Args:
            load_path: Path to load embeddings from
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        load_file = Path(load_path)
        
        try:
            # Load embeddings
            data = np.load(load_file.with_suffix('.npz'), allow_pickle=True)
            
            # Load metadata
            metadata_file = load_file.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            result = {
                'embeddings': data['embeddings'],
                'texts': data['texts'].tolist(),
                'metadata': data['metadata'].tolist(),
                'embedding_dimension': metadata.get('embedding_dimension', 
                                                   data['embeddings'].shape[1] if len(data['embeddings']) > 0 else 0),
                'model_name': metadata.get('model_name', 'unknown'),
                'total_chunks': metadata.get('total_chunks', len(data['embeddings']))
            }
            
            logger.info(f"Loaded embeddings from {load_file.with_suffix('.npz')}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

def create_embedder(model_name: str = "intfloat/multilingual-e5-small",
                   device: Optional[str] = None,
                   cache_dir: Optional[str] = None) -> MultilingualEmbedder:
    """
    Factory function to create embedder instance
    
    Args:
        model_name: Model name
        device: Device to use
        cache_dir: Cache directory
        
    Returns:
        MultilingualEmbedder instance
    """
    return MultilingualEmbedder(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir
    )

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    
    # Add parent directory to path to import other modules
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from ingestion.extract_pdf import extract_pdf_content
        from ingestion.preprocess import preprocess_document
        
        pdf_path = "../data/hsc26.pdf"
        
        if Path(pdf_path).exists():
            try:
                # Extract and preprocess PDF
                logger.info("Extracting PDF content...")
                document_data = extract_pdf_content(pdf_path)
                
                logger.info("Preprocessing text...")
                chunks = preprocess_document(document_data, chunk_size=256)
                
                # Create embedder
                logger.info("Creating embedder...")
                embedder = create_embedder(cache_dir="../cache/embeddings")
                
                # Create chunk embedder
                chunk_embedder = TextChunkEmbedder(embedder)
                
                # Embed chunks
                logger.info("Generating embeddings...")
                embedding_data = chunk_embedder.embed_chunks(
                    chunks, 
                    batch_size=16,
                    save_path="../data/embeddings/chunk_embeddings"
                )
                
                logger.info(f"Generated {len(embedding_data['embeddings'])} embeddings")
                logger.info(f"Embedding dimension: {embedding_data['embedding_dimension']}")
                
                # Test query embedding
                test_query = "বাংলাদেশের ইতিহাস কি?"
                query_embedding = embedder.encode_query(test_query)
                logger.info(f"Query embedding shape: {query_embedding.shape}")
                
                # Save cache
                embedder.save_cache()
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
        else:
            logger.info(f"PDF file not found: {pdf_path}")
            logger.info("Testing embedder with sample text...")
            
            # Test with sample texts
            embedder = create_embedder()
            
            sample_texts = [
                "This is a sample English text for testing.",
                "এটি একটি নমুনা বাংলা টেক্সট পরীক্ষার জন্য।",
                "Mixed language text with both English and বাংলা content."
            ]
            
            embeddings = embedder.encode_batch(sample_texts)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            # Test similarity
            similarity = embedder.compute_similarity(embeddings[0], embeddings[1])
            logger.info(f"Similarity between texts: {similarity:.4f}")
    
    except ImportError:
        logger.warning("Could not import ingestion modules, running basic test...")
        
        # Basic test without dependencies
        embedder = create_embedder()
        
        test_text = "This is a test sentence."
        embedding = embedder.encode_single(test_text)
        logger.info(f"Test embedding shape: {embedding.shape}")
        
        embedder.save_cache() 