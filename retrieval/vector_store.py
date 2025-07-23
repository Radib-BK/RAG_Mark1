"""
FAISS Vector Store for Multilingual RAG System

This module provides:
- FAISS-based vector storage and retrieval
- Efficient similarity search
- Metadata management for text chunks
- Index persistence and loading
- Support for different FAISS index types
- Batch operations for better performance
"""

import os
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import faiss
from loguru import logger

@dataclass
class RetrievalResult:
    """
    Data class for retrieval results
    """
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_index: int

class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search
    """
    
    def __init__(self, 
                 embedding_dimension: int,
                 index_type: str = "flat",
                 metric: str = "cosine",
                 store_dir: Optional[str] = None):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dimension: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('cosine', 'l2', 'ip')
            store_dir: Directory to store index and metadata
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.metric = metric
        self.store_dir = Path(store_dir) if store_dir else Path("./vector_store")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Storage for metadata and texts
        self.chunk_metadata = []
        self.chunk_texts = []
        self.chunk_ids = []
        
        # Index mapping
        self.id_to_index = {}  # chunk_id -> index in FAISS
        self.index_to_id = {}  # FAISS index -> chunk_id
        
        logger.info(f"Initialized FAISSVectorStore with {index_type} index, "
                   f"{metric} metric, dimension {embedding_dimension}")
    
    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index based on configuration
        
        Returns:
            FAISS index
        """
        if self.metric == "cosine":
            # For cosine similarity, we'll use inner product with normalized vectors
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(self.embedding_dimension)
            elif self.index_type == "ivf":
                # IVF with inner product
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, 100)
            elif self.index_type == "hnsw":
                # HNSW with inner product
                index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
        elif self.metric == "l2":
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(self.embedding_dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.embedding_dimension)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
        elif self.metric == "ip":
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(self.embedding_dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return index
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        if self.metric == "cosine":
            # Normalize for cosine similarity using inner product
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            return embeddings / norms
        return embeddings
    
    def add_embeddings(self, 
                      embeddings: np.ndarray,
                      chunk_ids: List[str],
                      texts: List[str],
                      metadata: List[Dict[str, Any]]):
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: Array of embedding vectors
            chunk_ids: List of chunk IDs
            texts: List of text content
            metadata: List of metadata dictionaries
        """
        if len(embeddings) != len(chunk_ids) != len(texts) != len(metadata):
            raise ValueError("All input lists must have the same length")
        
        logger.info(f"Adding {len(embeddings)} embeddings to vector store")
        
        # Normalize embeddings if using cosine similarity
        normalized_embeddings = self._normalize_embeddings(embeddings.astype(np.float32))
        
        # Get current index size for mapping
        current_size = self.index.ntotal
        
        # Train index if needed (for IVF indices)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(normalized_embeddings)
        
        # Add to FAISS index
        self.index.add(normalized_embeddings)
        
        # Update metadata and mappings
        for i, (chunk_id, text, meta) in enumerate(zip(chunk_ids, texts, metadata)):
            faiss_index = current_size + i
            
            self.chunk_ids.append(chunk_id)
            self.chunk_texts.append(text)
            self.chunk_metadata.append(meta)
            
            self.id_to_index[chunk_id] = faiss_index
            self.index_to_id[faiss_index] = chunk_id
        
        logger.info(f"Successfully added embeddings. Total vectors: {self.index.ntotal}")
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 10,
              threshold: Optional[float] = None) -> List[RetrievalResult]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of retrieval results
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Normalize query embedding
        query_embedding = self._normalize_embeddings(
            query_embedding.reshape(1, -1).astype(np.float32)
        )
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            # Convert distance to similarity score
            if self.metric == "cosine":
                score = float(distance)  # Inner product with normalized vectors = cosine similarity
            elif self.metric == "l2":
                score = 1.0 / (1.0 + float(distance))  # Convert L2 distance to similarity
            else:  # ip
                score = float(distance)
            
            # Apply threshold if specified
            if threshold is not None and score < threshold:
                continue
            
            # Get metadata
            chunk_id = self.index_to_id.get(idx, f"unknown_{idx}")
            chunk_index = int(idx)
            
            if chunk_index < len(self.chunk_texts):
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    content=self.chunk_texts[chunk_index],
                    score=score,
                    metadata=self.chunk_metadata[chunk_index],
                    chunk_index=chunk_index
                )
                results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"Retrieved {len(results)} results for query")
        return results
    
    def search_by_text(self, 
                      query_text: str,
                      embedder,
                      k: int = 10,
                      threshold: Optional[float] = None) -> List[RetrievalResult]:
        """
        Search using text query (requires embedder)
        
        Args:
            query_text: Query text
            embedder: Embedder instance to encode query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of retrieval results
        """
        query_embedding = embedder.encode_query(query_text)
        return self.search(query_embedding, k=k, threshold=threshold)
    
    def get_by_id(self, chunk_id: str) -> Optional[RetrievalResult]:
        """
        Get chunk by ID
        
        Args:
            chunk_id: Chunk ID to retrieve
            
        Returns:
            RetrievalResult if found, None otherwise
        """
        if chunk_id not in self.id_to_index:
            return None
        
        faiss_index = self.id_to_index[chunk_id]
        
        if faiss_index < len(self.chunk_texts):
            return RetrievalResult(
                chunk_id=chunk_id,
                content=self.chunk_texts[faiss_index],
                score=1.0,
                metadata=self.chunk_metadata[faiss_index],
                chunk_index=faiss_index
            )
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'total_chunks': len(self.chunk_texts),
            'index_size_mb': 0.0
        }
        
        # Calculate approximate index size
        if self.index.ntotal > 0:
            # Rough estimate: embedding_dim * num_vectors * 4 bytes (float32)
            # Plus metadata overhead
            embedding_size = (self.embedding_dimension * self.index.ntotal * 4) / (1024 * 1024)
            metadata_size = (len(self.chunk_texts) * 100) / (1024 * 1024)  # Rough metadata estimate
            stats['index_size_mb'] = round(embedding_size + metadata_size, 2)
        
        return stats
    
    def save_index(self, index_path: Optional[str] = None) -> str:
        """
        Save FAISS index and metadata to disk
        
        Args:
            index_path: Optional custom path for index file
            
        Returns:
            Path where index was saved
        """
        if index_path is None:
            index_path = self.store_dir / "faiss_index.idx"
        else:
            index_path = Path(index_path)
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'chunk_ids': self.chunk_ids,
                'chunk_texts': self.chunk_texts,
                'chunk_metadata': self.chunk_metadata,
                'id_to_index': self.id_to_index,
                'index_to_id': {str(k): v for k, v in self.index_to_id.items()},
                'embedding_dimension': self.embedding_dimension,
                'index_type': self.index_type,
                'metric': self.metric
            }
            
            metadata_path = index_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Also save as pickle for faster loading
            pickle_path = index_path.with_suffix('.metadata.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved to {index_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            return str(index_path)
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, index_path: str) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Args:
            index_path: Path to index file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = Path(index_path)
        
        if not index_path.exists():
            logger.error(f"Index file not found: {index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata (try pickle first, then JSON)
            pickle_path = index_path.with_suffix('.metadata.pkl')
            metadata_path = index_path.with_suffix('.metadata.json')
            
            metadata = None
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    metadata = pickle.load(f)
            elif metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                logger.error("No metadata file found")
                return False
            
            # Restore metadata
            self.chunk_ids = metadata['chunk_ids']
            self.chunk_texts = metadata['chunk_texts']
            self.chunk_metadata = metadata['chunk_metadata']
            self.id_to_index = metadata['id_to_index']
            self.index_to_id = {int(k): v for k, v in metadata['index_to_id'].items()}
            
            # Verify configuration matches
            if (metadata['embedding_dimension'] != self.embedding_dimension or
                metadata['index_type'] != self.index_type or
                metadata['metric'] != self.metric):
                logger.warning("Index configuration doesn't match current settings")
            
            logger.info(f"Successfully loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def clear(self):
        """Clear all data from the vector store"""
        self.index.reset()
        self.chunk_metadata.clear()
        self.chunk_texts.clear()
        self.chunk_ids.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        
        logger.info("Vector store cleared")
    
    def update_embedding(self, chunk_id: str, new_embedding: np.ndarray, 
                        new_text: str, new_metadata: Dict[str, Any]):
        """
        Update an existing embedding (requires rebuilding index)
        
        Args:
            chunk_id: ID of chunk to update
            new_embedding: New embedding vector
            new_text: New text content
            new_metadata: New metadata
        """
        if chunk_id not in self.id_to_index:
            logger.warning(f"Chunk ID {chunk_id} not found")
            return
        
        faiss_index = self.id_to_index[chunk_id]
        
        # Update stored data
        self.chunk_texts[faiss_index] = new_text
        self.chunk_metadata[faiss_index] = new_metadata
        
        # For FAISS, we need to rebuild the index to update embeddings
        # This is expensive, so we'll log a warning
        logger.warning("Updating embeddings requires rebuilding the index. "
                      "Consider batch updates for better performance.")
    
    def remove_by_id(self, chunk_id: str):
        """
        Remove chunk by ID (requires rebuilding index)
        
        Args:
            chunk_id: ID of chunk to remove
        """
        if chunk_id not in self.id_to_index:
            logger.warning(f"Chunk ID {chunk_id} not found")
            return
        
        faiss_index = self.id_to_index[chunk_id]
        
        # Remove from lists (this breaks index consistency)
        # For production use, consider using a soft-delete approach
        logger.warning("Removing items from FAISS index requires rebuilding. "
                      "Consider using filtering instead.")

def create_vector_store(embedding_dimension: int,
                       index_type: str = "flat",
                       metric: str = "cosine",
                       store_dir: Optional[str] = None) -> FAISSVectorStore:
    """
    Factory function to create vector store
    
    Args:
        embedding_dimension: Dimension of embeddings
        index_type: Type of FAISS index
        metric: Distance metric
        store_dir: Storage directory
        
    Returns:
        FAISSVectorStore instance
    """
    # Use default store directory if none provided
    if store_dir is None:
        store_dir = "vector_store"
    
    # Create vector store
    vector_store = FAISSVectorStore(
        embedding_dimension=embedding_dimension,
        index_type=index_type,
        metric=metric,
        store_dir=store_dir
    )
    
    # Try to load existing index if it exists
    index_path = Path(store_dir) / "faiss_index.idx"
    if index_path.exists():
        logger.info(f"Loading existing index from {index_path}")
        if vector_store.load_index(str(index_path)):
            logger.info(f"Successfully loaded existing index with {vector_store.index.ntotal} vectors")
        else:
            logger.warning("Failed to load existing index, starting with empty vector store")
    else:
        logger.info(f"No existing index found at {index_path}, starting with empty vector store")
    
    return vector_store

def build_vector_store_from_embeddings(embedding_data: Dict[str, Any],
                                     index_type: str = "flat",
                                     metric: str = "cosine",
                                     store_dir: Optional[str] = None) -> FAISSVectorStore:
    """
    Build vector store from embedding data
    
    Args:
        embedding_data: Dictionary with embeddings and metadata
        index_type: Type of FAISS index
        metric: Distance metric
        store_dir: Storage directory
        
    Returns:
        Populated FAISSVectorStore instance
    """
    # Create vector store
    vector_store = create_vector_store(
        embedding_dimension=embedding_data['embedding_dimension'],
        index_type=index_type,
        metric=metric,
        store_dir=store_dir
    )
    
    # Extract data
    embeddings = embedding_data['embeddings']
    texts = embedding_data['texts']
    metadata = embedding_data['metadata']
    
    # Generate chunk IDs if not present
    chunk_ids = [meta.get('chunk_id', f'chunk_{i}') for i, meta in enumerate(metadata)]
    
    # Add to vector store
    vector_store.add_embeddings(embeddings, chunk_ids, texts, metadata)
    
    return vector_store

def load_vector_store(store_dir: Optional[str] = None,
                     embedding_dimension: int = 384,
                     index_type: str = "flat",
                     metric: str = "cosine") -> FAISSVectorStore:
    """
    Load existing vector store or create empty one
    
    Args:
        store_dir: Storage directory (default: "vector_store")
        embedding_dimension: Dimension of embeddings
        index_type: Type of FAISS index
        metric: Distance metric
        
    Returns:
        FAISSVectorStore instance
    """
    if store_dir is None:
        store_dir = "vector_store"
    
    return create_vector_store(
        embedding_dimension=embedding_dimension,
        index_type=index_type,
        metric=metric,
        store_dir=store_dir
    )

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from embeddings.embed_model import create_embedder, TextChunkEmbedder
        from ingestion.extract_pdf import extract_pdf_content
        from ingestion.preprocess import preprocess_document
        
        pdf_path = "../data/hsc26.pdf"
        
        if Path(pdf_path).exists():
            try:
                # Process PDF and create embeddings
                logger.info("Processing PDF...")
                document_data = extract_pdf_content(pdf_path)
                chunks = preprocess_document(document_data, chunk_size=512)
                
                # Create embedder
                embedder = create_embedder(cache_dir="../cache/embeddings")
                chunk_embedder = TextChunkEmbedder(embedder)
                
                # Generate embeddings
                embedding_data = chunk_embedder.embed_chunks(chunks, batch_size=16)
                
                # Build vector store
                logger.info("Building vector store...")
                vector_store = build_vector_store_from_embeddings(
                    embedding_data,
                    index_type="flat",
                    metric="cosine",
                    store_dir="../vector_store"
                )
                
                # Save index
                index_path = vector_store.save_index()
                logger.info(f"Vector store saved to {index_path}")
                
                # Test search
                test_query = "বাংলাদেশের ইতিহাস"
                results = vector_store.search_by_text(test_query, embedder, k=5)
                
                logger.info(f"Found {len(results)} results for query: {test_query}")
                for i, result in enumerate(results):
                    logger.info(f"Result {i+1}: Score={result.score:.4f}, "
                              f"Content='{result.content[:100]}...'")
                
                # Show statistics
                stats = vector_store.get_statistics()
                logger.info(f"Vector store statistics: {stats}")
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                
        else:
            logger.info("Testing vector store with sample data...")
            
            # Create sample embeddings
            embedding_dim = 384  # E5-small dimension
            num_samples = 100
            
            sample_embeddings = np.random.rand(num_samples, embedding_dim).astype(np.float32)
            sample_texts = [f"Sample text {i}" for i in range(num_samples)]
            sample_metadata = [{'chunk_id': f'sample_{i}', 'source_page': i//10} for i in range(num_samples)]
            sample_ids = [f'sample_{i}' for i in range(num_samples)]
            
            # Create and test vector store
            vector_store = create_vector_store(
                embedding_dimension=embedding_dim,
                index_type="flat",
                metric="cosine"
            )
            
            vector_store.add_embeddings(sample_embeddings, sample_ids, sample_texts, sample_metadata)
            
            # Test search
            query_embedding = np.random.rand(1, embedding_dim).astype(np.float32)
            results = vector_store.search(query_embedding[0], k=5)
            
            logger.info(f"Found {len(results)} results for random query")
            for result in results:
                logger.info(f"Score: {result.score:.4f}, Content: {result.content}")
                
            # Test save/load
            save_path = vector_store.save_index("test_index.idx")
            logger.info(f"Index saved to {save_path}")
            
    except ImportError as e:
        logger.warning(f"Could not import dependencies: {e}")
        logger.info("Running basic vector store test...")
        
        # Basic test
        embedding_dim = 128
        vector_store = create_vector_store(embedding_dim)
        
        # Add sample data
        embeddings = np.random.rand(10, embedding_dim).astype(np.float32)
        ids = [f'test_{i}' for i in range(10)]
        texts = [f'Test text {i}' for i in range(10)]
        metadata = [{'index': i} for i in range(10)]
        
        vector_store.add_embeddings(embeddings, ids, texts, metadata)
        
        # Test search
        query = np.random.rand(embedding_dim).astype(np.float32)
        results = vector_store.search(query, k=3)
        
        logger.info(f"Basic test completed: {len(results)} results found")
        
        stats = vector_store.get_statistics()
        logger.info(f"Statistics: {stats}") 