# Multilingual RAG System for Educational Content

A robust, production-grade Retrieval-Augmented Generation (RAG) system designed for HSC-level educational content in both **Bangla** and **English**. This system uses only free and open-source tools and can run entirely locally without external APIs.

## 🌟 Features

- **Multilingual Support**: Handles both Bangla and English text seamlessly
- **Advanced PDF Processing**: Extracts clean text from complex HSC textbooks with MCQs and structured content
- **Intelligent Chunking**: Context-aware text segmentation with overlap for better retrieval
- **Multilingual Embeddings**: Uses `intfloat/multilingual-e5-small` for high-quality embeddings
- **Fast Vector Search**: FAISS-powered similarity search for efficient retrieval  
- **Language-Aware Generation**: Automatic model switching between `bongLlama` (Bangla) and `mistral-instruct` (English)
- **Conversation Memory**: LangChain-based memory management for contextual conversations
- **REST API**: FastAPI-based API with comprehensive endpoints
- **Comprehensive Evaluation**: ROUGE, BERTScore, and custom metrics for performance assessment
- **100% Local**: No external API dependencies - runs entirely on your machine

## 📁 Project Structure

```
RAG_Mark1/
├── data/                          # PDF files and processed data
│   └── hsc26.pdf                 # Place your HSC textbook here
├── embeddings/                    # Multilingual embedding models
│   ├── __init__.py
│   └── embed_model.py            # E5 multilingual embedder
├── ingestion/                     # Text extraction and processing
│   ├── __init__.py
│   ├── extract_pdf.py            # PDF text extraction
│   └── preprocess.py             # Text cleaning and chunking
├── retrieval/                     # Vector storage and search
│   ├── __init__.py
│   └── vector_store.py           # FAISS vector store
├── generation/                    # LLM integration and RAG chain
│   ├── __init__.py
│   └── rag_chain.py              # LangChain RAG pipeline
├── api/                          # REST API server
│   ├── __init__.py
│   └── app.py                    # FastAPI application
├── evaluation/                    # Performance evaluation
│   ├── __init__.py
│   └── eval_metrics.py           # Evaluation metrics
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM (16GB recommended)
- CUDA GPU (optional, for faster processing)

### 2. Install Ollama Models

```bash
# Install required models
ollama pull bongllama              # For Bangla text generation
ollama pull mistral:instruct       # For English text generation

# Verify models are available
ollama list
```

### 3. Setup Python Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG_Mark1

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Add Your PDF

Place your HSC textbook PDF in the `data/` directory:
```bash
cp /path/to/your/hsc_textbook.pdf data/hsc26.pdf
```

### 5. Run the System

#### Option A: Run as API Server
```bash
# Start the FastAPI server
python -m api.app --host 0.0.0.0 --port 8000

# Access the API at http://localhost:8000
# View interactive docs at http://localhost:8000/docs
```

#### Option B: Use Individual Components
```bash
# Extract and process PDF
cd ingestion
python extract_pdf.py

# Generate embeddings
cd ../embeddings  
python embed_model.py

# Build vector store
cd ../retrieval
python vector_store.py

# Test RAG chain
cd ../generation
python rag_chain.py
```

## 📝 API Usage

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "বাংলাদেশের রাজধানী কী?",
       "max_chunks": 5,
       "threshold": 0.3,
       "include_sources": true
     }'
```

### Example Response

```json
{
  "answer": "বাংলাদেশের রাজধানী হলো ঢাকা।",
  "query_language": "bn",
  "model_used": "bn:bongllama",
  "confidence_score": 0.85,
  "sources": [
    {
      "chunk_id": "page_1_para_2",
      "content": "বাংলাদেশের রাজধানী ঢাকা...",
      "score": 0.92,
      "metadata": {
        "source_page": 1,
        "content_type": "paragraph"
      }
    }
  ],
  "response_time_ms": 1250,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### Other Endpoints

- `GET /health` - System health check
- `GET /stats` - System statistics  
- `GET /config` - Current configuration
- `POST /clear-memory` - Clear conversation history
- `GET /conversation-history` - Get chat history
- `GET /docs` - Interactive API documentation

## 🔧 Configuration

### Environment Variables

```bash
# Optional configuration
export PDF_PATH="./data/hsc26.pdf"
export VECTOR_STORE_DIR="./vector_store" 
export CACHE_DIR="./cache"
export BANGLA_MODEL="bongllama"
export ENGLISH_MODEL="mistral:instruct"
export OLLAMA_URL="http://localhost:11434"
export CHUNK_SIZE="512"
export EMBEDDING_MODEL="intfloat/multilingual-e5-small"
```

### Custom Configuration File

```bash
# Run with custom config
python -m api.app --config-file config.json
```

Example `config.json`:
```json
{
  "pdf_path": "./data/custom_book.pdf",
  "chunk_size": 256,
  "bangla_model": "custom-bangla-model",
  "english_model": "custom-english-model"
}
```

## 📊 Evaluation

### Run Evaluation

```python
from evaluation.eval_metrics import RAGEvaluator, create_sample_test_set
from embeddings.embed_model import create_embedder
from generation.rag_chain import create_rag_chain

# Create evaluator
embedder = create_embedder()
evaluator = RAGEvaluator(embedder=embedder)

# Load your test set
test_cases = create_sample_test_set()

# Run evaluation
summary = evaluator.evaluate_test_set(
    test_cases, 
    rag_chain, 
    output_file="evaluation_results.json"
)

print(f"Average ROUGE-L: {summary.avg_rouge_l:.3f}")
print(f"Average Semantic Similarity: {summary.avg_semantic_similarity:.3f}")
print(f"Average Answer Relevance: {summary.avg_answer_relevance:.3f}")
```

### Evaluation Metrics

- **Retrieval Metrics**: Precision, Recall, F1, MRR
- **Generation Metrics**: ROUGE-1/2/L, BERTScore, Semantic Similarity
- **Custom Metrics**: Answer Relevance, Factual Accuracy, Language Consistency
- **Performance Metrics**: Response Time, Confidence Scores

## 🛠️ Development

### Project Structure Details

#### **Ingestion Module** (`ingestion/`)
- `extract_pdf.py`: Handles complex PDF extraction with OCR cleanup
- `preprocess.py`: Intelligent text chunking with language awareness

#### **Embeddings Module** (`embeddings/`)
- `embed_model.py`: Multilingual E5 embedder with caching

#### **Retrieval Module** (`retrieval/`) 
- `vector_store.py`: FAISS vector store with metadata management

#### **Generation Module** (`generation/`)
- `rag_chain.py`: LangChain pipeline with automatic language switching

#### **API Module** (`api/`)
- `app.py`: FastAPI server with comprehensive endpoints

#### **Evaluation Module** (`evaluation/`)
- `eval_metrics.py`: Complete evaluation framework

### Adding New Languages

1. **Update Language Detection**:
   ```python
   # In generation/rag_chain.py
   self.language_keywords = {
       'your_language': ['keyword1', 'keyword2', ...]
   }
   ```

2. **Add Ollama Model**:
   ```bash
   ollama pull your-language-model
   ```

3. **Update Prompt Templates**:
   ```python
   # In generation/rag_chain.py  
   your_language_template = """Your prompt template..."""
   ```

### Custom Chunking Strategies

```python
# In ingestion/preprocess.py
def chunk_by_custom_strategy(self, text: str) -> List[TextChunk]:
    # Implement your custom chunking logic
    pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama service if needed
   ollama serve
   ```

2. **Out of Memory During Embedding**
   ```python
   # Reduce batch size in embed_model.py
   embeddings = embedder.encode_batch(texts, batch_size=8)
   ```

3. **PDF Extraction Issues**
   ```python
   # Try alternative extraction method
   document_data = extract_pdf_content(pdf_path, force_pypdf2=True)
   ```

4. **Slow Vector Search**
   ```python
   # Use IVF index for large datasets
   vector_store = create_vector_store(
       embedding_dimension=384,
       index_type="ivf"  # Instead of "flat"
   )
   ```

### Performance Optimization

1. **GPU Acceleration**:
   - Install `torch` with CUDA support
   - Set `device="cuda"` in embedder initialization

2. **Memory Management**:
   - Reduce chunk size for lower memory usage
   - Use streaming for large PDF processing

3. **Caching**:
   - Embeddings are automatically cached
   - Vector indices are persisted to disk

## 📚 Advanced Usage

### Batch Processing Multiple PDFs

```python
from pathlib import Path
from ingestion.extract_pdf import extract_pdf_content
from ingestion.preprocess import preprocess_document
from embeddings.embed_model import TextChunkEmbedder, create_embedder

pdf_dir = Path("./data/books/")
embedder = create_embedder()
chunk_embedder = TextChunkEmbedder(embedder)

all_chunks = []
for pdf_path in pdf_dir.glob("*.pdf"):
    document_data = extract_pdf_content(str(pdf_path))
    chunks = preprocess_document(document_data)
    all_chunks.extend(chunks)

# Create embeddings for all chunks
embedding_data = chunk_embedder.embed_chunks(all_chunks)
```

### Custom Evaluation Metrics

```python
from evaluation.eval_metrics import RAGEvaluator

class CustomEvaluator(RAGEvaluator):
    def custom_metric(self, predicted: str, reference: str) -> float:
        # Implement your custom evaluation logic
        return score
```

### Integration with Other Tools

```python
# Use with Jupyter notebooks
from IPython.display import display, Markdown

def display_rag_response(response):
    display(Markdown(f"**Answer**: {response.answer}"))
    display(Markdown(f"**Confidence**: {response.confidence_score:.2%}"))
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the multilingual E5 embedding model
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [LangChain](https://langchain.com/) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM serving
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API framework

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the [documentation](http://localhost:8000/docs) when running the API
- Review the [troubleshooting](#-troubleshooting) section

---

**Built with ❤️ for multilingual education**