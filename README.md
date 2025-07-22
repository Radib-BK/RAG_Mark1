# HSC Bangla RAG System

A multilingual RAG (Retrieval-Augmented Generation) system for HSC educational content in **Bangla** and **English**. Optimized for HSC textbook structure with paragraphs → tables → MCQs → answer tables.

## 🚀 Quick Start

### Step 1: Install Ollama & Model
1. Install [Ollama](https://ollama.ai/) 
2. Open terminal and run:
```bash
ollama pull aya-expanse:8b     # Multilingual model for both Bangla & English
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the System

**🎯 Super Easy Way (Windows):**
Double-click `start.bat` - it handles everything!

**🖥️ Command Line Way:**
```bash
python run.py
```
Choose option 1 for the web interface.

**🌐 Manual Way:**
```bash
# Terminal 1: API Server
python -m api.app

# Terminal 2: Web Interface  
streamlit run streamlit_app.py
```

**✅ That's it!** Open http://localhost:8501 and start asking questions!

## 📋 Sample Questions to Try (Based on HSC Textbook)

**Bangla Questions:**
- অনুপমের বয়স কত বছর?
- অনুপমের মামার চরিত্রের বৈশিষ্ট্য কী?
- গল্পে 'ফল্গুর বালির মতো' বলতে কী বোঝানো হয়েছে?

**English Questions:**
- What is the main theme of 'Aparichita' story?
- Who is Harish in the story and what role does he play?
- What does the gold testing scene reveal about the character dynamics?

## 🔧 System Features

- **Smart PDF Processing**: Optimized for HSC note structure (paragraphs → tables → MCQs → answers)
- **Multilingual**: Handles both Bangla and English questions with single powerful model
- **Local LLM**: Uses `aya-expanse:8b` for excellent multilingual performance via Ollama
- **Fast Search**: FAISS vector store with E5 multilingual embeddings
- **Web Interface**: Simple Streamlit frontend for easy testing
- **REST API**: FastAPI backend for integration

## 🛠️ How It Works

1. **PDF Processing**: Extracts text from your HSC PDF, identifying paragraphs, tables, MCQs, and answer sections
2. **Smart Chunking**: Groups related content (paragraph + table + MCQs) for better context
3. **Multilingual Embeddings**: Creates semantic vectors using E5 model for both Bangla and English
4. **Vector Search**: Finds relevant content using FAISS similarity search
5. **Answer Generation**: Uses `aya-expanse:8b` via Ollama to generate contextual answers in the question language

## 📊 System Status

Check these endpoints:
- **Web UI**: http://localhost:8501 (Streamlit interface)
- **API Docs**: http://localhost:8000/docs (Interactive API documentation)
- **Health Check**: http://localhost:8000/health (System status)

## 🚨 Troubleshooting

**Ollama not working?**
```bash
# Check if Ollama is running
ollama list

# If not installed, install from https://ollama.ai/
```

**Web interface not loading?**
- Make sure API server is running first: `python -m api.app`
- Then start Streamlit: `streamlit run streamlit_app.py`

**Out of memory?**
- Close other applications
- Reduce batch size in processing

**Still having issues?**
- Check system status: http://localhost:8000/health
- View API docs: http://localhost:8000/docs

---

**🎯 Ready to explore HSC Bangla literature with AI!**