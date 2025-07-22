#!/usr/bin/env python3
"""
Simple runner script for HSC Bangla RAG System

This script helps you start the system components easily.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_ollama():
    """Check if Ollama is available and has required model"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            models = result.stdout
            has_aya = 'aya-expanse' in models
            
            if not has_aya:
                print("⚠️  Missing required Ollama model!")
                print("Please install:")
                print("  ollama pull aya-expanse:8b")
                return False
            
            print("✅ Ollama model available")
            return True
        else:
            print("❌ Ollama not responding. Please start Ollama first.")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install from https://ollama.ai/")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def check_dependencies():
    """Check if Python dependencies are installed"""
    required_modules = ['fastapi', 'streamlit', 'langchain', 'sentence_transformers']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ Python dependencies available")
    return True

def check_pdf():
    """Check if PDF file exists"""
    pdf_path = Path("data/HSC26-Bangla1st-Paper.pdf")
    if pdf_path.exists():
        print("✅ HSC PDF found")
        return True
    else:
        print("❌ HSC PDF not found at data/HSC26-Bangla1st-Paper.pdf")
        print("Please place your HSC PDF in the data/ directory")
        return False

def run_api_server():
    """Start the API server"""
    print("\n🚀 Starting API server...")
    try:
        subprocess.run([sys.executable, '-m', 'api.app'], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def run_streamlit():
    """Start the Streamlit interface"""
    print("\n🌐 Starting Streamlit interface...")
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Streamlit stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def run_both():
    """Run both API server and Streamlit in separate processes"""
    import threading
    import signal
    
    def signal_handler(signum, frame):
        print("\n⏹️  Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Wait a bit for API to start
    print("⏳ Waiting for API server to start...")
    time.sleep(5)
    
    # Start Streamlit
    print("🌐 Opening web interface...")
    time.sleep(2)
    webbrowser.open('http://localhost:8501')
    
    run_streamlit()

def main():
    print("🎓 HSC Bangla RAG System Setup")
    print("=" * 40)
    
    # Check all prerequisites
    if not check_ollama():
        return
    
    if not check_dependencies():
        return
    
    if not check_pdf():
        return
    
    print("\n✅ All prerequisites met!")
    print("\nChoose an option:")
    print("1. Run Web Interface (Recommended)")
    print("2. Run API Server Only") 
    print("3. Run Streamlit Only")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting complete system...")
            run_both()
        elif choice == '2':
            run_api_server()
        elif choice == '3':
            run_streamlit()
        elif choice == '4':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Please run again.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 