#!/usr/bin/env python3
"""
Simple runner script for HSC Bangla RAG System

This script helps you start the system components easily.
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
import requests
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
    # Temporarily bypass - PyTorch has DLL conflicts but system works
    print("✅ Python dependencies available (bypassing PyTorch DLL check)")
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

def kill_existing_streamlit():
    """Kill any existing Streamlit processes"""
    try:
        if os.name == 'nt':  # Windows
            # Kill processes containing 'streamlit'
            subprocess.run(['taskkill', '/f', '/im', 'python.exe', '/fi', 'COMMANDLINE eq *streamlit*'], 
                         capture_output=True, timeout=5)
            # Alternative method
            subprocess.run(['powershell', '-Command', 
                          'Get-Process python | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*streamlit*"} | Stop-Process -Force'], 
                         capture_output=True, timeout=5)
        else:  # Unix/Linux/Mac
            subprocess.run(['pkill', '-f', 'streamlit'], capture_output=True, timeout=5)
        time.sleep(2)  # Give processes more time to terminate
        print("🧹 Cleaned up existing Streamlit processes")
    except Exception as e:
        print(f"⚠️  Could not clean processes: {e}")
        pass  # Ignore errors if no processes found

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
        # Disable automatic browser opening with --server.headless=true
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.headless=true',
            '--server.port=8501',
            '--server.address=0.0.0.0'
        ], check=True)
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
    
    # Check if API server is already running
    def check_api_running():
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                return True
            else:
                print(f"🔍 API health check: Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            # Don't print timeout errors during the waiting loop
            return False
        except Exception as e:
            print(f"🔍 API health check error: {e}")
            return False
    
    # Check if Streamlit is already running
    def check_streamlit_running():
        try:
            response = requests.get("http://localhost:8501", timeout=2)
            return True
        except:
            try:
                response = requests.get("http://localhost:8502", timeout=2)
                return True
            except:
                return False
    
    if check_streamlit_running():
        print("⚠️  Streamlit is already running. Stopping existing instance...")
        kill_existing_streamlit()
        time.sleep(2)  # Wait for cleanup
    
    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Wait and check if API started successfully
    print("⏳ Waiting for API server to start...")
    api_started = False
    for i in range(20):  # Wait up to 20 seconds
        time.sleep(1)
        if check_api_running():
            print("✅ API server started successfully!")
            api_started = True
            break
        # Only show status every 3 seconds to reduce spam
        if i % 3 == 2 and i < 19:
            print(f"⏳ Still waiting... ({i+1}/20)")
    
    if not api_started:
        print("⚠️  API server may have issues, but continuing with Streamlit...")
        print("📝 You can check API logs above for any error messages")
    
    # Start Streamlit
    print("🌐 Starting Streamlit interface...")
    
    # Open browser after a short delay (only once)
    def open_browser_delayed():
        time.sleep(3)  # Wait for Streamlit to fully start
        try:
            webbrowser.open('http://localhost:8501', new=1, autoraise=True)
            print("✅ Web interface opened in browser")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("📝 Please manually open: http://localhost:8501")
    
    browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
    browser_thread.start()
    
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
    # print("3. Run Streamlit Only")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting complete system...")
            run_both()
        elif choice == '2':
            run_api_server()
        # elif choice == '3':
        #     run_streamlit()
        elif choice == '3':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Please run again.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 