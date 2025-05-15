import os
import sys

# Set critical environment variables
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"
os.environ["PYTHONTRACEMALLOC"] = "0"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# Fix the asyncio event loop issue
import asyncio
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
except Exception as e:
    print(f"Note: Asyncio setup: {e}")

# Simply use sys.executable to run streamlit as a subprocess
import subprocess

print("Starting Streamlit app...")
print("Press Ctrl+C to stop the application")

try:
    # Use subprocess to run streamlit directly
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", 
                   "--server.headless=true", "--logger.level=error"], check=True)
except KeyboardInterrupt:
    print("Streamlit app stopped by user")
except Exception as e:
    print(f"Error running Streamlit app: {e}")
    sys.exit(1)
