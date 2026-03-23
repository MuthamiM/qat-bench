import os
import time
import json
import sqlite3
import threading
from functools import lru_cache
import torch
import yaml
import uvicorn
import webbrowser
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

PORT = 8501
app = FastAPI()

base_dir = os.path.dirname(os.path.abspath(__file__))

# Database setup
db_path = os.path.join(base_dir, 'results', 'telemetry.db')
conn = sqlite3.connect(db_path, check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS metrics
             (timestamp REAL, model TEXT, latency REAL, throughput REAL)''')
conn.commit()

# Global state
target_model_name = "QAT-INT8"
active_model_name = None
model_instance = None

with open(os.path.join(base_dir, 'configs', 'config.yaml')) as f:
    config = yaml.safe_load(f)

live_history = []
MAX_HISTORY = 12

def live_inference_loop():
    global active_model_name, model_instance, target_model_name, live_history
    print("Initializing live inference engine...")
    from bench.evaluator import load_model
    device = torch.device('cpu')
    dummy_input = torch.randint(0, 1014, (1, 64), device=device)

    while True:
        try:
            if target_model_name != active_model_name:
                print(f"Hot-swapping live model to {target_model_name}...")
                model_instance = load_model(target_model_name, config, 1014, os.path.join(base_dir, 'results', 'checkpoints'), device)
                model_instance.eval()
                active_model_name = target_model_name
                live_history.clear()

            if model_instance is not None:
                start = time.perf_counter()
                with torch.no_grad():
                    model_instance(dummy_input)
                end = time.perf_counter()
                
                latency = (end - start) * 1000
                tokens_sec = 64 / (end - start)
                
                live_history.append(tokens_sec)
                if len(live_history) > MAX_HISTORY:
                    live_history.pop(0)
                    
                # DB LOGGING: Save telemetry history reliably to local disk
                conn.execute("INSERT INTO metrics VALUES (?, ?, ?, ?)", (time.time(), active_model_name, latency, tokens_sec))
                conn.commit()
                
                # Invalidate in-memory cache instantly 
                get_cached_history.cache_clear()

        except Exception as e:
            print(f"Inference error: {e}")
            
        time.sleep(5)

# High-performance in-memory cache
@lru_cache(maxsize=1)
def get_cached_history():
    return list(live_history), active_model_name

# Unified API endpoints
@app.get("/api/live")
def get_live():
    hist, model_name = get_cached_history()
    return JSONResponse({'history': hist, 'active_model': model_name})

@app.get("/api/set_model")
def set_model(m: str):
    global target_model_name
    target_model_name = m
    return {"status": "swapping"}

@app.get("/api/config")
def get_config():
    return JSONResponse(config)

@app.get("/api/report")
def get_report():
    report_path = os.path.join(base_dir, 'results', 'report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"error": "report missing"})

# Mount local directories gracefully
app.mount("/results", StaticFiles(directory=os.path.join(base_dir, "results")), name="results")
app.mount("/", StaticFiles(directory=os.path.join(base_dir, "web_dashboard"), html=True), name="static")

if __name__ == "__main__":
    inference_thread = threading.Thread(target=live_inference_loop, daemon=True)
    inference_thread.start()
    
    # Give uvicorn a moment to mount before auto-loading the URL
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
