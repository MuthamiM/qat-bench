import time
import torch

@torch.no_grad()
def benchmark_inference(model, device, seq_len=128, iterations=100):
    model.eval()
    dummy_input = torch.randint(0, 100, (1, seq_len), device=device)
    
    # Warmup phase
    for _ in range(10):
        model(dummy_input)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(iterations):
        model(dummy_input)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    tokens_per_sec = (iterations * seq_len) / total_time
    
    return avg_latency_ms, tokens_per_sec
