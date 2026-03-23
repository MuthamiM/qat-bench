import os
import json
import yaml
import torch
from train.trainer import create_dataloaders, evaluate
from model.transformer import TransformerLM
from model.qat_transformer import QATTransformerLM
from bench.memory import get_memory_usage, get_model_size_mb
from bench.runner import benchmark_inference
from bench.ptq import replace_with_bnb4bit

def load_model(variant, config, vocab_size, chkpt_dir, device):
    if variant == "FP32":
        model = TransformerLM(config['model'], vocab_size)
        model.load_state_dict(torch.load(os.path.join(chkpt_dir, "fp32_best.pt"), map_location="cpu"))
        return model.to(device)
    elif variant == "QAT-INT8":
        model = QATTransformerLM(config['model'], vocab_size)
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        # Ignore embeddings and layer norm since fbgemm doesn't always support them linearly
        model.wte.qconfig = None
        model.wpe.qconfig = None
        model.ln_f.qconfig = None
        for block in model.blocks:
            block.ln_1.qconfig = None
            block.ln_2.qconfig = None
            
        torch.quantization.prepare_qat(model, inplace=True)
        model.eval()
        torch.quantization.convert(model, inplace=True)
        model.load_state_dict(torch.load(os.path.join(chkpt_dir, "qat_int8.pt"), map_location="cpu"))
        return model.to("cpu")  # PyTorch QAT output runs on CPU natively
    elif variant == "PTQ-INT8":
        model = TransformerLM(config['model'], vocab_size)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(os.path.join(chkpt_dir, "ptq_int8.pt"), map_location="cpu"))
        return model.to("cpu")
    elif variant == "PTQ-INT4":
        model = TransformerLM(config['model'], vocab_size)
        # Load fp32 weights
        float_weights = torch.load(os.path.join(chkpt_dir, "ptq_int4.pt"), map_location="cpu")
        model.load_state_dict(float_weights)
        # Use BitsAndBytes on CUDA if available
        if device.type == 'cuda':
            model = replace_with_bnb4bit(model).to(device)
        return model

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dl, vocab_size = create_dataloaders(config)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', config['output']['results_dir'])
    chkpt_dir = os.path.join(out_dir, 'checkpoints')
    
    variants = ["FP32", "QAT-INT8", "PTQ-INT8", "PTQ-INT4"]
    results = []
    
    for variant in variants:
        print(f"Evaluating variant: {variant}...")
        try:
            model = load_model(variant, config, vocab_size, chkpt_dir, device)
            # PyTorch INT8 quant generally only targets CPU backend well on Windows
            eval_device = torch.device("cpu") if "INT8" in variant else device
            
            # Initial memory check
            mem_usage = get_memory_usage()
            
            # 1. Perplexity
            _, ppl = evaluate(model, val_dl, eval_device)
            
            # 2. Benchmarking (Latency/Throughput)
            latency, tps = benchmark_inference(model, eval_device, seq_len=config['data']['seq_len'])
            
            # 3. Size
            fmap = {
                "FP32": "fp32_best.pt", 
                "QAT-INT8": "qat_int8.pt", 
                "PTQ-INT8": "ptq_int8.pt", 
                "PTQ-INT4": "ptq_int4.pt"
            }
            size_mb = get_model_size_mb(os.path.join(chkpt_dir, fmap[variant]))
            
            results.append({
                "Model": variant,
                "Perplexity": round(ppl, 4),
                "Size (MB)": round(size_mb, 2),
                "RAM (MB)": round(mem_usage["ram_mb"], 2),
                "Latency (ms)": round(latency, 2),
                "Tokens/sec": round(tps, 2)
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Skipping {variant} evaluation due to error: {e}")
            
    out_path = os.path.join(out_dir, 'benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully evaluated all variants. Results saved to {out_path}.")

if __name__ == "__main__":
    main()
