import os
import torch
import torch.nn as nn
import yaml
import json
import bitsandbytes as bnb
from model.transformer import TransformerLM

def replace_with_bnb4bit(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            has_bias = module.bias is not None
            new_layer = bnb.nn.Linear4bit(
                module.in_features, 
                module.out_features, 
                bias=has_bias, 
                compute_dtype=torch.float32,
                quant_type="nf4"
            )
            setattr(model, name, new_layer)
        else:
            replace_with_bnb4bit(module)
    return model

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    data_dir = os.path.join(os.path.dirname(__file__), '..', config['output']['results_dir'], 'data')
    with open(os.path.join(data_dir, 'tokenizer.json'), 'r') as f:
        vocab_size = json.load(f)['vocab_size']
        
    chkpt_dir = os.path.join(os.path.dirname(__file__), '..', config['output']['checkpoint_dir'])
    fp32_path = os.path.join(chkpt_dir, "fp32_best.pt")
    
    if not os.path.exists(fp32_path):
        print("FP32 model not found. Run train mode first.")
        return

    # Load FP32 model
    model = TransformerLM(config['model'], vocab_size)
    model.load_state_dict(torch.load(fp32_path, map_location="cpu"))
    model.eval()

    # 1. PTQ Dynamic INT8
    print("Applying PTQ Dynamic INT8...")
    ptq_int8_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(ptq_int8_model.state_dict(), os.path.join(chkpt_dir, "ptq_int8.pt"))
    print("Saved PTQ-INT8 model.")

    # 2. PTQ INT4 (BitsAndBytes)
    print("Applying PTQ INT4 (bitsandbytes structure mapping)...")
    # For BitsAndBytes, the active quantization happens on CUDA injection. 
    # We save the state dict such that the evaluator can construct the Linear4bit model and inject states.
    torch.save(model.state_dict(), os.path.join(chkpt_dir, "ptq_int4.pt"))
    print("Saved PTQ-INT4 model representation.")

if __name__ == "__main__":
    main()
