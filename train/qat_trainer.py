import os
import yaml
import torch
import torch.nn as nn
from train.trainer import create_dataloaders, get_cosine_schedule_with_warmup, train_epoch, evaluate
from model.qat_transformer import QATTransformerLM
from torch.optim import AdamW

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for QAT training")
    
    train_dl, val_dl, vocab_size = create_dataloaders(config)
    
    model = QATTransformerLM(config['model'], vocab_size)
    model.to("cpu") # Must prepare QAT on CPU
    
    model.train()
    print("Fusing modules for QAT...")
    model.fuse_model()
    
    print("Preparing QAT stubs and observers...")
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Ignore embeddings and layer norm since fbgemm doesn't always support them linearly
    model.wte.qconfig = None
    model.wpe.qconfig = None
    model.ln_f.qconfig = None
    for block in model.blocks:
        block.ln_1.qconfig = None
        block.ln_2.qconfig = None
        
    torch.quantization.prepare_qat(model, inplace=True)
    
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    epochs = 1
    total_steps = epochs * len(train_dl)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config['training']['warmup_steps'], total_steps)
    
    best_val_ppl = float('inf')
    chkpt_dir = os.path.join(os.path.dirname(__file__), '..', config['output']['checkpoint_dir'])
    os.makedirs(chkpt_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        if epoch == 4:
            print("Disabling observers (Epoch 4)...")
            model.apply(torch.quantization.disable_observer)
            
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, config, device)
        val_loss, val_ppl = evaluate(model, val_dl, device)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.4f}")
        
    print("Training complete. Converting to QAT INT8 model...")
    model.eval()
    model.to("cpu")
    torch.quantization.convert(model, inplace=True)
    
    # Save the explicitly evaluated and converted int8 model
    save_path = os.path.join(chkpt_dir, "qat_int8.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved QAT-INT8 quantized model to {save_path}")

if __name__ == "__main__":
    main()
