import os
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import json
from tqdm import tqdm

from model.transformer import TransformerLM

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, dataloader, optimizer, scheduler, config, device):
    model.train()
    total_loss = 0
    grad_clip = config['training']['grad_clip']
    
    pbar = tqdm(dataloader, desc="Training Batch")
    for batch in pbar:
        x = batch[0].to(device)
        y = batch[1].to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        x = batch[0].to(device)
        y = batch[1].to(device)
        logits, loss = model(x, y)
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    avg_loss = total_loss / len(dataloader)
    return avg_loss, math.exp(avg_loss) if avg_loss < 20 else float('inf')

def create_dataloaders(config):
    data_dir = os.path.join(os.path.dirname(__file__), '..', config['output']['results_dir'], 'data')
    train_data = torch.load(os.path.join(data_dir, f"{config['data']['train_split']}.pt"), weights_only=True)
    val_data = torch.load(os.path.join(data_dir, f"{config['data']['val_split']}.pt"), weights_only=True)
    
    seq_len = config['data']['seq_len']
    bs = config['training']['batch_size']
    
    def to_dataset(data):
        n = len(data) - 1
        n_chunks = n // seq_len
        data = data[:n_chunks * seq_len + 1]
        
        x_list = []
        y_list = []
        for i in range(0, len(data) - seq_len, seq_len):
            x_list.append(data[i:i+seq_len])
            y_list.append(data[i+1:i+1+seq_len])
        
        x_tensor = torch.stack(x_list)
        y_tensor = torch.stack(y_list)
        return TensorDataset(x_tensor, y_tensor)

    train_ds = to_dataset(train_data)
    val_ds = to_dataset(val_data)
    
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    with open(os.path.join(data_dir, 'tokenizer.json'), 'r') as f:
        meta = json.load(f)
        vocab_size = meta['vocab_size']
        
    return train_dl, val_dl, vocab_size

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dl, val_dl, vocab_size = create_dataloaders(config)
    
    model = TransformerLM(config['model'], vocab_size).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    epochs = config['training']['epochs']
    total_steps = epochs * len(train_dl)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config['training']['warmup_steps'], total_steps)
    
    best_val_ppl = float('inf')
    chkpt_dir = os.path.join(os.path.dirname(__file__), '..', config['output']['checkpoint_dir'])
    os.makedirs(chkpt_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, config, device)
        val_loss, val_ppl = evaluate(model, val_dl, device)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.4f}")
        
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(chkpt_dir, "fp32_best.pt"))
            print("Saved new best model FP32")

if __name__ == "__main__":
    main()
