import os
import torch
import yaml
import json
from datasets import load_dataset

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print(f"Loading dataset {config['data']['dataset']} ({config['data']['subset']})...")
    ds = load_dataset(config['data']['dataset'], config['data']['subset'])
    
    # Extract text from the training split to build the vocabulary
    train_text = " ".join(ds[config['data']['train_split']]['text'])
    
    # Build character-level vocabulary
    chars = sorted(list(set(train_text)))
    vocab_size = len(chars) + 1  # +1 for <UNK> token
    stoi = {ch: i for i, ch in enumerate(chars)}
    stoi['<UNK>'] = vocab_size - 1
    itos = {i: ch for ch, i in stoi.items()}
    
    # Tokenizer function using a simple generator or list comprehension
    def encode(text):
        return [stoi.get(c, stoi['<UNK>']) for c in text]
    
    # Prepare output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', config['output']['results_dir'], 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all splits
    splits = [
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split']
    ]
    
    for split in splits:
        print(f"Tokenizing {split} split...")
        split_text = " ".join(ds[split]['text'])
        data_ids = encode(split_text)
        data_tensor = torch.tensor(data_ids, dtype=torch.long)
        
        # Save flattened 1D tensor
        save_path = os.path.join(output_dir, f"{split}.pt")
        torch.save(data_tensor, save_path)
        print(f"Saved {split}.pt with {len(data_tensor)} tokens.")
    
    # Save tokenizer metadata
    tokenizer_meta = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos
    }
    with open(os.path.join(output_dir, 'tokenizer.json'), 'w') as f:
        json.dump(tokenizer_meta, f)
        
    print(f"Data preparation complete. Vocab size: {vocab_size} characters.")

if __name__ == '__main__':
    main()
