import torch
import torch.nn as nn
from model.transformer import TransformerLM

class QATTransformerLM(TransformerLM):
    def __init__(self, config, vocab_size):
        super().__init__(config, vocab_size)
        # Stubs for QAT
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Quantize activations
        x = self.quant(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Dequantize back to float
        logits = self.dequant(logits)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
        
    def fuse_model(self):
        """
        Fuses specific modules for better integer execution.
        We fuse Linear and ReLU in the MLP logic.
        """
        for block in self.blocks:
            torch.quantization.fuse_modules(block.mlp, [['c_fc', 'act']], inplace=True)
