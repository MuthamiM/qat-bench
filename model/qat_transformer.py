import torch
import torch.nn as nn
from model.transformer import TransformerLM, Block

class QATBlock(nn.Module):
    def __init__(self, config, block):
        super().__init__()
        self.ln_1 = block.ln_1
        self.attn = block.attn
        self.ln_2 = block.ln_2
        self.mlp = block.mlp
        
        self.dequant_1 = torch.quantization.DeQuantStub()
        self.quant_1 = torch.quantization.QuantStub()
        self.dequant_2 = torch.quantization.DeQuantStub()
        self.quant_2 = torch.quantization.QuantStub()
        self.f_add_1 = torch.nn.quantized.FloatFunctional()
        self.f_add_2 = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        # x is quantized
        float_x = self.dequant_1(x)
        ln_1_out = self.ln_1(float_x)
        
        # attn must run in float because quantized bmm is not supported by PyTorch CPU
        attn_out = self.attn(ln_1_out)
        quant_attn_out = self.quant_1(attn_out)
        
        # add
        x = self.f_add_1.add(x, quant_attn_out)
        
        # ln_2 needs float
        float_x2 = self.dequant_2(x)
        ln_2_out = self.ln_2(float_x2)
        quant_ln_2 = self.quant_2(ln_2_out)
        
        # mlp takes quantized
        mlp_out = self.mlp(quant_ln_2)
        
        # add
        x = self.f_add_2.add(x, mlp_out)
        return x

class QATTransformerLM(TransformerLM):
    def __init__(self, config, vocab_size):
        super().__init__(config, vocab_size)
        # Stubs for QAT
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.dequant_ln_f = torch.quantization.DeQuantStub()
        self.quant_ln_f = torch.quantization.QuantStub()
        
        # Replace blocks with QATBlocks
        qat_blocks = []
        for b in self.blocks:
            b.attn.qconfig = None  # Don't quantize attn, keep it in float
            qat_blocks.append(QATBlock(config, b))
        self.blocks = nn.ModuleList(qat_blocks)
        
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

        # ln_f needs float
        x_float = self.dequant_ln_f(x)
        x_float = self.ln_f(x_float)
        x = self.quant_ln_f(x_float)
        
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
