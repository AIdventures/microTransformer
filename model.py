"""
Transformer implementation in PyTorch.
References:
1) The official paper: https://arxiv.org/pdf/1706.03762.pdf
2) nanoGPT, by Karpathy: https://github.com/karpathy/nanoGPT
3) My personal blog post: https://aidventure.es/blog/transformer
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import VOCABULARY, SENTENCE_LEN
from utils import tokenize_sentence


class MultiHeadAttention(nn.Module):

    def __init__(self, config, use_mask=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.use_mask = use_mask
        # key, query, value projections for all heads
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # register buffer in Pytorch -> If you have parameters in your model,
        # which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.block_size, config.block_size
                )
            ).view(1, 1, config.block_size, config.block_size)
        )
       
    def forward(self, q, k, v):
        B, T, E = k.size() # batch size, sequence length, embedding dimensionality (n_embd)
        T_Q = q.size(1) # length of query sequence

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q  = self.q_attn(q)
        k  = self.k_attn(k)
        v  = self.v_attn(v)
        q = q.view(B, T_Q, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.use_mask:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T_Q, E) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config, use_mask=False)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        out = self.ln_1(x)
        x = x + self.attn(out, out, out)
        
        out = self.ln_2(x)
        x = x + self.mlp(out)
        return x
    

class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.masked_attn = MultiHeadAttention(config, use_mask=True)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.cross_attn = MultiHeadAttention(config, use_mask=False)
        self.ln_3 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, enc_out):
        out = self.ln_1(x)
        x = x + self.masked_attn(out, out, out)
        
        out = self.ln_2(x)
        x = x + self.cross_attn(out, enc_out, enc_out)
        
        out = self.ln_3(x)
        x = x + self.mlp(out)
        return x


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            transformer_blocks = nn.ModuleList([
                EncoderBlock(config) for _ in range(config.n_encoder_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            transformer_blocks = nn.ModuleList([
                DecoderBlock(config) for _ in range(config.n_decoder_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
      
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.encoder.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_encoder_layer))

        for pn, p in self.decoder.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_decoder_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encoder_forward(self, src):
        device = src.device
        tok_emb = self.encoder.wte(src) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.encoder.wpe(torch.arange(0, src.size(1), dtype=torch.long, device=device)) # positional embeddings (t, n_embd)
        x = self.encoder.drop(tok_emb + pos_emb)
        for block in self.encoder.transformer_blocks:
            x = block(x)
        enc_out = self.encoder.ln_f(x)
        return enc_out
    
    def decoder_forward(self, enc_out, tgt):
        device = tgt.device
        tok_emb = self.decoder.wte(tgt)
        pos_emb = self.decoder.wpe(torch.arange(0, tgt.size(1), dtype=torch.long, device=device)) # positional embeddings (t, n_embd)
        x = self.decoder.drop(tok_emb + pos_emb)
        for block in self.decoder.transformer_blocks:
            x = block(x, enc_out)
        dec_out = self.decoder.ln_f(x)

        logits = self.lm_head(dec_out)
        return logits

    def forward(self, src, target):

        ## ENCODER
        # forward the Encoder model
        enc_out = self.encoder_forward(src)

        ## DECODER
        # forward the Decoder model
        logits = self.decoder_forward(enc_out, target)

        return logits
    
    def generate(self, src, device, temperature=1.0, top_k=None):
        """
        Generate a sequence of tokens given a source sequence
        """
        self.eval()
        with torch.no_grad():
            # src is a list of words from the vocabulary. 
            # Convert it to tokens and then to a batched tensor
            src = torch.tensor(tokenize_sentence(src, VOCABULARY)).unsqueeze(0).to(device)
            encoder_out = self.encoder_forward(src)
            decoder_out = torch.tensor([[VOCABULARY["<s>"]]]).to(device)
            for _ in range(SENTENCE_LEN -1):
                # forward the model to get the logits for the index in the sequence
                logits = self.decoder_forward(encoder_out, decoder_out)
                # pluck the logits at the final step and scale by desired temperature
                # logits shape (batch, block_size, vocab_size)
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                decoder_out = torch.cat((decoder_out, next_token), dim=1)
        return decoder_out
    

def get_accuracy(model, dataloader, device, max_batches=None):
    """
    Compute the accuracy of the model on the given CharactersDataset

    Args:
        model: the model to evaluate
        dataloader: the DataLoader object to use
        device: the device to run the model on
        max_batches: the maximum number of batches to evaluate on

    Returns:
        The accuracy of the model on the given dataset
    """
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_shifted = batch["tgt_shifted"].to(device)
            logits = model(src, tgt_shifted)
            predictions = torch.argmax(logits, dim=-1)
            # A sequence is correct if all the tokens are correct
            # shape (batch_size, sequence_length)
            correct += torch.sum(
                torch.all(
                    torch.eq(predictions, tgt),
                    dim=1
                )
            ).item()
            total += len(src)
            if max_batches is not None and batch_index >= max_batches:
                break
    model.train()
    return correct / total
