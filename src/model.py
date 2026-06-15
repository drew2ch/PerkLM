""" Decoder-only, GPT-style Transformer Implementation.

    Author: Andrew Chung
"""

import re
import math
import torch
import torch.nn.functional as F
from torch import nn
from dataset import FriendsDataset
from transformers import GPT2TokenizerFast

def precompute_rope_freqs(d_head: int, seq_len: int, base: float = 10000.0, device = None):
    """ Compute complex frequency tensor for RoPE
        Returns:
            freqs_cis: (seq_len, d_head // 2);
            freqs_cis[m, i] = exp(i*m*theta_i)
    """
    # theta_i = base^{-2i/d}, shape (head_dim // 2,)
    i = torch.arange(0, d_head, 2, dtype = torch.float32, device = device)
    theta = 1.0 / (base ** (i / d_head))

    # outer product: m * theta_i -> angle m&theta_i
    positions = torch.arange(seq_len, dtype = torch.float32, device = device)
    angles = torch.outer(positions, theta)

    # exp(i * angle) = cos(angle) + i*sin(angle)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """ Apply RoPE to Q/K tensors.
        Args:
            x: (batch, seq_len, n_heads, d_head)
            freqs_cis: (seq_len, d_head // 2)
        Returns:
            x_rot: (batch, seq_len, n_heads, d_head)
    """
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)

    # Broadcast freqs_cis over batch and heads
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Elementwise rotation (complex mult)
    x_rot = x_complex * freqs

    # Back to real
    x_out = torch.view_as_real(x_rot) # (B, T, H, d/2, 2)
    x_out = x_out.reshape(*x.shape)   # (B, T, H, d)
    
    return x_out.type_as(x)
    

class DialogueEmbedding(nn.Module):
    """ Embedding Layer for Friends Dialogue Corpus
        Note: I implemented RoPE in lieu of sinusoidal position embeddings
        in the Attention Head.
    """
    def __init__(self, tokenizer, d_model, maxt: int = 512):
        
        super().__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.maxt = maxt
        self.embedding = nn.Embedding(num_embeddings = len(self.tokenizer),
                                      embedding_dim = self.d_model)
        
        # new: Responder Embedding
        self.responder_embedding = nn.Embedding(num_embeddings = len(FriendsDataset.SPEAKER_LOOKUP), 
                                                embedding_dim = self.d_model)

    def forward(self, batch):
        """ Params
                batch: DataLoader batch
        """

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        assert input_ids.shape == attention_mask.shape, \
            f'Error: input_ids {input_ids.shape} and attention_mask {attention_mask.shape} must have the same shape'
        embeddings = self.embedding(input_ids)

        # new: Responder Embedding
        r = self.responder_embedding(batch['responder'])
        x = embeddings + r.unsqueeze(1)

        return x, attention_mask
    
class DialogueMultiHeadAttention(nn.Module):
    """ Transformer Decoder Block w/ RoPE
    """
    def __init__(self, d_model: int, n_heads: int, base: float = 10000.0, dropout: float = 0.2, maxt: int = 512):
        
        super().__init__()
        assert d_model % n_heads == 0, \
            f'Error: d_model {d_model} must be divisible by n_heads {n_heads}'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.dropout = dropout
        self.maxt = maxt

        # vectorized Q, K, V
        self.w_q = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias = False)
        self.scale = 1 / math.sqrt(self.d_head)

        freqs_cis = precompute_rope_freqs(self.d_head, self.maxt, base = base)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(self, embedding, attention_mask):
        
        assert embedding.shape[-1] == self.d_model, \
            f'Error: embedding dimension {embedding.shape[-1]} and d_model {self.d_model} must match'
        
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        MASK = (1 - attention_mask.float()) * -1e9
        B, T = embedding.shape[:2]
        
        Q = self.w_q(embedding).view(B, T, self.n_heads, self.d_head)
        K = self.w_k(embedding).view(B, T, self.n_heads, self.d_head)
        V = self.w_v(embedding).view(B, T, self.n_heads, self.d_head)
        
        freqs = self.freqs_cis[:T]
        Q_rope = apply_rope(Q, freqs).transpose(1, 2)
        K_rope = apply_rope(K, freqs).transpose(1, 2)
        V = V.transpose(1, 2)
        A = (Q_rope @ K_rope.transpose(-2, -1)) * self.scale

        # Masked Multi-Head Attention
        T_mask = A.size(-1)
        mask = torch.triu(torch.ones(T_mask, T_mask, device = embedding.device), diagonal = 1).bool()
        A = A.masked_fill(mask, float('-inf')) + MASK

        A = F.softmax(A, dim = -1)
        A = F.dropout(A, p = self.dropout, training = self.training)
        attention = A @ V

        attn_out = attention.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(attn_out)
    
class DialogueDecoderLayer(nn.Module):
    """ Transformer Decoder Block for Dialogue Embeddings
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.2, maxt: int = 512):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.maxt = maxt

        self.attn = DialogueMultiHeadAttention(self.d_model,
                                               self.n_heads,
                                               dropout = self.dropout,
                                               maxt = self.maxt)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ff1 = nn.Linear(self.d_model, self.d_ff)
        self.ff2 = nn.Linear(self.d_ff, self.d_model)
        
    def forward(self, embedding, attention_mask):
        """ Pre-LN (GPT) Add/Norm and FFN
        """

        attn_out = self.attn(self.norm1(embedding), attention_mask)
        x = embedding + F.dropout(attn_out, p = self.dropout, training = self.training)
        ff_out = self.ff2(F.dropout(F.gelu(
            self.ff1(self.norm2(x))), p = self.dropout, training = self.training))
        x = x + F.dropout(ff_out, p = self.dropout, training = self.training)

        return x
    
class FriendsTransformer(nn.Module):
    """ Master Class encompassing the Embedding Layer plus
        a stack of DialogueDecoderLayers.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, 
                 dropout: float = 0.2, maxt: int = 512, tokenizer = None):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.maxt = maxt
        self.tokenizer = tokenizer

        self.embedder = DialogueEmbedding(self.tokenizer, self.d_model, self.maxt)
        self.decoder = nn.ModuleList([DialogueDecoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout, self.maxt)\
                                      for _ in range(self.n_layers)])
        self.final_norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, len(self.tokenizer), bias = False)
        self.lm_head.weight = self.embedder.embedding.weight

    def forward(self, batch):
        """ Params
            batch: DataLoader batch
        """

        embedding, attention_mask = self.embedder(batch)
        x = embedding
        for layer in self.decoder:
            x = layer(x, attention_mask)
        logits = self.lm_head(self.final_norm(x))

        return logits
    
    @classmethod
    def load(cls, path: str, device = None):
        """ Load from pre-trained model checkpoint
            Args:
                path (str): Path to the checkpoint file.
                device: The device to load the model onto. If None, uses the current device.
            Returns:
                An instance of FriendsTransformer with loaded weights.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location = device, weights_only = True)
        config = checkpoint['config']
        tokenizer_path = config['model'].pop('tokenizer')
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        model = cls(**config['model'], tokenizer = tokenizer).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    @torch.no_grad()
    def generate(self, prompt, responder: int, tokenizer = None, temperature: float = 1.0, device = None,
                 min_length: int = 0, max_length: int = 256, penalty: float = 1.0, random_state = None):
        """ Generator function
            Args:
                prompt (str): The input text to generate a response for, pre-encoded.
                responder (int): The ID of the responder to generate a response for (0-12).
                tokenizer: The tokenizer to use for encoding the input.
                temperature (float): The temperature for softmax sampling.
                min_length (int): The minimum length of the generated output.
                max_length (int): The maximum length of the generated output.
                random_state (int): The random seed for reproducibility.
                penalty (float): Repetition penalty strength. Default 1.0 (no penalty).
        """

        def get_speaker(responder: int):
            """ Helper to retrieve speaker name from responder ID
                FriendsDataset.SPEAKER_LOOKUP is a list of speaker names with 
                unique mapping indices.
            """
            return next((s for s, i in FriendsDataset.SPEAKER_LOOKUP.items() \
                        if i == responder), "OTHER")
        
        def penalize(logits: torch.Tensor, gen_ids: torch.Tensor, 
                     alpha: float = 1.0) -> torch.Tensor:
            """ Apply repetition penalty to logits of frequent tokens
            """
            counter = torch.bincount(gen_ids, minlength = logits.shape[-1]).float()
            return logits - alpha * torch.log1p(counter)
        
        def clean(text: str) -> str:
            """ Post-process generated text
                - Remove <EOT> token
                - Replace multiple newlines with a single newline
                - Enforce common grammatical/syntactical rules
            """

            # 1. trim consecutive spaces, eliminate <EOT>
            text = re.sub(' +', ' ', text.replace('\n', ' ')).replace('<EOT>', '').strip()
            
            # 2. Enforce capitalization at beginning and sentence boundaries
            text = text[0].upper() + text[1:] if text else text
            text = re.sub(r'(?<=[.!?])\s+([a-z])', 
                  lambda m: m.group(0).upper(), text)
            
            # 3. Capitalize standalone 'i'
            text = re.sub(r'\bi\b', 'I', text)

            # 4. Space before and after punctuation/apostrophe, redundant punctuation repeats
            text = re.sub(r'([,;:])\1+', r'\1', text)
            text = re.sub(r'([.!?])\1{3,}', r'\1\1\1', text)
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
            text = re.sub(r"'\s+([a-z])", r"'\1", text)
            
            return text

        # verify the responder ID exists
        assert responder in range(len(FriendsDataset.SPEAKER_LOOKUP)), \
            f'Error: Invalid responder ID {responder} not in range({len(FriendsDataset.SPEAKER_LOOKUP)})'

        # Tokenizer/Device initialization, Fixed output seeding
        if tokenizer is None: tokenizer = self.tokenizer
        if device is None: device = next(self.parameters()).device
        if random_state is not None: torch.manual_seed(random_state)
        EOT_ID = tokenizer.convert_tokens_to_ids('<EOT>')
        
        # process input prompt
        prompt += f"\n<RESPONSE>\n<SPEAKER={get_speaker(responder)}>"
        encoded = tokenizer(prompt, add_special_tokens = False, return_tensors = 'pt')
        prompt_ids = encoded['input_ids'].to(device)
        responder_tensor = torch.tensor([responder], device = device)

        # enumerate forbidden tokens (speaker tokens, context/response indicators)
        BANNED_TOKENS = [i for i in tokenizer.all_special_ids if i != EOT_ID] + [9860] # 'yer'

        # generate output
        gen_ids = []
        for _ in range(max_length):
            
            gen_tensor = torch.tensor(gen_ids, device = device, dtype = torch.long)
            current_ids = torch.cat([prompt_ids, gen_tensor.unsqueeze(0)], dim = -1)

            batch = {'input_ids': current_ids,
                           'attention_mask': torch.ones_like(current_ids, device = device),
                           'responder': responder_tensor}
            logits = self(batch)

            # softmax over raw logits then sample next token
            # temperature controls concentration of softmax probabilities
            next_token_logits = logits[0, -1, :] / temperature
            next_token_logits = penalize(next_token_logits, gen_tensor, alpha = penalty)
            next_token_logits[BANNED_TOKENS] = float('-inf')

            if len(gen_ids) < min_length: next_token_logits[EOT_ID] = float('-inf')
            probs = torch.softmax(next_token_logits, dim = -1)
            next_token = torch.multinomial(probs, num_samples = 1).item()
            gen_ids.append(next_token)

            # maximum length or <EOT> reached
            if len(gen_ids) >= max_length: 
                # gen_ids.append(EOT_ID)
                break
            if next_token == EOT_ID: break

        # decode into text, post-processing
        gen_ids = torch.tensor(gen_ids, device = device)
        gen_seq = tokenizer.decode(gen_ids, skip_special_tokens = False)

        return clean(gen_seq)
