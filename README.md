# PerkLM -- The One With A.I.

**PerkLM**: Friends Dialogue Micro-Language Model is a **bespoke, decoder-only autoregressive transformer** that I constructed from first principles. My language model, trained on dialogue lines from all 10 seasons of the ageless American sitcom **Friends** (1994-2004), generates a character-conditioned line of dialogue in response to user-provided dialogue.

**Motivation**: For a generative language modeling task, I could have just as easily fine-tuned an open source LLM from Hugging Face, but my desire was to learn the ins-and-outs of engineering the transformer architecture that underlies essentially all modern communicative AI models and deepen my mathematical intuition of the self-attention mechanism that underlies them. To that end, I constructed the entire decoder-only Transformer stack from scratch -- no pre-packaged `nn.Module` objects, pre-trained weights, nor copy-and-pasted tutorial code, just PyTorch primitives. The fascinating truth that I learned -- under the bonnet of the most powerful language models as of this writing is math almost deceptively simple, encompassing matrix inner products, softmax normalization, affine transformations, and causal masking.

## Quickstart Guide and Usage Instructions

The full codebase written to scrape and process dialogue data, define the transformer modules and tokenizer, and implement training loop is open-sourced on this repository. The actual model weights (`.pt` and `.safetensors`) can be accessed on [Hugging Face](https://huggingface.co/drew2ch/perklm). Requirements depend on whether your end goal is inference with the frozen model or full reproduction of my pipeline; see `requirements.txt` for the full stack.

```{bash}
# Full (training + inference)
pip install -r requirements.txt
# Inference only
pip install torch transformers
```

Below are the 2 methods to import and set up PerkLM on your local machine. 

### Option A: Load from Hugging Face (recommended)

Weights and config are hosted at `drew2ch/perklm`. The model uses a custom architecture class, so `trust_remote_code = True` is required.

```{python}
from transformers import AutoModelForCausalLM, GPT2TokenizerFast

model = AutoModelForCausalLM.from_pretrained('drew2ch/perklm', trust_remote_code = True)
tokenizer = GPT2TokenizerFast.from_pretrained('drew2ch/perklm', subfolder = 'tokenizer')
model.eval()
```

### Option B: Load from Local Checkpoint

If you elect to use the weights `perklm.pt` (downloaded from HF or obtained from your own training run), the model can be instantiated with its built-in `load()` module, analogous to the above `load_pretrained()` modules.

```{python}
from model import FriendsTransformer

model = FriendsTransformer.load('perklm.pt')
model.eval()
```

### Dialogue Generation (Inference)

Now here's the fun part: **line generation**. As with model loading, the built-in `generate()` function neatly bundles together the full generation loop. The user can specify the **context window**, the **responder ID** (conditions the output on target character's voice), and various numerical parameters concerning response quality and constraints. Refer to the following lookup table for the integer ID-character mapping:

| ID | Character | ID | Character |
|:--|:--|:--|:--|
| 0 | Ross     | 6 | Gunther |
| 1 | Monica   | 7  | Janice |
| 2 | Chandler | 8  | Richard |
| 3 | Joey     | 9  | Carol |
| 4 | Rachel   | 10 | Susan |
| 5 | Phoebe   | 11 | Mike |
| | | 12 | Other |

Below is example code to format your context window and prompt PerkLM accordingly, along with the parameter catalogue:

```{python}
context = ("<CONTEXT>\n"
            "<SPEAKER=ROSS> We were on a break!\n"
            "<SPEAKER=RACHEL> That is not the point.\n"
            "</CONTEXT>")

# For option A, call model.transformer.generate()
# For option B, call model.generate()

# You can refer to the responder ID table above or 
# alternatively query the built-in responder ID map.

responder = 2 # Chandler
responder = FriendsDataset.SPEAKER_LOOKUP["CHANDLER"] # All Caps, will return 2

response = model.generate(prompt = context,
                        responder = responder,
                        temperature = 0.85,
                        penalty = 1.2,
                        min_length = 10,
                        max_length = 128,
                        random_state = 893)

response
# ex: Could this BE any more of a third-wheel situation?
```

|Parameter|Type|Default|Description|
|:--|:--|:--|:--|
|`prompt`|`str`|--|Formatted context string; **required**|
|`responder`|`int`|--|Target Character ID (0-12); **required**|
|`temperature`|`float`|`1.0`|Softmax Temperature; lower = more conservative, higher = more chaotic/random|
|`penalty`|`float`|`1.0`|Repetition penalty strength; values above `1.0` suppress repeated tokens|
|`min_length`|`int`|`0`|Minimum token length before `<EOT>` allowed|
|`max_length`|`int`|`256`|Ceiling on generated token count|
|`random_state`|`int`|`None`|Seed for reproducible sampling; random if `None`|

## Architecture and Model Constituents

PerkLM is a **44.6M-parameter** Decoder-Only, GPT-style Transformer, with every component implemented by hand. Some design choices include:

- GPT-2 Tokenizer (`GPT2TokenizerFast`) for Dialogue Tokenization
- Rotary Position Encoding (RoPE) Embeddings
- Pre-LN (Layer Normalization) as Default Mode for GPT-style Models
- Additive Learned Embedding for Speaker (Responder) Conditioning

### Embedding Layer -- `DialogueEmbedding`

The pre-attention embedding step, where each input token is embedded through a learned token embedding. Instead of additive sinusoidal position embeddings (see RoPE below), we instead introduce an additive learned embedding conditional on the **responder's character** (e.g. Rachel responding to Ross's "We were on a break!" vs, say, Phoebe). This has the effect of shifting the representation space toward the target speaker's voice before the first attention layer.

### Multi-Head Self-Attention (MHSA) -- `DialogueMultiHeadAttention`

This module encodes the **causal masked multi-head attention (MHA)**. For the decoder, an input token is only permitted to attend to the tokens preceding it, producing a lower triangular causal attention mask. In lieu of sinusoidal or learned absolute position encodings at the embedding step, my model implements **Rotary Positional Embeddings (RoPE)**. The Query ($Q$) and Key ($K$) matrices are essentially rotated element-wise in the complex plane before the inner product ($QK^\top$) computation, which has the effect of encoding **relative positional relationships** directly into the attention scores. Each of the `n_heads` heads learn different representations of the model before concatenation and a linear layer.

### Decoder Block -- `DialogueDecoderLayer`

The decoder block incorporates the multi-head attention (MHA) step above with **pre-LN** (GPT-style) residual blocks: performing layer normalization before the MHA and feed-forward network to improve gradient flow during training. Each block contains:
- Pre-LN masked MHSA and position-wise FFN (GELU activation)
- Residual connections with dropout (0.1) at both sublayers

### Master Model -- `FriendsTransformer`

Aptly named, the master model encompasses the embedding step, multiple stacked  `DialogueDecoderLayer` layers, and a final output head -- a linear projection from `d_model` to $\lvert V\rvert$ (vocab size) with **weight tying** to the token embedding matrix, reducing parameter count and generally improving generalization. The module returns raw logits that can be converted into softmax probabilities at varying temperatures.

### Model Configuration

|Hyperparameter|Value|Notes|
|:--|:--|:--|
|`d_model`|512|Model Dimension|
|`n_heads`|8|Attention Heads|
|`n_layers`|6|Decoder Layers|
|`d_ff`|2048|FFN Dimension|
|`maxt`|512|Max Token Length|
|`dropout`|0.1|Dropout Rate|
|**Parameters**|**~44.6M**|

## Data Pipeline

### Scraping -- `scrape.py`

The full transcripts for all 10 seasons of Friends were obtained from the [Friends Scripts Database](https://edersoncorbari.github.io/friends/). Though the transcript quality itself was pristine, inconsistent HTML formatting across individual episodes and within seasons made the scraping process a lot more tedious that it could have been. The biggest offenders were structured dialogue in `<p>` blocks, `<br>`-delimited text nodes, and interspersed `<b>` tags mid-line. My `scrape.py` code implements episode-specific heuristics and a node-level DOM traversal strategy to handle this with as much grace as possible.

### Preprocessing -- `preprocess.py`

Freshly acquired raw transcripts undergo a multi-stage cleaning pipeline, as follows:
1. **Speaker Normalization**: using a hand-curated alias dictionary to resolve typos, alternate spellings, nicknames, and alter egos (e.g. `Fat Monica` = `Monica`, `Rach` = `Rachel`, `Dr. Drake Remoray` = `Dr. Drake Ramoray`)
2. **Stage Direction Ablation**: regex is applied to strip away inline parenthetical and bracketed stage directions
3. **Multi-Speaker Drop**: lines belonging to ambiguous or multi-person attributes (e.g. `All`, `Both`, `Everyone`, `Monica and Chandler`) and unnamed generic roles (waiter, nurse, cop, etc.) were dropped entirely; only named characters are retained
4. **Dangling Stage Direction Merge**: stage entries continuing a disrupted dialogue line are concatenated to the previous turn before stage ablation; additionally, dialogue lines misformatted as scene descriptions are modified accordingly

To prevent temporal data leakage, I split the cleaned corpus by season: **Seasons 1-8 = Train, Season 9 = Validation, Season 10 = Test**.

### Dataset and Tokenizer -- `dataset.py`, `tokenizer.py`

I loaded a pre-trained GPT-2 tokenizer, `GPT2TokenizerFast` and added special tokens encoding the context window, speaker ID, response, and end of turn (EOT). The 6 main Friends and 6 additional hand-selected characters (Janice, Gunther, Richard, Carol, Susan, Mike) received their own speaker identities (by dialogue count and plot significance), and all other speakers were categorized as "other," altogether contributing 17 new tokens. Tokenization is performed on the fly.

`FriendsDataset` constructs **sliding context windows** of 1-3 prior turns per response target, yielding approximately three training examples per response. Each training example is formatted as:

```
# 2-turn context example (hypothetical)
<CONTEXT>
<SPEAKER=ROSS> But we were on a break!
<SPEAKER=RACHEL> That is not the point.
</CONTEXT>
<RESPONSE>
<SPEAKER=CHANDLER> Could this BE any more of a third-wheel situation?
<EOT>
```

## Training Scheme

As aforementioned, the model was initially trained on dialogue turns from seasons 1-8, validated on season 9, and tested on season 10. Considering the generative nature of the model, I tracked the perplexity score in addition to the training and validation loss. On training documentation:
- **Optimization**: AdamW with GPT-3 paper betas ($\beta_1=0.9, \beta_2=0.95$) and weight decay $10^{-2}$.
- **Scheduler**: Linear warmup (500 steps) followed by cosine annealing decay to a minimum learning rate of $10^{-5}$.
- Mixed-precision training via `torch.autocast` (FP16) with gradient scaling
- Gradient accumulation over 32 steps (batch size 4; effective batch size 128), clipping at `max_norm = 1.0`
- Early stopping on validation loss (with patience 10)
- TensorBoard logging for loss, perplexity, gradient norm, and LR

After the initial training and test inference, I trained a deployment model on all 10 seasons on a fixed number of 12 epochs (no early stopping), observed as the optimum under training. 

```{bash}
# Experimental training
python train.py --config config.yaml --tensorboard
# Deployment training
python train.py --config config.yaml --deploy
```

Inference is handled by the `FriendsTransformer`'s built-in functions `load()` and `generate()` (see above).

## Honest Word

LLMs' success can be largely attributed to the astronomical scale of their training corpus; transformers thrive on large volumes of data spanning a vast range of domains. PerkLM, on the contrary, is a micro-language model trained on a small (~50K lines), single-domain corpus. I've observed that generated outputs are grammatically coherent but can often make no semantic sense; the model captures shallower dialogue cadence but may face a wall in learning character voices or content consistency. 

Despite limitations on data quantity and model scale, all model internals -- attention mechanism, RoPE, causal masking, repsonder conditioning, neural network components -- were written from mathematical first principles and work flawlessly, which was my aim for this undertaking.

## Repository Structure

```
.
├── src/
│   ├── model.py          # FriendsTransformer, DialogueMultiHeadAttention, RoPE
│   ├── dataset.py        # FriendsDataset, context window construction
│   └── tokenizer.py      # GPT-2 tokenizer extension with speaker special tokens
├── pipeline/
│   ├── scrape.py         # HTML transcript scraper
│   └── preprocess.py     # Speaker normalization, stage direction ablation
├── tokenizer/            # Saved GPT-2 tokenizer (generated by tokenizer.py)
│   ├── added_tokens.json
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── train.py              # Training loop, optimizer, scheduler, early stopping
├── config.yaml           # Hyperparameter configuration
├── requirements.txt
└── .gitignore
```

**Author**: **Andrew (Ho-Young) Chung** · Cornell BS/MPS Statistics, Information Science
