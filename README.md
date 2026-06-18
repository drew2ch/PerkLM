# PerkLM -- Friends Dialogue Micro-Language Model

**PerkLM** is a **bespoke, decoder-only autoregressive transformer** that I constructed from first principles. My language model, trained on dialogue lines from all 10 seasons of the ageless American sitcom **Friends** (1994-2004), generates a character-conditioned line of dialogue in response to user-provided dialogue.

**Motivation**: For a generative language modeling task, I could have just as easily pre-trained an open source LLM from Hugging Face, but my desire was to learn the ins-and-outs of engineering the transformer architecture that underlies essentially all modern communicative AI models and deepen my mathematical intuition of the self-attention mechanism that underlies them. To that end, I constructed the entire decoder-only Transformer stack from scratch -- no pre-packaged `nn.Module` objects, pre-trained weights, nor copy-and-pasted tutorial code, just PyTorch primitives. The fascinating truth that I learned -- under the bonnet of the most powerful language models as of written is math almost deceptively simple, encompassing matrix inner products, softmax normalization, affine transformations, and causal masking.

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

The full transcripts for all 10 seasons of Friends were obtained from the [Friends Scripts Database](https://edersoncorbari.github.io/friends/). Though the transcript quality itself was pristine, inconsistent HTML formatting across individual episodes and within seasons m made the scraping process a lot more tedious that it could have been. The biggest offenders were structured dialogue in `<p>` blocks, `<br>`-delimited text nodes, and interspersed `<b>` tags mid-line. My `scraper.py` code implements episode-specific heuristics and a node-level DOM traversal strategy to handle this with as much grace as possible.

### Preprocessing -- `preprocess.py`

Freshly acquired raw transcripts undergo a multi-stage cleaning pipeline, as follows:
1. **Speaker Normalization**: using a hand-curated alias dictionary to resolve typos, alternate spellings, nicknames, and alter egos (e.g. `Fat Monica` = `Monica`, `Rach` = `Rachel`, `Dr. Drake Remoray` = `Dr. Drake Ramoray`)
2. **Stage Direction Ablation**: regex is applied to strip away inline parenthetical and bracketed stage directions
3. **Multi-Speaker Drop**: lines belonging to ambiguous or multi-person attributes (e.g. `All`, `Both`, `Everyone`, `Monica and Chandler`) and unnamed generic roles (waiter, nurse, cop, etc.) were dropped entirely' only named characters are retained
4. **Dangling Stage Direction Merge**: stage entries continuing a disrupted dialogue line are concatenated to the previous turn before stage ablation; additionally, dialogue lines misformatted as scene descriptions are modified accordingly

To prevent temporal data leakage, I split the cleaned corpus by season: **Seasons 1-8 = Train, Season 9 = Validation, Season 10 = Test**.

### Dataset and Tokenizer -- `dataset.py`, `tokenizer.py`

I loaded a pre-trained GPT-2 tokenizer, `GPT2TokenizerFast` 
