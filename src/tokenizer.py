import json
from pathlib import Path
from transformers import GPT2TokenizerFast

CORPUS_DIR = Path('corpus/corpus.json')

def main():

    # Load Corpus
    with open(CORPUS_DIR, 'r', encoding = 'utf-8') as f:
        corpus = json.load(f)
    if not corpus:
        raise ValueError('Corpus not found')

    # Initialize GPT-2 Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Get unique speaker names
    speakers = set()
    for scene in corpus:
        for turn in scene.get('turns', []):
            speaker = turn.get('speaker')
            if speaker: speakers.add(speaker.strip())
    speakers = sorted(speakers)

    # append special tokens
    SPECIAL_TOKENS = [f"<SPEAKER={speaker.upper()}>" for speaker in speakers] + \
                    ["<CONTEXT>", "</CONTEXT>", "<RESPONSE>", "<EOT>"]
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # save GPT-2 tokenizer
    tokenizer.save_pretrained('tokenizer')

if __name__ == "__main__":
    main()
