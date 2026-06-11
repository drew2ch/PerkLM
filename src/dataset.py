import os
import json
import torch
from torch.utils.data import Dataset

class FriendsDataset(Dataset):
    """ Dataset class for Friends transcript dialogue data (.json).
        Args:
            data: the .json corpus file (path name)
            tokenizer: custom GPT-2 tokenizer
            maxt: maximum token length, default 128
    """
    def __init__(self, data, tokenizer, maxt = 128):
        self.data = []
        self.tokenizer = tokenizer
        self.maxt = maxt
        self.pad_id = tokenizer.pad_token_id

        # import corpus
        with open(data, 'r', encoding = 'utf-8') as f:
            corpus = json.load(f)

        def format_turn(turn: dict[str, str]) -> str:
            """ Context Format Helper
            """
            return f"<SPEAKER={turn['speaker'].upper()}> {turn['text']}"
        
        # unravel scenes and construct input sequence tensors
        for scene in corpus:
            turns = scene['turns']
            n = len(turns)

            # Context Windows (1,2,3)
            for i in range(n):
                for window in range(1, 4):
                    
                    if i - window < 0: continue
                    
                    context = turns[i - window:i]
                    response = turns[i]

                    # build sequence
                    context_str = "\n".join(format_turn(t) for t in context)
                    sequence = ("<CONTEXT>\n" + context_str + \
                                "\n</CONTEXT>\n<RESPONSE>\n" + \
                                format_turn(response) + "\n<EOT>")
                    encoded = tokenizer(sequence, add_special_tokens = False,
                                        max_length = self.maxt,
                                        truncation = True,
                                        padding = 'max_length',
                                        return_tensors = 'pt')
                    
                    self.data.append({
                        'input_ids': encoded['input_ids'].squeeze(0),
                        'attention_mask': encoded['attention_mask'].squeeze(0)})

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index]
