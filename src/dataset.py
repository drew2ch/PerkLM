import json
import torch
from typing import Final, Dict
from torch.utils.data import Dataset

class FriendsDataset(Dataset):
    """ Dataset class for Friends transcript dialogue data (.json).
        Args:
            data: the .json corpus file (path name)
            tokenizer: custom GPT-2 tokenizer
            maxt: maximum token length, default 128
    """

    # hard code speakers
    SPEAKER_LOOKUP: Final[Dict[str, int]] = {"ROSS": 0, "MONICA": 1, "CHANDLER": 2,
                "JOEY": 3, "RACHEL": 4, "PHOEBE": 5,
                "GUNTHER": 6, "JANICE": 7, "RICHARD": 8,
                "CAROL": 9, "SUSAN": 10, "MIKE": 11, "OTHER": 12}
    
    def __init__(self, data, tokenizer, maxt = 128):
        self.sequences = []
        self.responders = []
        self.tokenizer = tokenizer
        self.maxt = maxt
        self.pad_id = tokenizer.pad_token_id

        # import corpus
        with open(data, 'r', encoding = 'utf-8') as f:
            corpus = json.load(f)

        def format_turn(turn: dict[str, str]) -> str:
            """ Context Format Helper
            """
            speaker = turn['speaker'].upper()
            return f"<SPEAKER={speaker}> {turn['text']}", speaker
        
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
                    context_str = "\n".join(format_turn(t)[0] for t in context)
                    response_str, responder = format_turn(response)
                    sequence = ("<CONTEXT>\n" + context_str + \
                                "\n</CONTEXT>\n<RESPONSE>\n" + \
                                response_str + "\n<EOT>")
                    self.sequences.append(sequence)
                    self.responders.append(self.SPEAKER_LOOKUP.get(
                        responder, self.SPEAKER_LOOKUP["OTHER"]))

    def __len__(self): return len(self.sequences)
    def __getitem__(self, index): 
        """ Tokenization on the fly
        """
        encoded = self.tokenizer(self.sequences[index], add_special_tokens = False,
                                  max_length = self.maxt,
                                  truncation = True,
                                  padding = 'max_length',
                                  return_tensors = 'pt')
        
        return {'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'responder': torch.tensor(self.responders[index], dtype = torch.long)}
