import os
import re
import json
from itertools import groupby
from pathlib import Path

TRANSCRIPT_DIR = Path("./transcripts")
OUT_PATH = Path('./corpus')

# Speaker Aliases (Typos, alter egos, inconsistencies, etc)
SPEAKER_ALIASES = {
    # Alter egos
    "Fat Monica": "Monica",
    "Fake Monica": "Monica",
    "Big Nosed Rachel": "Rachel",
    # Main cast shortcuts
    "Mnca": "Monica",
    "MOnica": "Monica",
    "Monica;": "Monica",
    "Rache": "Rachel",
    "Racel": "Rachel",
    "Rach": "Rachel",
    "Phoe": "Phoebe",
    "Chan": "Chandler",
    "Chandlers": "Chandler",
    "Estl": "Estelle",
    "Fbob": "Fun Bobby",

    # Period normalization
    "Dr Green": "Dr. Green",
    "Mrs Green": "Mrs. Green",
    "Phoebe Sr": "Phoebe Sr.",
    "Phoebe Sr.": "Phoebe Sr.",
    "Frank Sr": "Frank Sr.",

    # Alternate spellings
    "Dr. Leedbetter": "Dr. Ledbetter",
    "Dr. Drake Remoray": "Dr. Drake Ramoray",
    "Dr. Stryker Remoray": "Dr. Stryker Ramoray",
    "Allesandro": "Alessandro",
    "Cailin": "Caitlin",
    "Jeannine": "Janine",   # same character — verify episode context
    "Jeanette": "Janine",   # likely same

    # Maître d' variants
    "Matire'd": "Maitre d'",
    "Maitre d'": "Maitre d'",
    "Maître d'": "Maitre d'",

    # Croupier
    "Croupler": "Croupier",
    "The Croupier": "Croupier",

    # C.H.E.E.S.E.
    "C.H.E.E.S.E": "C.H.E.E.S.E.",

    # Prof
    "Prof. Sherman": "Professor Sherman"}

# Invalid patterns
DROP_PATTERNS = [
    r"^\(.*\)$", # (stage direction)
    r"^\[.*\]$", # [Scene description]
    r"^Commercial", # Commercial break
    r"^End", # End, Ending Credits, "END"
    r"^Credits",
    r"^To Be Continued",
    r"^The Next Morning"]

# Exact Drops
DROP_SPEAKERS_EXACT = {
    "All", "Both", "Everyone", "Everybody", "Gang", "Guys", "Girls",
    "Others", "Quartet", "Kids",
    # Inanimate
    "Machine", "Hold Voice", "Hypnosis Tape", "Intercom", "TV",
    "TV Announcer", "Voice", "Message", "Second Message", "Second message",
    "Commercial", "Announcer",
    # Script artifacts
    "Rtst", "Professore Clerk"
}

# Pattern drops — generic roles
DROP_SPEAKER_PATTERNS = [
    r"^(The\s+)?Waiter(\s+No\.\s*\d+|#\d+)?$",
    r"^(The\s+)?Waitress$",
    r"^(A\s+)?Nurse(\s+#\d+)?$",
    r"^(The\s+)?Doctor$",
    r"^(The\s+)?Director$",
    r"^(Stage\s+)?Director$",
    r"^(The\s+)?A\.D\.?$",
    r"^(The\s+)?(Casting\s+)?Director(\s+#\d+)?$",
    r"^(The\s+)?Teacher$",
    r"^(The\s+)?Cooking Teacher$",
    r"^(The\s+)?Acting Teacher$",
    r"^(The\s+)?Instructor$",
    r"^(A\s+)?Student$",
    r"^(Female|Male)\s+Student$",
    r"^(First|Second)\s+Dorm\s+Guy$",
    r"^(The\s+)?(Fireman|Firemen)(\s+(#|No\.)\s*\d+)?$",
    r"^(The\s+)?Policeman$",
    r"^(The\s+)?Cop$",
    r"^(The\s+)?Customer$",
    r"^\d+(st|nd|rd|th)\s+Customer$",
    r"^(The\s+)?Salesman$",
    r"^(The\s+)?Saleslady$",
    r"^(The\s+)?Saleswoman$",
    r"^(The\s+)?Receptionist$",
    r"^(The\s+)?Security Guard$",
    r"^(The\s+)?Interviewer$",
    r"^(The\s+)?Supervisor$",
    r"^(The\s+)?Photographer$",
    r"^(The\s+)?Stripper$",
    r"^(The\s+)?Minister$",
    r"^(The\s+)?Rabbi$",
    r"^(The\s+)?Judge$",
    r"^Woman(\s+No\.\s*\d+)?$",
    r"^Man(\s+At.*)?$",
    r"^Woman(\s+At.*|On Train)?$",
    r"^(Little\s+)?Girl$",
    r"^Guy(\s+#\d+)?$",
    r"^Friend\s+No\.\s*\d+$",
    r"^(The\s+)?Lurker$",
    r"^(The\s+)?Fan$",
    r"^(The\s+)?Old Man$",
    r"^(The\s+)?Hot Girl$",
    r"^(The\s+)?Food Critic$",
    r"^(The\s+)?(Potential\s+)?Roommate$",
    r"^(The\s+)?Porsche Owner$",
    r"^(The\s+)?Dry Cleaner$",
    r"^(The\s+)?Housekeeper$",
    r"^(The\s+)?(Singing\s+)?Man$",
    r"^(The\s+)?Woman$",
    r"^(The\s+)?Producer$",
    r"^(The\s+)?Museum Official$",
    r"^(A\s+)?Crew Member$",
    r"^(The\s+)?Grip$",
    r"^(The\s+)?Paramedic$",
    r"^Flight Attendant$",
    r"^(Ticket\s+)?(Counter\s+)?Attendant$",
    r"^Airline Employee$",
    r"^(Hotel|Front Desk)?\s*Clerk$",
    r"^(Female\s+)?Clerk$",
    r"^Shop assistant$",
    r"^Kitchen Worker$",
    r"^Sleep Clinic Worker$",
    r"^Delivery Room\s*Nurse$",
    r"^PBS Volunteer$",
    r"^Bank Officer$",
    r"^Stage Manager$",
    r"^Tour Guide$",
    r"^(Blackjack\s+)?Dealer$",
    r"^(A\s+)?Drunken Gambler$",
    r"^Older Scientist$",
    r"^Another Scientist$",
    r"^(A\s+)?Waiter in Drag$",
    r"^Bandleader$",
    r"^(The\s+)?Conductor$",
    r"^Gym Employee$",
    r"^Guest\s+#\d+$",
    r"^(Boy in the Cape|Cowgirl|Hitchhiker|Passerby|Stranger)$",
    r"^Anxious Wedding Guest$",
    r"^Fat Girl$",
    r"^Second Girl$",
    r"^(The\s+)?Vampire$",
    r"^Bitter (lady|woman)$",
    r"^(The\s+)?Smoking Woman$"]

MAIN = ["Rachel", "Ross", "Chandler", "Monica", "Joey", "Phoebe",
        "Janice", "Gunther", "Richard", "Susan", "Carol", "Mike"]

def classify_speaker(name: str) -> str:
    return name if name in MAIN else "Other"

def is_multi_speaker(s: str) -> bool:
    """ Detects multiple speakers (e.g. Monica and Chandler)
    """
    patterns = [
        r"\band\b", r"&", r" and ",
        r"Everyone", r"\bAll\b", r"\bBoth\b",
        r"\bGuys\b", r"\bGirls\b", r"\bGang\b",
        r"Quartet", r"Others", r"Everybody"]
    return any(re.search(p, s, re.IGNORECASE) for p in patterns)

def normalize_speaker(raw: str) -> str | None:
    """ Standardize speaker names
    """
    s = raw.strip()

    # Drop garbage
    for pat in DROP_PATTERNS:
        if re.match(pat, s, re.IGNORECASE):
            return None
    
    s_title = s.title()
    if s in DROP_SPEAKERS_EXACT or s_title in DROP_SPEAKERS_EXACT: return None
        
    # Drop multi-speaker
    for pat in DROP_SPEAKER_PATTERNS:
        if re.fullmatch(pat, s, re.IGNORECASE):
            return None
    
    # drop single-word non-name artifacts and multiple speakers
    if is_multi_speaker(s_title): return None
    if s.islower() and len(s) > 1: return None

    return SPEAKER_ALIASES.get(s_title, s_title)

def stage_to_dialogue(transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    """ Convert misspecified "stage" entries to "dialogue" entries
    """
    fixed_transcript = []
    for entry in transcript:
        new_entry = entry.copy()
        text_content = new_entry.get('text', '')

        if ':' in text_content:
            parts = text_content.split(':', 1)
            new_entry['type'] = 'dialogue'
            new_entry['speaker'] = parts[0].strip().replace(':', '').title()
            new_entry['text'] = parts[1].strip()
        
        fixed_transcript.append(new_entry)

    return fixed_transcript

def merge_dialogue(transcript: list[dict[str,str]]):
    """ Concatenate dangling "stage" dialogues to preceding dialogues
        and ablate residual bracketed/parenthetical stage directions
    """
    merged_transcript = []
    last_dialogue = None
    can_append = False
    sentence_enders = (".", "!", "?", "\"", "'", "]", ")", "”", "’")

    for entry in transcript:
        item = entry.copy()
        item_text = item.get('text', '').strip()

        if item.get('type') == 'dialogue':
            merged_transcript.append(item)
            last_dialogue = item

            if item_text and item_text.endswith(sentence_enders):
                can_append = False
            else: can_append = True

        elif item.get('type') == 'stage':
            if 'scene' in item_text.lower():
                can_append = False
            if can_append and last_dialogue is not None:
                if item_text: last_dialogue['text'] += ' ' + item_text
                if item_text.endswith(sentence_enders): can_append = False
                continue

            merged_transcript.append(item)

    ablation_pattern = re.compile(r'\[[^\]]*\]?|\([^)]*\)?|\{[^\}]*\}?')
    
    for entry in merged_transcript:
        if entry.get('type') == 'dialogue':
            text = entry.get('text', '')
            clean_text = ablation_pattern.sub('', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            entry['text'] = clean_text
    
    return merged_transcript

def main():

    os.makedirs('./sequences', exist_ok = True)
    EPISODE_RE = re.compile(r'S(?P<season>\d+)E(?P<episode>\d+)', re.IGNORECASE)
    corpus = []

    # Traverse all .json files
    json_files = list(TRANSCRIPT_DIR.rglob('*.json'))
    json_files.sort()
    for json_file in json_files:
        print(f'Processing: {json_file}')
        try:
            with open(json_file, 'r', encoding = 'utf-8') as f:
                transcript_data = json.load(f)
            transcript = transcript_data['transcript']

            # 1. Convert misspecified "stage" entries to "dialogue" entries
            transcript = stage_to_dialogue(transcript)
            # 2. Concatenate dangling "stage" dialogues to preceding dialogues
            transcript = merge_dialogue(transcript)
            # 3. Normalize Speaker Names
            cleaned_transcript = []
            for entry in transcript:
                if entry.get('type') == 'dialogue' and 'speaker' in entry:
                    normalized = normalize_speaker(entry['speaker'])
                    if normalized is None: continue
                    
                    entry['speaker'] = classify_speaker(normalized)
                    cleaned_transcript.append(entry)
                else: cleaned_transcript.append(entry)

            # Save modified JSON
            transcript_data['transcript'] = cleaned_transcript
            with open(json_file, 'w', encoding = 'utf-8') as f:
                json.dump(transcript_data, f, indent = 2, ensure_ascii = False)

            # Collect turns and append to corpus
            match = EPISODE_RE.search(json_file.name)
            if match:
                season = int(match.group('season'))
                episode = int(match.group('episode'))
                print(f'Processing Season {season} Episode {episode}')

                scene_index = 0
                for scene_name, group in groupby(cleaned_transcript, key = lambda x: x.get('scene')):
                    dialogues = [d for d in group if d.get('type') == 'dialogue']
                    if dialogues:
                        turns = [{'speaker': d['speaker'], 'text': d['text']} for d in dialogues]
                        corpus.append({'season': season, 'episode': episode,
                                    'scene': scene_index, 'turns': turns})
                        scene_index += 1
            else:
                print(f'File {json_file.name} does not match Regex format.')

        except Exception as e:
            print(f'Error reading {json_file}: {e}')
    
    # save final master corpus
    os.makedirs('./corpus', exist_ok = True)
    if corpus:
        with open(os.path.join(OUT_PATH, 'corpus.json'), 'w', encoding = 'utf-8') as f:
            json.dump(corpus, f, indent = 2, ensure_ascii = False)
        print(f'Master Corpus saved with {len(corpus)} total scenes.')

        # train/val/test split
        train_corpus = [x for x in corpus if x['season'] <= 8]
        val_corpus   = [x for x in corpus if x['season'] == 9]
        test_corpus  = [x for x in corpus if x['season'] == 10]

        splits = {'train.json': train_corpus,
                    'val.json': val_corpus,
                    'test.json': test_corpus}
        
        for filename, split in splits.items():
            with open(os.path.join(OUT_PATH, filename), 'w', encoding = 'utf-8') as f:
                json.dump(split, f, indent = 2, ensure_ascii = False)
        
        print(
            f'Train: {len(train_corpus)} scenes | '
            f'Validation: {len(val_corpus)} scenes | '
            f'Test: {len(test_corpus)} scenes')   

    else: print('\nNo dialogue data was collected into the corpus.')

if __name__ == "__main__":
    main()
    