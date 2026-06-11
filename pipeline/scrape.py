import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://edersoncorbari.github.io/friends-scripts/season/"
LOG_FILE = "./transcripts/scrape_log_2.txt"

# Regex elements
SCENE_RE = re.compile(r"\[(?:Scene|Flashback scene|Cold open):\s*(.*?)\]", re.I)
CREDIT_RE = re.compile(r"(Written by|Transcribed by|Trascribed by|Teleplay by|Story by|Directed by|Opening Credits|Closing Credits|"
                       r"Produced by|Final check|Previously on Friends|Opening Titles|Closing Titles|End Credits|END$)", re.I)
SKIP_RE = re.compile(
    r"^\[?("
    r"OPENING\s*(CREDITS|TITLES)?|CLOSING\s*(CREDITS|TITLES)?|"
    r"END\s*(CREDITS)?|COMMERCIAL\s*BREAK|FADE\s*(IN|OUT)|"
    r"THE\s*END|COLD\s*OPEN|TAG\s*SCENE|&nbsp;"
    r")\]?$", re.I)

# Speaker Line: "NAME: dialogue"
SPEAKER_RE = re.compile(r"^([A-Z][a-zA-Z]*(?:[\s\.\-'][A-Z][a-zA-Z]*){0,6}):\s+(\S.*)$")
INLINE_STAGE_RE = re.compile(r"\[.*?\]|\(.*?\)")
SPEAKER_BLOCKLIST = {
    "NOTE", "CUT TO", "INT", "EXT", "SCENE", "COLD OPEN",
    "OPENING", "CLOSING", "END", "COMMERCIAL", "CREDITS",
    "FADE IN", "FADE OUT", "TAG", "PREVIOUSLY",
    "TELEPLAY BY", "WRITTEN BY", "TRANSCRIBED BY", "STORY BY"}

def log(msg: str):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok = True)
    with open(LOG_FILE, 'a', encoding = "utf-8") as f:
        f.write(msg + '\n')

def is_valid_speaker(name: str) -> bool:
    """ Reject invalid or garbage names

        1. Not in structural blocklist
        2. Does not begin with a bracket
        3. Not purely numeric or punctuation
    """
    if name.strip().upper() in SPEAKER_BLOCKLIST: return False
    if name.lstrip().startswith(('[', '(')): return False
    if not re.search(r'[A-Za-z]', name): return False
    return True

def clean_dialogue(raw: str) -> str:
    """ Strip inline stage directions and collapse whitespace
    """
    t = INLINE_STAGE_RE.sub('', raw)
    t = re.sub(r'\s*\n\s*', ' ', t) # collapse embedded newlines
    return re.sub(r'\s{2,}', ' ', t).strip()

def url_exists(url: str) -> bool:
    """ Probe for dynamic episode scraping 
        (e.g. S1E24 exists, but S4E24 doesn't)
    """
    try:
        r = requests.get(url, timeout = 5)
        return r.status_code == 200
    except Exception:
        return False
    
def discover_episodes() -> list[tuple[int, int, str]]:
    """ Collect valid episodes
    """

    valid = []
    
    for season in range(9, 10):
        for episode in range(1, 30):

            url = f"{BASE_URL}{season:02d}{episode:02d}.html"
            if url_exists(url):
                valid.append((season, episode, url))

            url2 = f"{BASE_URL}{season:02d}{episode:02d}-{season:02d}{episode+1:02d}.html"
            if url_exists(url2):
                valid.append((season, episode, url2))

    return valid

"""
def extract_transcript(url: str) -> dict | None:
    ''' Scrape the full Friends Transcript corpus.
        https://edersoncorbari.github.io/friends/

        Each entry is one of:
            {"type": "dialogue", "scene": str|None, "speaker": str, "text": str}
            {"type": "stage",    "scene": str|None, "text": str}
    '''

    # generate GET request
    r = requests.get(url, timeout = 10)
    if r.status_code != 200: return None

    soup = BeautifulSoup(r.text, "html.parser")

    head = soup.find('head')
    if head: head.decompose()

    title_tag = soup.find('h1')
    title = title_tag.get_text(strip = True) if title_tag else None
    if title_tag: title_tag.decompose()

    hr = soup.find('hr')
    if hr:
        for node in list(hr.previous_siblings):
            if hasattr(node, 'decompose'): node.decompose()
        hr.decompose()

    for br in soup.find_all('br'):
        br.replace_with('\n')
    
    # lines = []
    # for p in soup.find_all(['p', 'div']):
        # if p.find_parent(['p', 'div']): continue
        # text = p.get_text(' ', strip = True).replace('**', '')
        # if text: lines.append(text)

    # Extract text blocks
    raw_text = soup.get_text("\n")
    lines = [ln.strip() for ln in raw_text.splitlines()]
    lines = [ln for ln in lines if ln]

    transcript = []
    current_scene = None

    for line in lines:
        line = line.strip().replace('\n', ' ')
        line = re.sub(r'\s{2,}', ' ', line)
        
        if not line: continue
        if CREDIT_RE.search(line): continue # Metadata
        if SKIP_RE.match(line): continue # Structural label

        scene_match = SCENE_RE.search(line)
        if scene_match:
            current_scene = scene_match.group(1).strip().rstrip('.')
            continue

        m = SPEAKER_RE.match(line)
        if m:
            name = m.group(1).strip()
            raw_diag = m.group(2).strip()

            if is_valid_speaker(name):
                speaker = name.title()
                dialogue = clean_dialogue(raw_diag)
                if dialogue:
                    transcript.append({
                        "type":    "dialogue",
                        "scene":   current_scene,
                        "speaker": speaker,
                        "text":    dialogue,
                    })
                continue
        
        stage = clean_dialogue(line)
        if stage:
            transcript.append({
                "type":  "stage",
                "scene": current_scene,
                "text":  stage,
            })
    
    return {"title": title, "transcript": transcript}
"""

def iter_line_nodes(body):
    """Yield top-level line containers — <p> children if present, else body.children directly."""
    has_p = any(getattr(n, 'name', None) == 'p' for n in body.children)
    containers = body.find_all('p') if has_p else [body]
    for container in containers:
        yield from container.children

def extract_transcript(url: str) -> dict | None:
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    if soup.head:
        soup.head.decompose()

    title_tag = soup.find('font', size="7") or soup.find('font', size="5") or soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else None
    if title_tag:
        title_tag.decompose()

    '''
    hr = soup.find('hr')
    if hr:
        for node in list(hr.previous_siblings):
            if hasattr(node, "decompose"):
                node.decompose()
        hr.decompose()
    '''
    for hr in soup.find_all('hr'):
        hr.decompose()
    transcript = []
    current_scene = None

    # ⭐ KEY CHANGE: scan entire body text, not <p>-restricted
    body = soup.find("body")
    '''
    if not body:
        return {"title": title, "transcript": []}

    # normalize <br> into separators (critical for E11)
    for br in body.find_all("br"):
        br.replace_with("\n")

    text = body.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    '''
    chunks = []
    current = ""
    for node in iter_line_nodes(body):
        if getattr(node, 'name', None) == 'br':
            if current.strip():
                chunks.append(current.strip())
            current = ""
        elif getattr(node, 'name', None) in ('b', 'i', 'em'):
            current += node.get_text(strip=True) + " "
        else:
            current += str(node)

    lines = [re.sub(r"\s+", " ", c).strip() for c in chunks if c.strip()]
    buffer_scene = None

    for line in lines:

        line = re.sub(r"\s+", " ", line)

        if CREDIT_RE.search(line) or SKIP_RE.match(line):
            continue

        # scene detection
        scene_match = SCENE_RE.search(line)
        if scene_match:
            buffer_scene = scene_match.group(1).strip().rstrip(".")
            current_scene = buffer_scene
            continue

        # speaker detection (E11 format: <b>Chandler:</b> OR Chandler:)
        speaker_match = re.match(r"(?:<b>)?([A-Za-z][A-Za-z\s\.\-']{1,40})(?:</b>)?:\s*(.+)", line)

        if speaker_match:
            speaker = speaker_match.group(1).strip()
            utterance = speaker_match.group(2).strip()

            if is_valid_speaker(speaker):
                transcript.append({
                    "type": "dialogue",
                    "scene": current_scene,
                    "speaker": speaker.title(),
                    "text": clean_dialogue(utterance),
                })
            continue

        # fallback stage direction
        stage = clean_dialogue(line)
        if stage:
            transcript.append({
                "type": "stage",
                "scene": current_scene,
                "text": stage,
            })

    return {"title": title, "transcript": transcript}

def get_path(season: int, episode: int, root: str = './transcripts') -> str:
    return os.path.join(root, f"S{season:02d}", f"S{season:02d}E{episode:02d}.json")

def save_episode(data: dict, season: int, episode: int, root: str = './transcripts'):
    """ Save transcript as .json
    """
    season_dir = os.path.join(root, f"S{season:02d}")
    os.makedirs(season_dir, exist_ok = True)
    with open(get_path(season, episode, root = root), "w", encoding = "utf-8") as f:
        json.dump(data, f, indent = 2, ensure_ascii = False)

if __name__ == "__main__":
    
    episodes = discover_episodes()
    print(f"Found {len(episodes)} episodes.")

    for season, episode, url in episodes:
        
        path = get_path(season, episode)
        if os.path.exists(path): continue

        log(f"START S{season}E{episode} {url}")

        data = None
        for attempt in range(3):
            try: 
                data = extract_transcript(url)
                if data is not None: break
            except Exception as e: 
                log(f"ERROR S{season}E{episode} attempt {attempt+1}: {e}")
            if data is None: time.sleep(1)

        if data: 
            save_episode(data, season, episode)
            log(f"SUCCESS S{season}E{episode}")
        else: log(f"SKIPPED S{season}E{episode} -- no data after retries.")
        
        time.sleep(0.3)
    