"""
Text utilities: Vietnamese tokenisation for BM25, document chunking, and
multiple-choice question parsing.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List

import config

# --------------------------------------------------------------------------- #
# Vietnamese word segmentation for BM25                                        #
# --------------------------------------------------------------------------- #
try:
    from pyvi import ViTokenizer

    def _segment(text: str) -> str:
        return ViTokenizer.tokenize(text)

except Exception:  # pyvi missing -> fall back to whitespace
    def _segment(text: str) -> str:
        return text


def tokenize_vi(text: str) -> List[str]:
    """
    Lowercase + word-segment + split into tokens for BM25.
    pyvi joins multi-syllable words with '_' which keeps Vietnamese phrases
    together (e.g. 'truy_vấn'), greatly improving lexical matching.
    """
    text = text.lower()
    seg = _segment(text)
    # keep word characters (incl. the '_' joiner and Vietnamese diacritics)
    tokens = re.findall(r"[0-9A-Za-zÀ-ỹ_]+", seg, flags=re.UNICODE)
    return tokens


# --------------------------------------------------------------------------- #
# Chunking                                                                     #
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\;:])\s+|\n{2,}")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # collapse 3+ newlines but keep paragraph boundaries
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """
    Sentence-aware sliding-window chunker measured in words.

    Sentences are grouped until the word budget is reached, then a new chunk
    starts while carrying `overlap` words from the tail of the previous chunk.
    This keeps semantic boundaries intact while preserving local context.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE_WORDS
    overlap = overlap or config.CHUNK_OVERLAP_WORDS

    text = _normalise(text)
    if not text:
        return []

    sentences = split_sentences(text)
    if not sentences:
        sentences = [text]

    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0

    for sent in sentences:
        words = sent.split()
        n = len(words)

        # A single very long sentence: hard-split it by words.
        if n > chunk_size:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_words = [], 0
            for i in range(0, n, chunk_size - overlap):
                piece = words[i : i + chunk_size]
                chunks.append(" ".join(piece))
            continue

        if cur_words + n > chunk_size and cur:
            chunks.append(" ".join(cur))
            # carry overlap words from the end of the current chunk
            carried = " ".join(cur).split()[-overlap:] if overlap > 0 else []
            cur = list(carried)
            cur_words = len(cur)

        cur.extend(words)
        cur_words += n

    if cur:
        chunks.append(" ".join(cur))

    # de-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for c in chunks:
        key = c.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


# --------------------------------------------------------------------------- #
# Multiple-choice question parsing                                             #
# --------------------------------------------------------------------------- #
@dataclass
class MCQuestion:
    stem: str                 # the question text without the options
    options: Dict[str, str]   # {"A": "...", "B": "...", ...}
    raw: str                  # original string

    @property
    def has_options(self) -> bool:
        return len(self.options) >= 2

    def option_letters(self) -> List[str]:
        return sorted(self.options.keys())


# Matches option markers like "A.", "A)", "A:", "(A)", " A -" at boundaries.
_OPTION_RE = re.compile(
    r"(?:(?<=^)|(?<=\s)|(?<=\())\(?\s*([A-Da-d])\s*[\.\)\:\-]\s+",
)


def parse_mc_question(question: str) -> MCQuestion:
    """
    Split a raw MC question string into stem + {letter: option_text}.

    Handles formats like:
        "RAG là gì? A. xxx B. yyy C. zzz D. ttt"
        "... \nA) xxx \nB) yyy ..."
    Falls back gracefully (no options) if the pattern isn't found.
    """
    raw = question.strip()
    text = unicodedata.normalize("NFC", raw)

    matches = list(_OPTION_RE.finditer(text))
    if len(matches) < 2:
        return MCQuestion(stem=raw, options={}, raw=raw)

    stem = text[: matches[0].start()].strip()
    options: Dict[str, str] = {}

    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        opt_text = text[start:end].strip(" \n\t.-")
        if letter not in options and opt_text:
            options[letter] = opt_text

    if len(options) < 2:
        return MCQuestion(stem=raw, options={}, raw=raw)

    return MCQuestion(stem=stem or raw, options=options, raw=raw)


_LETTER_RE = re.compile(r"[ABCD]")


def extract_answer_letter(text: str, valid: List[str] | None = None) -> str | None:
    """
    Pull a single A/B/C/D letter out of an LLM response.

    Strategy:
      1. If the whole (stripped) reply is one letter -> use it.
      2. Look for patterns like 'Answer: B', 'Đáp án: C', 'B)'.
      3. Otherwise take the LAST standalone A-D letter (reasoning models often
         restate options then end with the final choice).
    Returns None if nothing valid is found.
    """
    if not text:
        return None
    valid = valid or ["A", "B", "C", "D"]
    t = text.strip().upper()

    # 1) exact single letter
    if t in valid:
        return t

    # 2) JSON {"answer":"B"} (common for small Qwen MCQ prompts)
    m_json = re.search(r'"ANSWER"\s*:\s*"([ABCD])"', t)
    if m_json and m_json.group(1) in valid:
        return m_json.group(1)

    # 3) explicit conclusion phrases (EN + VI). Take the LAST match.
    concl = re.findall(
        r"(?:CORRECT\s+ANSWER\s+IS|ANSWER\s+IS|FINAL\s+ANSWER\s*[:\-]?|"
        r"ANSWER\s*[:\-]|ĐÁP\s*ÁN\s*(?:LÀ|ĐÚNG\s*LÀ)?\s*[:\-]?|"
        r"DAP\s*AN\s*(?:LA)?\s*[:\-]?|CHỌN|CHON)\s*\(?\s*([ABCD])\b",
        t,
    )
    concl = [x for x in concl if x in valid]
    if concl:
        return concl[-1]

    # 3) leading 'B.' / 'B)' / '(B)'
    m = re.match(r"\(?\s*([ABCD])\s*[\.\)\:\-]", t)
    if m and m.group(1) in valid:
        return m.group(1)

    # 4) last standalone valid letter
    found = [ch for ch in _LETTER_RE.findall(t) if ch in valid]
    if found:
        return found[-1]
    return None
