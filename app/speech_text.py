"""Conservative phrase boundaries for local speech synthesis."""
import re


def speech_chunks(text, minimum=45, maximum=180):
    """Keep short adjacent sentences together; split long text at punctuation.

    Never split decimals, initials or common Turkish/English abbreviations.
    Long unpunctuated sentences stay intact to preserve their prosody.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    boundaries = []
    for match in re.finditer(r'[.!?…]+[”"\']?\s+', text):
        prefix = text[:match.start()+1]
        token = prefix.rsplit(" ", 1)[-1].lower()
        if token in {"dr.", "prof.", "mr.", "mrs.", "ms.", "vb.", "vs.", "örn.", "sn."}:
            continue
        if re.search(r"\b[A-ZÇĞİÖŞÜ]\.$", prefix):
            continue
        boundaries.append(match.end())
    chunks, begin = [], 0
    for end in boundaries:
        if end - begin >= minimum and len(text) - begin > maximum:
            chunks.append(text[begin:end].strip())
            begin = end
    if text[begin:].strip():
        chunks.append(text[begin:].strip())
    return chunks
