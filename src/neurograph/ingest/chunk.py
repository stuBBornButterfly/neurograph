def chunk_lines(text: str, max_words: int, overlap_words: int) -> list[str]:
    # pack whole lines into chunks of <= max_words; the next chunk repeats the last ~overlap_words for context
    chunks, cur = [], []
    for line in text.split("\n"):
        if cur and sum(len(l.split()) for l in cur) + len(line.split()) > max_words:
            chunks.append("\n".join(cur))
            tail, n = [], 0
            for l in reversed(cur):
                n += len(l.split())
                if n > overlap_words:
                    break
                tail.insert(0, l)
            cur = tail
        cur.append(line)
    if cur:
        chunks.append("\n".join(cur))
    return chunks
