import re
from typing import List, Dict, Any
""""
Combined text chuncking using 
Section-based chunking - ensures everything in a chunk is semantically local (same heading / topic).
Paragraph-based chunking - preserves natural boundaries: sentences and code blocks stay together.
Character-based chunking (with overlap) - keeps chunks within a target size for embeddings / context, 
while overlap helps the agent maintain continuity across cuts.
"""

def split_into_sections(markdown_text) -> List[Dict[str, Any]]:

    """Split markdown into sections by heading level (#, ##, ###, ...)"""

    sections = []
    current_title = "__preamble__" # the intro text before any official section heading
    current_level = 0
    current_lines: List[str] = []

    heading_pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)

    def _add_to_section(title, level, lines) -> None:
        content = "\n".join(lines).strip()
        if not content:
            return
        sections.append({
            "title": title,
            "level": level,
            "content": content
        })

    for line in markdown_text.splitlines():
        head_match = heading_pattern.match(line)
        if head_match:
            _add_to_section(title=current_title, level=current_level, lines=current_lines)
            current_level = len(head_match.group(1))
            current_title = head_match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    _add_to_section(title=current_title, level=current_level, lines=current_lines)

    return sections

def split_into_paragraphs(section_text) -> List[str]:

    """Split section text into paragraphs/blocks using blank lines as separators.
    Keeps code blocks and lists as 'blocks' too."""

    # strip to avoid empty leading/trailing blocks
    if not section_text.strip():
        return []

    raw_blocks = re.split(r'\n\s*\n', section_text.strip())
    blocks = [b.strip() for b in raw_blocks if b.strip()]
    return blocks

def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i+size]
        result.append({'start': i, 'chunk': chunk})
        if i + size >= n:
            break

    return result

def chunk_repo_docs(repo_docs: List[Dict[str, Any]], max_chars: int = 1200, overlap_chars: int = 200) -> List[Dict[str, Any]]:
    """
    Hybrid chunking over all repo markdown docs.
    Returns list of dicts:
        {
            "text": str,
            "metadata": {...}
        }
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc in repo_docs:
        text = doc.get("content", "")
        filename = doc.get("filename", "")

        sections = split_into_sections(text) # by headings

        for section in sections:
            paragraphs = split_into_paragraphs(section_text=section["content"])
            if not paragraphs:
                continue

            section_chunks = sliding_window(seq=paragraphs, size=6, step=3)

            for i, win in enumerate(section_chunks):
                chunk_text = "\n\n".join(win["chunk"])
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "section_title": section["title"],
                        "section_level": section["level"],
                        "chunk_index": i,
                        "para_start": win["start"],
                    },
                })
        return all_chunks
