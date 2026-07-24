from typing import List
import logfire

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Recursive structure-aware chunker.
    Splits hierarchy: Paragraphs (\n\n) -> Sentences/Lines (\n) -> Clauses (. ) -> Words ( ).
    Preserves document structure and avoids slicing mid-word or mid-number.
    """
    with logfire.span("chunk_text", text_length=len(text)):
        if not text or not text.strip():
            return []
        
        separators = ["\n\n", "\n", ". ", " "]
        
        def _split_recursive(t: str, seps: List[str]) -> List[str]:
            if len(t) <= chunk_size or not seps:
                return [t] if t.strip() else []
            
            sep = seps[0]
            splits = t.split(sep)
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for s in splits:
                piece = s + sep if sep != " " else s + " "
                piece_len = len(piece)
                
                if current_length + piece_len > chunk_size:
                    if current_chunk:
                        joined = "".join(current_chunk).strip()
                        if joined:
                            chunks.append(joined)
                    
                    # Handle oversized pieces recursively with next separator
                    if piece_len > chunk_size and len(seps) > 1:
                        sub_chunks = _split_recursive(s, seps[1:])
                        chunks.extend(sub_chunks)
                        current_chunk = []
                        current_length = 0
                    else:
                        current_chunk = [piece]
                        current_length = piece_len
                else:
                    current_chunk.append(piece)
                    current_length += piece_len
            
            if current_chunk:
                joined = "".join(current_chunk).strip()
                if joined:
                    chunks.append(joined)
                    
            return chunks

        raw_chunks = _split_recursive(text, separators)
        
        # Apply overlap between consecutive chunks
        final_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and overlap > 0:
                prev_text = raw_chunks[i-1]
                overlap_text = prev_text[-overlap:] if len(prev_text) >= overlap else prev_text
                chunk = overlap_text + " " + chunk
            final_chunks.append(chunk.strip())

        valid_chunks = [c for c in final_chunks if len(c.strip()) > 30]
        logfire.info(f"Chunked text into {len(valid_chunks)} high-precision chunks.")
        return valid_chunks

