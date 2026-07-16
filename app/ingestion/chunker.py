from typing import List
import logfire

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Simple semantic chunker that splits text into chunks of a specified size."""
    with logfire.span("chunk_text", text_length=len(text)):
        if not text.strip():
            return []
        
        # Strictly split text into chunks to prevent OOM
        chunks = []
        start = 0
        text = text.replace('\n', ' ') # Clean up formatting
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += (chunk_size - overlap)
        
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"Chunked text into {len(valid_chunks)} chunks.")
        return valid_chunks
