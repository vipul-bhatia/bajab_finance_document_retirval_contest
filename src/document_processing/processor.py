class DocumentProcessor:
    """Handles document loading and chunking operations"""
    
    @staticmethod
    def load_document(file_path: str, chunk_size: int = 500) -> list:
        """Load document and split into chunks"""
        print(f"🔄 Processing document from file...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split document into chunks by paragraphs or sentences
            # You can modify this logic based on your document structure
            paragraphs = content.split('\n\n')
            chunks = []
            
            current_chunk = ""
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                # If adding this paragraph would exceed chunk_size, save current chunk
                if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            
            # Add the last chunk if it exists
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            print(f"✅ Processed document and split into {len(chunks)} chunks")
            return chunks
            
        except FileNotFoundError:
            print(f"Document file '{file_path}' not found. Please provide the document.")
            return []
        except Exception as e:
            print(f"Error loading document: {e}")
            return [] 