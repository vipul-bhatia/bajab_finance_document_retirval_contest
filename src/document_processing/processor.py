import PyPDF2
import os
import tempfile
import requests
from urllib.parse import urlparse
from typing import List, Optional
from ..config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    """Handles document loading and chunking operations"""
    
    @staticmethod
    def load_document(file_path: str, chunk_size: int = CHUNK_SIZE) -> list:
        """Load document and split into chunks
        
        Required packages:
        - PyPDF2: pip install PyPDF2
        - python-docx: pip install python-docx
        """
        print(f"🔄 Processing document from file...")
        try:
            content = ""
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'txt':
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
            elif file_extension == 'pdf':
                # Requires: pip install PyPDF2
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n\n"
                        
            elif file_extension in ['docx', 'doc']:
                # Requires: pip install python-docx
                from docx import Document
                doc = Document(file_path)
                for para in doc.paragraphs:
                    content += para.text + "\n\n"
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Improved chunking algorithm
            chunks = DocumentProcessor._smart_chunk_text(content, chunk_size)
            
            print(f"✅ Processed document and split into {len(chunks)} chunks")
            return chunks
            
        except FileNotFoundError:
            print(f"Document file '{file_path}' not found. Please provide the document.")
            return []
        except Exception as e:
            print(f"Error loading document: {e}")
            return []
    
    @staticmethod
    def download_and_process_document(url: str) -> Optional[List[str]]:
        """
        Download document from URL and process it
        
        Args:
            url: URL of the document to download
            
        Returns:
            List of document chunks or None if failed
        """
        try:
            # Download the document
            print(f"📥 Downloading document from URL...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file extension from URL or Content-Type
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1].lower()
            
            if not file_extension:
                # Try to get extension from Content-Type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = '.pdf'
                else:
                    file_extension = '.pdf'  # Default to PDF
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            print(f"✅ Document downloaded successfully")
            
            # Process the downloaded document
            chunks = DocumentProcessor.load_document(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return chunks
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading document: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing downloaded document: {e}")
            return None
    
    @staticmethod
    def download_and_process_document_with_size(url: str, chunk_size: int) -> Optional[List[str]]:
        """
        Download document from URL and process it with custom chunk size
        
        Args:
            url: URL of the document to download
            chunk_size: Custom chunk size for processing
            
        Returns:
            List of document chunks or None if failed
        """
        try:
            # Download the document
            print(f"📥 Downloading document from URL...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file extension from URL or Content-Type
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1].lower()
            
            if not file_extension:
                # Try to get extension from Content-Type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = '.pdf'
                else:
                    file_extension = '.pdf'  # Default to PDF
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            print(f"✅ Document downloaded successfully")
            
            # Process the downloaded document with custom chunk size
            chunks = DocumentProcessor.load_document(temp_file_path, chunk_size)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return chunks
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading document: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing downloaded document: {e}")
            return None
    
    @staticmethod
    def _smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
        """
        Improved text chunking algorithm that respects sentence boundaries
        and ensures proper chunk sizes regardless of document structure.
        
        Args:
            text: The text to chunk
            chunk_size: Target size for each chunk (in characters)
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Clean and normalize the text
        text = text.strip()
        
        # First, try to split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If the paragraph itself is larger than chunk_size, split it further
            if len(paragraph) > chunk_size:
                # If we have content in current_chunk, save it first
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split large paragraph by sentences
                sub_chunks = DocumentProcessor._split_large_text(paragraph, chunk_size, overlap)
                chunks.extend(sub_chunks)
                
            else:
                # Check if adding this paragraph would exceed chunk_size
                test_chunk = current_chunk + ("\n\n" + paragraph if current_chunk else paragraph)
                
                if len(test_chunk) > chunk_size and current_chunk:
                    # Save current chunk and start new one
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Add paragraph to current chunk
                    current_chunk = test_chunk
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Only filter out completely empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    @staticmethod
    def _split_large_text(text: str, chunk_size: int, overlap: int = CHUNK_OVERLAP) -> list:
        """
        Split large text that exceeds chunk_size by sentence boundaries.
        If no sentence boundaries, split by word boundaries.
        """
        import re
        
        # Try to split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) == 1:
            # No sentence boundaries found, split by words
            return DocumentProcessor._split_by_words(text, chunk_size, overlap)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            if len(test_chunk) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap if possible
                if overlap > 0 and len(current_chunk) > overlap:
                    # Take last 'overlap' characters from previous chunk
                    overlap_text = current_chunk[-overlap:].strip()
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    @staticmethod
    def _split_by_words(text: str, chunk_size: int, overlap: int = CHUNK_OVERLAP) -> list:
        """
        Split text by word boundaries when sentence splitting isn't possible.
        """
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + (" " + word if current_chunk else word)
            
            if len(test_chunk) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0:
                    current_words = current_chunk.split()
                    if len(current_words) > 3:  # Keep some context
                        overlap_words = current_words[-3:]
                        current_chunk = " ".join(overlap_words) + " " + word
                    else:
                        current_chunk = word
                else:
                    current_chunk = word
            else:
                current_chunk = test_chunk
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks