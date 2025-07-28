import PyPDF2
import os
import io
import requests
from urllib.parse import urlparse
from typing import List, Optional
from ..config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    """Handles document loading and chunking operations"""

    @staticmethod
    def load_document_from_memory(file_stream: io.BytesIO, file_extension: str, chunk_size: int = CHUNK_SIZE) -> list:
        """Load document from in-memory stream and split into chunks"""
        print(f"🔄 Processing document from memory...")
        try:
            content = ""
            # Corrected the condition to check for '.pdf'
            if file_extension == '.pdf':
                pdf_reader = PyPDF2.PdfReader(file_stream)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n\n"
            else:
                # Assuming text for other types, which might need more robust handling
                content = file_stream.read().decode('utf-8', errors='ignore')

            chunks = DocumentProcessor._smart_chunk_text(content, chunk_size)
            print(f"✅ Processed document and split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error loading document from memory: {e}")
            return []

    @staticmethod
    def download_and_process_document_with_size(url: str, chunk_size: int) -> Optional[List[str]]:
        """
        Download document from URL and process it with custom chunk size in memory
        """
        try:
            print(f"📥 Downloading document from URL...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            file_extension = os.path.splitext(urlparse(url).path)[1].lower() or '.pdf'

            with io.BytesIO(response.content) as file_stream:
                chunks = DocumentProcessor.load_document_from_memory(file_stream, file_extension, chunk_size)

            return chunks

        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading document: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing downloaded document: {e}")
            return None

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
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n\n"
                        
            elif file_extension in ['docx', 'doc']:
                from docx import Document
                doc = Document(file_path)
                for para in doc.paragraphs:
                    content += para.text + "\n\n"
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
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
        """
        try:
            print(f"📥 Downloading document from URL...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1].lower()
            
            if not file_extension:
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = '.pdf'
                else:
                    file_extension = '.pdf'  # Default to PDF
            
            with io.BytesIO(response.content) as file_stream:
                chunks = DocumentProcessor.load_document_from_memory(file_stream, file_extension)
            
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
        """
        if not text.strip():
            return []
        
        text = text.strip()
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                sub_chunks = DocumentProcessor._split_large_text(paragraph, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                test_chunk = current_chunk + ("\n\n" + paragraph if current_chunk else paragraph)
                if len(test_chunk) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    @staticmethod
    def _split_large_text(text: str, chunk_size: int, overlap: int = CHUNK_OVERLAP) -> list:
        """
        Split large text by sentence boundaries.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) == 1:
            return DocumentProcessor._split_by_words(text, chunk_size, overlap)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            if len(test_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:].strip()
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    @staticmethod
    def _split_by_words(text: str, chunk_size: int, overlap: int = CHUNK_OVERLAP) -> list:
        """
        Split text by word boundaries.
        """
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + (" " + word if current_chunk else word)
            
            if len(test_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if overlap > 0:
                    current_words = current_chunk.split()
                    if len(current_words) > 3:
                        overlap_words = current_words[-3:]
                        current_chunk = " ".join(overlap_words) + " " + word
                    else:
                        current_chunk = word
                else:
                    current_chunk = word
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks