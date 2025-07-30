import PyPDF2
import fitz
import os
import tempfile
import requests
import io
from urllib.parse import urlparse
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import email
import extract_msg
import gc
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import project-specific config
from ..config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    """Optimized document processor for AWS EC2 performance"""
    
    def __init__(self):
        # Configure requests session with connection pooling and retry strategy
        self.session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set optimal headers
        self.session.headers.update({
            'User-Agent': 'Document-Processor/1.0',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
    
    def download_and_process_document(self, url: str, chunk_size: int = CHUNK_SIZE) -> Optional[List[str]]:
        """
        Highly optimized document download and processing for AWS EC2.
        This method replaces the previous `download_and_process_document_optimized`.
        
        Key optimizations:
        1. Streaming download with progress tracking
        2. Memory-efficient processing
        3. Parallel page processing with controlled concurrency
        4. Aggressive garbage collection
        5. Connection pooling and retry logic
        """
        total_start = time.time()
        
        try:
            # Step 1: Optimized streaming download
            print(f" Starting optimized download from URL...")
            download_start = time.time()
            
            # Use streaming download for large files
            response = self.session.get(url, stream=True, timeout=(10, 30))  # Connect timeout, read timeout
            response.raise_for_status()
            
            # Get file size for progress tracking
            file_size = int(response.headers.get('content-length', 0))
            print(f"   -> File size: {file_size / (1024*1024):.2f} MB")
            
            # Stream download with memory management
            downloaded_data = bytearray()
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                if chunk:
                    downloaded_data.extend(chunk)
                    downloaded += len(chunk)
                    
                    # Progress indication for large files
                    if file_size > 0 and downloaded % (1024*1024) == 0:  # Every MB
                        progress = (downloaded / file_size) * 100
                        print(f"   -> Downloaded: {progress:.1f}%")
            
            # Convert to bytes for processing
            file_bytes = bytes(downloaded_data)
            del downloaded_data  # Free memory immediately
            gc.collect()
            
            download_end = time.time()
            print(f"   -> Download completed in: {download_end - download_start:.2f} seconds")
            
            # Step 2: Get file extension
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1].lower().replace('.', '')
            
            if not file_extension:
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = 'pdf'
                else:
                    file_extension = 'pdf'  # Default to PDF
            
            # Step 3: Memory-optimized processing
            processing_start = time.time()
            chunks = self._process_document_from_memory(file_bytes, file_extension, chunk_size)
            processing_end = time.time()
            
            print(f"   -> Processing completed in: {processing_end - processing_start:.2f} seconds")
            
            total_end = time.time()
            print(f"✅ Total time: {total_end - total_start:.2f} seconds")
            
            return chunks
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return None
        finally:
            # Cleanup
            gc.collect()

    def load_document(self, file_path: str, chunk_size: int = CHUNK_SIZE) -> list:
        """Load document from a local file path and split into chunks"""
        print(f"🔄 Processing document from local file: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            file_extension = os.path.splitext(file_path)[1].lower().replace('.', '')
            
            return self._process_document_from_memory(file_bytes, file_extension, chunk_size)
            
        except FileNotFoundError:
            print(f"❌ Document file '{file_path}' not found.")
            return []
        except Exception as e:
            print(f"❌ Error loading document from file: {e}")
            return []

    def load_document_from_memory(self, file_bytes: bytes, file_extension: str, chunk_size: int = CHUNK_SIZE) -> list:
        """Public method to process a document from memory bytes."""
        return self._process_document_from_memory(file_bytes, file_extension, chunk_size)

    def _process_document_from_memory(self, file_bytes: bytes, file_extension: str, chunk_size: int) -> List[str]:
        """
        Memory-optimized document processing with controlled concurrency.
        This is the central processing hub for all document types.
        """
        print(f" Processing {file_extension.upper()} document with memory optimization...")
        
        try:
            if file_extension == 'pdf':
                return self._process_pdf_optimized(file_bytes, chunk_size)
            elif file_extension == 'txt':
                content = file_bytes.decode('utf-8')
                return self._smart_chunk_text(content, chunk_size)
            elif file_extension in ['docx', 'doc']:
                return self._process_docx_optimized(file_bytes, chunk_size)
            elif file_extension == 'eml':
                return self._process_eml_optimized(file_bytes, chunk_size)
            elif file_extension == 'msg':
                return self._process_msg_optimized(file_bytes, chunk_size)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            print(f"❌ Error in memory-optimized processing: {e}")
            return []
    
    def _process_pdf_optimized(self, file_bytes: bytes, chunk_size: int) -> List[str]:
        """
        Highly optimized PDF processing with controlled parallel execution
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            print(f"   -> Processing {total_pages} pages")
            
            max_workers = min(4, max(1, total_pages // 10))
            print(f"   -> Using {max_workers} threads")
            
            page_texts = []
            
            if total_pages <= 5 or max_workers == 1:
                for page_num in range(total_pages):
                    try:
                        page = doc[page_num]
                        text = page.get_text()
                        if text.strip():
                            page_texts.append(text)
                        if page_num % 10 == 0:
                            gc.collect()
                    except Exception as e:
                        print(f"   -> Warning: Error processing page {page_num}: {e}")
                        continue
            else:
                batch_size = max(1, total_pages // max_workers)
                
                def process_page_batch(start_idx, end_idx):
                    batch_texts = []
                    for page_num in range(start_idx, min(end_idx, total_pages)):
                        try:
                            page = doc[page_num]
                            text = page.get_text()
                            if text.strip():
                                batch_texts.append((page_num, text))
                        except Exception as e:
                            print(f"   -> Warning: Error processing page {page_num}: {e}")
                    return batch_texts
                
                batches = [(i, i + batch_size) for i in range(0, total_pages, batch_size)]
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {executor.submit(process_page_batch, start, end): (start, end) for start, end in batches}
                    
                    all_page_results = []
                    for future in as_completed(future_to_batch):
                        try:
                            batch_results = future.result(timeout=30)
                            all_page_results.extend(batch_results)
                        except Exception as e:
                            start, end = future_to_batch[future]
                            print(f"   -> Batch {start}-{end} failed: {e}")
                    
                    all_page_results.sort(key=lambda x: x[0])
                    page_texts = [text for _, text in all_page_results]
            
            doc.close()
            
            print(f"   -> Extracted text from {len(page_texts)} pages")
            content = "\n\n".join(page_texts)
            del page_texts
            gc.collect()
            
            chunks = self._smart_chunk_text(content, chunk_size)
            print(f"✅ PDF processed into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Error in PDF processing: {e}")
            return []
    
    def _process_docx_optimized(self, file_bytes: bytes, chunk_size: int) -> List[str]:
        """Optimized DOCX processing"""
        try:
            from docx import Document
            doc_obj = Document(io.BytesIO(file_bytes))
            
            content_parts = []
            batch_size = 100
            
            paragraphs = doc_obj.paragraphs
            for i in range(0, len(paragraphs), batch_size):
                batch = paragraphs[i:i + batch_size]
                batch_text = "\n\n".join(para.text for para in batch if para.text.strip())
                if batch_text.strip():
                    content_parts.append(batch_text)
                if i % 500 == 0:
                    gc.collect()
            
            content = "\n\n".join(content_parts)
            return self._smart_chunk_text(content, chunk_size)
            
        except Exception as e:
            print(f"❌ Error processing DOCX: {e}")
            return []
    
    def _process_eml_optimized(self, file_bytes: bytes, chunk_size: int) -> List[str]:
        """Optimized EML processing"""
        try:
            msg = email.message_from_bytes(file_bytes)
            content = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            content += payload.decode() + "\n\n"
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    content = payload.decode()
            
            subject = msg.get('subject', 'No Subject')
            from_addr = msg.get('from', 'Unknown')
            to_addr = msg.get('to', 'Unknown')
            
            full_content = f"Subject: {subject}\n\nFrom: {from_addr}\n\nTo: {to_addr}\n\n{content}"
            return self._smart_chunk_text(full_content, chunk_size)
            
        except Exception as e:
            print(f"❌ Error processing EML: {e}")
            return []
    
    def _process_msg_optimized(self, file_bytes: bytes, chunk_size: int) -> List[str]:
        """Optimized MSG processing"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.msg', delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                msg = extract_msg.Message(temp_path)
                content = f"Subject: {msg.subject}\n\nFrom: {msg.sender}\n\nTo: {msg.to}\n\n{msg.body}"
                msg.close()
                return self._smart_chunk_text(content, chunk_size)
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ Error processing MSG: {e}")
            return []
    
    def _smart_chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Optimized text chunking with memory efficiency
        """
        if not text.strip():
            return []
        
        text = text.strip()
        
        if len(text) > 1000000:
            return self._chunk_large_text_optimized(text, chunk_size, overlap)
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                sub_chunks = self._split_large_text(paragraph, chunk_size, overlap)
                chunks.extend(sub_chunks)
            else:
                test_chunk = current_chunk + ("\n\n" + paragraph if current_chunk else paragraph)
                
                if len(test_chunk) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def _chunk_large_text_optimized(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Memory-optimized chunking for very large texts
        """
        chunks = []
        text_length = len(text)
        start = 0
        
        while start < text_length:
            end = min(start + chunk_size * 10, text_length)
            section = text[start:end]
            
            section_chunks = self._smart_chunk_text(section, chunk_size, overlap)
            chunks.extend(section_chunks)
            
            start = end - overlap
            
            if len(chunks) % 100 == 0:
                gc.collect()
        
        return chunks
    
    def _split_large_text(self, text: str, chunk_size: int, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split large text by sentences with memory optimization"""
        import re
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) == 1:
            return self._split_by_words(text, chunk_size, overlap)
        
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
    
    def _split_by_words(self, text: str, chunk_size: int, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split by words with memory optimization"""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + (" " + word if current_chunk else word)
            
            if len(test_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                if overlap > 0:
                    current_words = current_chunk.split()
                    if len(current_words) > 5:
                        overlap_words = current_words[-5:]
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