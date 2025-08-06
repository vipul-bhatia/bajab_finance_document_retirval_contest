import PyPDF2
import fitz
import os
import tempfile
import requests
import io
from urllib.parse import urlparse
from typing import List, Optional
from ..config import CHUNK_SIZE, CHUNK_OVERLAP
import time
import email
import extract_msg

import zipfile

class DocumentProcessor:
    """Handles document loading and chunking operations"""
    
    @staticmethod
    def load_document(file_path: str, chunk_size: int = CHUNK_SIZE) -> list:
        """Load document and split into chunks
        
        Required packages:
        - PyMuPDF: pip install PyMuPDF (much faster than PyPDF2)
        - python-docx: pip install python-docx
        - extract-msg: pip install extract-msg
        """
        print(f"🔄 Processing document from file...")
        try:
            content = ""
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'txt':
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
            elif file_extension == 'pdf':
                # Use PyMuPDF (fitz) for much faster PDF processing
                with fitz.open(stream=file_path, filetype="pdf") as doc:
                    for page in doc:
                        content += page.get_text() + "\n\n"
                        
            elif file_extension in ['docx', 'doc']:
                # Requires: pip install python-docx
                from docx import Document
                doc = Document(file_path)
                for para in doc.paragraphs:
                    content += para.text + "\n\n"

            elif file_extension == 'eml':
                with open(file_path, 'r', encoding='utf-8') as f:
                    msg = email.message_from_file(f)
                    # Get email body
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                content += part.get_payload(decode=True).decode() + "\n\n"
                    else:
                        content = msg.get_payload(decode=True).decode()
                    # Add subject and other metadata
                    content = f"Subject: {msg['subject']}\n\nFrom: {msg['from']}\n\nTo: {msg['to']}\n\n{content}"

            elif file_extension == 'msg':
                msg = extract_msg.Message(file_path)
                content = f"Subject: {msg.subject}\n\nFrom: {msg.sender}\n\nTo: {msg.to}\n\n{msg.body}"
                msg.close()
            
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
        Download document from URL and process it (in-memory processing)
        
        Args:
            url: URL of the document to download
            
        Returns:
            List of document chunks or None if failed
        """
        try:
            # Download the document directly into memory
            print(f"📥 Downloading document from URL...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Get file extension
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1].lower().replace('.', '')
            
            if not file_extension:
                # Try to get extension from Content-Type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = 'pdf'
                elif 'zip' in content_type:
                    file_extension = 'zip'
                elif 'image' in content_type:
                    file_extension = content_type.split('/')[-1]
                else:
                    file_extension = 'pdf'  # Default to PDF
            
            print(f"✅ Document downloaded successfully")
            
            # Process the downloaded document from memory
            chunks = DocumentProcessor.load_document_from_memory(response.content, file_extension)
            
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
        Download document from URL and process it with custom chunk size (in-memory processing)
        
        Args:
            url: URL of the document to download
            chunk_size: Custom chunk size for processing
            
        Returns:
            List of document chunks or None if failed
        """
        try:
            # Step 1: Download the document directly into memory
            print(f"📥 Downloading document from URL...")
            download_start = time.time()
            # Remove stream=True to download the whole file into memory at once
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            download_end = time.time()
            print(f"   -> Download time: {download_end - download_start:.2f} seconds")

            # Step 2: Process the document directly from memory bytes
            processing_start = time.time()
            
            # Get file extension to pass to the processing function
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1].lower().replace('.', '')
            if not file_extension:
                # Try to get extension from Content-Type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = 'pdf'
                elif 'zip' in content_type:
                    file_extension = 'zip'
                elif 'image' in content_type:
                    file_extension = content_type.split('/')[-1]
                else:
                    file_extension = 'pdf'  # Default to PDF

            # Call the new in-memory processing function
            chunks = DocumentProcessor.load_document_from_memory(response.content, file_extension, chunk_size)
            
            processing_end = time.time()
            print(f"   -> In-memory processing time: {processing_end - processing_start:.2f} seconds")
            
            return chunks
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading document: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing downloaded document: {e}")
            return None
    
    @staticmethod
    def _extract_text_from_page(page):
        """Helper function to extract text from a single PDF page (for parallel processing)."""
        return page.get_text()

    @staticmethod
    def load_document_from_memory(file_bytes: bytes, file_extension: str, chunk_size: int = CHUNK_SIZE) -> list:
        """
        Load document from memory using PyMuPDF with parallel page processing and efficient string handling.
        """
        print(f"🔄 Processing document from memory with PARALLELIZED PyMuPDF...")
        try:
            content = ""
            if file_extension == 'pdf':
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                from concurrent.futures import ThreadPoolExecutor
                # Parallel extraction of page texts
                with ThreadPoolExecutor() as executor:
                    page_texts = list(executor.map(DocumentProcessor._extract_text_from_page, doc))
                doc.close()
                content = "\n\n".join(page_texts)
            elif file_extension == 'txt':
                content = file_bytes.decode('utf-8')
            elif file_extension in ['docx', 'doc']:
                from docx import Document
                import io
                doc_obj = Document(io.BytesIO(file_bytes))
                para_texts = [para.text for para in doc_obj.paragraphs]
                content = "\n\n".join(para_texts)
            elif file_extension == 'eml':
                msg = email.message_from_bytes(file_bytes)
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            content += part.get_payload(decode=True).decode() + "\n\n"
                else:
                    content = msg.get_payload(decode=True).decode()
                content = f"Subject: {msg['subject']}\n\nFrom: {msg['from']}\n\nTo: {msg['to']}\n\n{content}"
            elif file_extension == 'msg':
                # Write bytes to temp file since extract-msg needs a file
                with tempfile.NamedTemporaryFile(suffix='.msg', delete=False) as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name
                
                try:
                    msg = extract_msg.Message(temp_path)
                    content = f"Subject: {msg.subject}\n\nFrom: {msg.sender}\n\nTo: {msg.to}\n\n{msg.body}"
                    msg.close()
                finally:
                    os.unlink(temp_path)  # Clean up temp file
            elif file_extension == 'pptx':
                # Use the same logic as a.py: PPTX → PDF → Images → OpenAI Vision
                import tempfile
                import subprocess
                import base64
                from pdf2image import convert_from_path
                from openai import OpenAI
                
                # Create temporary directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Step 1: Write PPTX bytes to temporary file
                    pptx_temp_path = os.path.join(temp_dir, "temp.pptx")
                    with open(pptx_temp_path, "wb") as f:
                        f.write(file_bytes)
                    
                    # Step 2: Convert PPTX → PDF via LibreOffice
                    try:
                        subprocess.run([
                            "soffice", "--headless",
                            "--convert-to", "pdf",
                            "--outdir", temp_dir,
                            pptx_temp_path
                        ], check=True, capture_output=True)
                        
                        # Step 3: Locate the generated PDF
                        pdf_path = os.path.join(temp_dir, "temp.pdf")
                        if not os.path.exists(pdf_path):
                            raise FileNotFoundError(f"Expected PDF at {pdf_path}")
                        
                        # Step 4: Convert PDF pages to images
                        slides = convert_from_path(pdf_path, dpi=200)
                        
                        # Step 5: Use OpenAI to extract text from each slide image
                        client = OpenAI()
                        
                        for idx, img in enumerate(slides, start=1):
                            # Save image to temp file
                            img_path = os.path.join(temp_dir, f"slide_{idx}.png")
                            img.save(img_path, "PNG")
                            
                            # Encode image to base64
                            with open(img_path, "rb") as f:
                                encoded = base64.b64encode(f.read()).decode("utf-8")
                            data_url = f"data:image/png;base64,{encoded}"
                            
                            # Use OpenAI to extract text
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[{
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": "Please extract the text from the image as is. Do not summarize or paraphrase. If no text present, then summarize the image in 2 lines."},
                                            {"type": "image_url", "image_url": {"url": data_url}},
                                        ],
                                    }],
                                )
                                slide_text = response.choices[0].message.content
                                content += f"--- Slide {idx} ---\n{slide_text}\n\n"
                            except Exception as e:
                                print(f"Warning: Could not process slide {idx} with OpenAI: {e}")
                                content += f"--- Slide {idx} ---\n[Could not extract text]\n\n"
                                
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        # Fallback to basic pptx processing if LibreOffice/pdf2image fails
                        print(f"Warning: PPTX processing failed with OpenAI method: {e}")
                        print("Falling back to basic pptx processing...")
                        from pptx import Presentation
                        prs = Presentation(io.BytesIO(file_bytes))
                        for slide in prs.slides:
                            for shape in slide.shapes:
                                if hasattr(shape, "text"):
                                    content += shape.text + "\n\n"
            elif file_extension == 'xlsx':
                import openpyxl
                import io
                workbook = openpyxl.load_workbook(io.BytesIO(file_bytes))
                for sheet_name in workbook.sheetnames:
                    sheet = workbook[sheet_name]
                    for row in sheet.iter_rows():
                        for cell in row:
                            if cell.value:
                                content += str(cell.value) + " "
                        content += "\n\n"
                    content += "\n\n"
            elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                import pytesseract
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(file_bytes))
                content = pytesseract.image_to_string(image)
            elif file_extension == 'zip':
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                    for filename in z.namelist():
                        with z.open(filename) as f:
                            file_content = f.read()
                            file_ext = os.path.splitext(filename)[1].lower().replace('.', '')
                            content += "\n\n--- New File: " + filename + " ---\n\n"
                            # Recursively process the file
                            chunks = DocumentProcessor.load_document_from_memory(file_content, file_ext, chunk_size)
                            content += "\n".join(chunks)
            else:
                raise ValueError(f"Unsupported file format for memory loading: {file_extension}")
            chunks = DocumentProcessor._smart_chunk_text(content, chunk_size)
            print(f"✅ Processed document and split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"❌ Error loading document from memory: {e}")
            return []
    
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