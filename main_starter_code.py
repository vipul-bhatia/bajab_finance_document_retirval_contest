import os
import sqlite3
import torch
import time
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ─── Configuration ───────────────────────────────────────────────────────────────
load_dotenv()

# Configure Google AI with API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY must be set in your .env file. Get one from https://aistudio.google.com/app/apikey")

genai.configure(api_key=api_key)

# ─── Device selection ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ─── Configuration ────────────────────────────────────────────────────────────────
DIM = 3072  # Gemini embedding model dimension
MODEL = "gemini-embedding-001"
TOP_K = 3  # Return top 3 results
DOCUMENT_PATH = "/Users/vipulbhatia/mycode/competetion_projecs/bajaj_finance_document_retrival_contest/sample_documents/test1.txt"  # Path to your text document

# ─── 1. Load and process document ────────────────────────────────────────────────
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

# ─── 2. Database setup ───────────────────────────────────────────────────────────
def get_db_path(document_name: str) -> str:
    """Get SQLite database path for a document"""
    return f"document_embeddings_{document_name}.db"

def ensure_table_exists(document_name: str):
    """Creates the document_embeddings table if it doesn't exist."""
    db_path = get_db_path(document_name)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
          chunk_index INTEGER PRIMARY KEY,
          chunk_text TEXT,
          embedding BLOB
        );
    """)
    conn.commit()
    conn.close()
    print(f"✅ Database table ready: {db_path}")

# ─── 3. Embedding helper using Gemini ────────────────────────────────────────────
def get_embedding(text: str) -> torch.Tensor:
    """Get embedding using Gemini model"""
    try:
        response = genai.embed_content(
            model=MODEL,
            content=text,
            task_type="retrieval_document"
        )
        
        # Extract embedding values
        embedding_values = response['embedding']
        
        # Convert to numpy array then to torch tensor
        arr = np.array(embedding_values, dtype=np.float32)
        vec = torch.from_numpy(arr).to(device)
        
        # Normalize the vector
        return vec / vec.norm()
        
    except Exception as e:
        print(f"Error getting embedding for text: {text[:50]}... Error: {e}")
        # Return zero vector as fallback
        return torch.zeros(DIM, dtype=torch.float32, device=device)

# ─── 4. Store embeddings in SQLite ───────────────────────────────────────────────
def store_embeddings(chunks: list, document_name: str):
    """Generate and store embeddings for all document chunks"""
    ensure_table_exists(document_name)
    
    db_path = get_db_path(document_name)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Clear existing data
    cur.execute("DELETE FROM document_embeddings;")
    
    sql = "INSERT INTO document_embeddings(chunk_index, chunk_text, embedding) VALUES(?,?,?);"
    
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1}/{len(chunks)}")
        
        # Get embedding for the chunk
        vec = get_embedding(chunk)
        blob = vec.cpu().numpy().tobytes()
        
        # Store in database
        cur.execute(sql, (idx, chunk, blob))
        
        if (idx + 1) % 10 == 0:
            print(f"Stored embeddings for {idx + 1}/{len(chunks)} chunks")
    
    conn.commit()
    conn.close()
    print(f"✅ All embeddings stored successfully in {db_path}!")

# ─── 5. Load embeddings into torch tensors ──────────────────────────────────────
def load_embeddings(M: int, document_name: str) -> tuple[torch.Tensor, list]:
    """Load precomputed embeddings into Torch tensors on device"""
    
    db_path = get_db_path(document_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get count of embeddings
    cursor.execute("SELECT COUNT(*) FROM document_embeddings")
    count = cursor.fetchone()[0]
    
    if count == 0:
        conn.close()
        print(f"❌ No embeddings found in {db_path}")
        return torch.zeros((M, DIM), dtype=torch.float32, device=device), []
    
    # Load embeddings and texts
    E = torch.zeros((count, DIM), dtype=torch.float32, device=device)
    chunks = []
    
    cursor.execute("SELECT chunk_index, chunk_text, embedding FROM document_embeddings ORDER BY chunk_index ASC")
    
    for idx, chunk_text, blob in cursor.fetchall():
        vec = torch.frombuffer(blob, dtype=torch.float32).to(device)
        E[idx] = vec
        chunks.append(chunk_text)
    
    conn.close()
    print(f"✅ Loaded {count} embeddings from {db_path}")
    return E, chunks

# ─── 6. Initialize embeddings ────────────────────────────────────────────────────
E = None
document_chunks = []

def init(document_path: str = DOCUMENT_PATH, document_name: str = None):
    """Initialize embeddings - generate if needed, then load"""
    global E, document_chunks
    
    # Generate document name from file path if not provided
    if not document_name:
        import os
        document_name = os.path.splitext(os.path.basename(document_path))[0]
        # Clean document name for use in file name (alphanumeric and underscore only)
        document_name = ''.join(c for c in document_name if c.isalnum() or c == '_')
    
    # Load document chunks
    chunks = load_document(document_path)
    if not chunks:
        print("No document chunks found. Please provide a valid document.")
        return False, document_name
    
    M = len(chunks)
    
    # Check if embeddings exist and match the number of chunks
    db_path = get_db_path(document_name)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='document_embeddings'")
    table_exists = cur.fetchone()[0] > 0
    
    embeddings_exist = False
    if table_exists:
        cur.execute("SELECT COUNT(*) FROM document_embeddings")
        count = cur.fetchone()[0]
        
        if count == M:
            print(f"✅ Embeddings database already exists, loading from {db_path}...")
            E, document_chunks = load_embeddings(M, document_name)
            embeddings_exist = True
    
    conn.close()
    
    if not embeddings_exist:
        print(f"🔄 Generating new embeddings for {document_name}...")
        store_embeddings(chunks, document_name)
        E, document_chunks = load_embeddings(M, document_name)
    
    return True, document_name

# ─── 7. Query function using PyTorch on device ───────────────────────────────────
def find_relevant_chunks(query: str, top_k: int = TOP_K):
    """Find relevant document chunks"""
    if E is None or not document_chunks:
        raise RuntimeError("Embeddings not initialized. Call init() first.")
    
    q_vec = get_embedding(query)  # lives on device
    
    # Calculate similarities
    similarities = E @ q_vec  # (num_chunks,) on device
    
    # Get top results
    scores, indices = torch.topk(similarities, k=min(top_k, len(document_chunks)))
    
    results = []
    for score, i in zip(scores, indices):
        results.append({
            "chunk_index": i.item(),
            "text": document_chunks[i.item()],
            "score": score.item()
        })
    
    return results

# ─── 8. Utility functions ───────────────────────────────────────────────────────
def list_documents():
    """List all available document databases"""
    import glob
    import os
    
    db_files = glob.glob("document_embeddings_*.db")
    if not db_files:
        print("No document databases found.")
        return []
    
    documents = []
    for db_file in db_files:
        # Extract document name from filename
        doc_name = db_file.replace("document_embeddings_", "").replace(".db", "")
        
        # Get chunk count
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_embeddings")
        count = cur.fetchone()[0]
        conn.close()
        
        documents.append((doc_name, count, db_file))
    
    print("Available document databases:")
    for i, (doc_name, count, db_file) in enumerate(documents, 1):
        print(f"  {i}. {doc_name} ({count} chunks) - {db_file}")
    
    return documents

def delete_document(document_name: str):
    """Delete a document database"""
    db_path = get_db_path(document_name)
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Deleted document database: {db_path}")
    else:
        print(f"❌ Document database not found: {db_path}")

# ─── 9. REPL Loop ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📚 Document Embeddings with SQLite")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Load/Create document embeddings")
        print("2. List existing documents")
        print("3. Delete document database")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\n--- Load/Create Document Embeddings ---")
            
            # You can specify a different document path here
            document_file = input("Enter document path (or press Enter for 'test1.txt'): ").strip()
            if not document_file:
                document_file = DOCUMENT_PATH
            
            # You can also specify a custom document name for the database
            doc_name = input("Enter document name for database (or press Enter for auto-generated): ").strip()
            if not doc_name:
                doc_name = None
            
            try:
                success, document_name = init(document_file, doc_name)
                if not success:
                    print("Failed to initialize. Please try again.")
                    continue
            except Exception as e:
                print(f"Error during initialization: {e}")
                continue
            
            print(f"\n✅ Document Search Ready! Loaded {len(document_chunks)} chunks for document '{document_name}'.")
            print("Enter your query (blank to go back to menu):")
            
            while True:
                query = input("\n> ").strip()
                if not query:
                    break
                
                start = time.time()
                try:
                    results = find_relevant_chunks(query)
                    elapsed = time.time() - start
                    
                    print(f"\nFound {len(results)} relevant chunks (took {elapsed:.3f}s):\n")
                    
                    for i, result in enumerate(results, 1):
                        print(f"{i}. Score: {result['score']:.4f}")
                        print(f"   Text: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
                        print(f"   [Chunk {result['chunk_index']}]")
                        print()
                except Exception as e:
                    print(f"Error during search: {e}")
        
        elif choice == "2":
            print("\n--- List Existing Documents ---")
            list_documents()
        
        elif choice == "3":
            print("\n--- Delete Document Database ---")
            documents = list_documents()
            if documents:
                try:
                    doc_num = int(input("Enter document number to delete: ")) - 1
                    if 0 <= doc_num < len(documents):
                        doc_name = documents[doc_num][0]
                        confirm = input(f"Are you sure you want to delete '{doc_name}'? (y/N): ").strip().lower()
                        if confirm == 'y':
                            delete_document(doc_name)
                        else:
                            print("Deletion cancelled.")
                    else:
                        print("Invalid document number.")
                except ValueError:
                    print("Please enter a valid number.")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-4.") 