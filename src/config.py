import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── API Configuration ───────────────────────────────────────────────────────────
# OpenAI API Key for GPT-4o
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in your .env file. Get one from https://platform.openai.com/api-keys")

# Google API Key (kept for potential future use with embeddings)
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY must be set in your .env file. Get one from https://aistudio.google.com/app/apikey")

# ─── Device Selection ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ─── Model Configuration ─────────────────────────────────────────────────────────
DIM = 3072  # OpenAI embedding model dimension
MODEL = "text-embedding-3-large"
TOP_K = 6 # Return top 5 results

# ─── Document Processing Configuration ───────────────────────────────────────────
CHUNK_SIZE = 1400  # Characters per chunk (2500 = ~2 chunks per page)
CHUNK_OVERLAP = 100  # Overlap between chunks

# ─── Embeddings Storage Configuration ────────────────────────────────────────────
# All FAISS indices will be stored under this directory
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "document_embeddings")

# ─── Download Optimization Configuration ─────────────────────────────────────────
DOWNLOAD_PARALLEL_WORKERS = 4  # Number of parallel download workers
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for parallel download
ENABLE_DOCUMENT_CACHE = True  # Enable document caching

DOCUMENT_PATH = "/Users/vipulbhatia/mycode/competetion_projecs/bajaj_finance_document_retrival_contest/sample_documents/test1.txt"

print(f"Using device: {device}") 