import os
import torch
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# ─── API Configuration ───────────────────────────────────────────────────────────
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY must be set in your .env file. Get one from https://aistudio.google.com/app/apikey")

genai.configure(api_key=api_key)

# ─── Device Selection ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ─── Model Configuration ─────────────────────────────────────────────────────────
# DIM = 1536  # Gemini embedding model dimension
# MODEL = "gemini-embedding-001"
DIM = 3072  # Gemini embedding model dimension
MODEL = "text-embedding-3-large"


# ─── Document Processing Configuration ───────────────────────────────────────────
CHUNK_OVERLAP = 150  # Overlap between chunks

# ─── Adaptive Configuration ──────────────────────────────────────────────────────
# Base chunk size and Top-K values
BASE_CHUNK_SIZE = 1300
BASE_TOP_K = 5

# Factors for dynamic adjustment
CHUNK_SIZE_PER_PAGE = 50  # Increase chunk size by 50 for each page
TOP_K_PER_10_PAGES = 1  # Increase Top-K by 1 for every 10 pages

# ─── Download Optimization Configuration ─────────────────────────────────────────
DOWNLOAD_PARALLEL_WORKERS = 4  # Number of parallel download workers
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for parallel download
ENABLE_DOCUMENT_CACHE = True  # Enable document caching

DOCUMENT_PATH = "/Users/vipulbhatia/mycode/competetion_projecs/bajaj_finance_document_retrival_contest/sample_documents/test1.txt"

print(f"Using device: {device}") 