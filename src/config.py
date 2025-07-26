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
TOP_K = 3  # Return top 3 results

# ─── Document Processing Configuration ───────────────────────────────────────────
CHUNK_SIZE = 2500  # Characters per chunk (2500 = ~2 chunks per page)
CHUNK_OVERLAP = 100  # Overlap between chunks

DOCUMENT_PATH = "/Users/vipulbhatia/mycode/competetion_projecs/bajaj_finance_document_retrival_contest/sample_documents/test1.txt"

print(f"Using device: {device}") 