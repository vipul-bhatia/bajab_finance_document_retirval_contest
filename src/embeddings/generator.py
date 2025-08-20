import torch
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import concurrent.futures
import time
import math
import random
from collections import deque
import threading
from typing import List

load_dotenv()

from ..config import MODEL, DIM, device

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")



class EmbeddingGenerator:
    """Handles embedding generation using Gemini model with parallel batch processing."""
    
    # Aggregate rate limit configuration
    TPM_LIMIT = 350_000  # tokens per minute (aggregate)
    TOKENS_PER_CHAR = 1.0 / 4.0  # heuristic ~4 chars per token
    MIN_TOKENS_PER_REQUEST = 16
    MAX_WORKERS_CAP = 25  # hard cap on threads to avoid oversubscription
    
    _token_lock = threading.Lock()
    _token_window: deque = deque()  # (timestamp, tokens)
    
    @staticmethod
    def get_embedding(text: str) -> torch.Tensor:
        """Get a single embedding using the OpenAI model."""
        try:
            # Estimate tokens and acquire allowance under TPM limit
            needed_tokens = max(
                EmbeddingGenerator.MIN_TOKENS_PER_REQUEST,
                int(math.ceil(len(text) * EmbeddingGenerator.TOKENS_PER_CHAR))
            )
            EmbeddingGenerator._acquire_tokens(needed_tokens, context="embedding")

            def _call():
                return openai.embeddings.create(input=text, model=MODEL)

            response = EmbeddingGenerator._retry_with_backoff(_call, desc="embeddings.create")
            embedding_values = response.data[0].embedding
            vec = torch.from_numpy(np.array(embedding_values, dtype=np.float32)).to(device)
            return vec / vec.norm()
        except Exception as e:
            print(f"Error getting embedding for text: {text[:50]}... Error: {e}")
            return torch.zeros(DIM, dtype=torch.float32, device=device)

    @staticmethod
    def get_embeddings_for_batch(batch: List[str]) -> List[torch.Tensor]:
        """Get embeddings for a batch of texts."""
        workers = min(len(batch), EmbeddingGenerator.MAX_WORKERS_CAP)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            embeddings = list(executor.map(EmbeddingGenerator.get_embedding, batch))
        return embeddings

    @staticmethod
    def get_embeddings_in_parallel(texts: List[str], batch_size: int = 50, max_workers: int = 10) -> torch.Tensor:
        """Get embeddings for a list of texts in parallel with batching."""
        all_embeddings = [None] * len(texts)
        
        # Create batches of a moderate size so limiter can smooth bursts
        optimal_batch = min(batch_size, EmbeddingGenerator.MAX_WORKERS_CAP * 2)
        batches = [texts[i:i + optimal_batch] for i in range(0, len(texts), optimal_batch)]
        print(f"      Creating {len(batches)} batches of size {optimal_batch}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, EmbeddingGenerator.MAX_WORKERS_CAP)) as executor:
            future_to_batch_index = {
                executor.submit(EmbeddingGenerator.get_embeddings_for_batch, batch): i
                for i, batch in enumerate(batches)
            }

            for future in concurrent.futures.as_completed(future_to_batch_index):
                batch_index = future_to_batch_index[future]
                try:
                    batch_embeddings = future.result()
                    start_index = batch_index * batch_size
                    for i, embedding in enumerate(batch_embeddings):
                        all_embeddings[start_index + i] = embedding
                    print(f"      ✅ Processed batch {batch_index + 1}/{len(batches)}")
                except Exception as e:
                    print(f"      ❌ Error processing batch {batch_index + 1}: {e}")

        # Filter out any None values from failed batches
        valid_embeddings = [emb for emb in all_embeddings if emb is not None]
        return torch.stack(valid_embeddings) 

    # ─── Internal helpers: token limiter and retry/backoff ─────────────────────────
    @staticmethod
    def _acquire_tokens(needed_tokens: int, context: str = "") -> None:
        while True:
            with EmbeddingGenerator._token_lock:
                now = time.time()
                # drop entries older than 60s
                while EmbeddingGenerator._token_window and (now - EmbeddingGenerator._token_window[0][0]) > 60.0:
                    EmbeddingGenerator._token_window.popleft()
                used = sum(t for _, t in EmbeddingGenerator._token_window)
                if used + needed_tokens <= EmbeddingGenerator.TPM_LIMIT:
                    EmbeddingGenerator._token_window.append((now, needed_tokens))
                    return
                oldest_ts = EmbeddingGenerator._token_window[0][0] if EmbeddingGenerator._token_window else now
                wait_seconds = max(0.05, (oldest_ts + 60.0) - now)
            print(f"⏳ Reached TPM window, waiting {wait_seconds:.2f}s (need {needed_tokens} tokens){' - ' + context if context else ''}")
            time.sleep(min(wait_seconds, 1.0))

    @staticmethod
    def _retry_with_backoff(func, *, desc: str = "", max_retries: int = 3):
        delay = 1.0
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return func()
            except Exception as e:
                msg = str(e)
                is_rate = ("429" in msg) or ("rate limit" in msg.lower())
                if is_rate:
                    print(f"⚠️ Rate limit hit for {desc or 'request'} (attempt {attempt}/{max_retries}). Retrying in {delay:.1f}s...")
                else:
                    print(f"⚠️ Error in {desc or 'request'} (attempt {attempt}/{max_retries}): {msg}. Retrying in {delay:.1f}s...")
                time.sleep(delay + random.uniform(0, 0.3))
                delay = min(delay * 2, 8.0)
                last_exc = e
        raise last_exc if last_exc else RuntimeError("Unknown error during retry")