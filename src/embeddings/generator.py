import torch
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import concurrent.futures
from typing import List

load_dotenv()

from ..config import MODEL, DIM, device

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")



class EmbeddingGenerator:
    """Handles embedding generation using Gemini model with parallel batch processing."""
    
    @staticmethod
    def get_embedding(text: str) -> torch.Tensor:
        """Get a single embedding using the OpenAI model."""
        try:
            response = openai.embeddings.create(input=text, model=MODEL)
            embedding_values = response.data[0].embedding
            vec = torch.from_numpy(np.array(embedding_values, dtype=np.float32)).to(device)
            return vec / vec.norm()
        except Exception as e:
            print(f"Error getting embedding for text: {text[:50]}... Error: {e}")
            return torch.zeros(DIM, dtype=torch.float32, device=device)

    @staticmethod
    def get_embeddings_for_batch(batch: List[str]) -> List[torch.Tensor]:
        """Get embeddings for a batch of texts."""
        # This method can be optimized further if the API supports batch inputs.
        # For now, we process them in parallel within the batch.
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            embeddings = list(executor.map(EmbeddingGenerator.get_embedding, batch))
        return embeddings

    @staticmethod
    def get_embeddings_in_parallel(texts: List[str], batch_size: int = 50, max_workers: int = 10) -> torch.Tensor:
        """Get embeddings for a list of texts in parallel with batching."""
        all_embeddings = [None] * len(texts)
        
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        print(f"      Creating {len(batches)} batches of size {batch_size}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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