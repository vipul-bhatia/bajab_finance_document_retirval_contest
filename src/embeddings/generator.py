import torch
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

from ..config import MODEL, DIM, device

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")



class EmbeddingGenerator:
    """Handles embedding generation using Gemini model"""
    
    @staticmethod
    def get_embedding(text: str) -> torch.Tensor:
        """Get embedding using Gemini model"""
        try:
            # response = genai.embed_content(
            #     model=MODEL,
            #     content=text,
            #     task_type="retrieval_document"
            # )

            response = openai.embeddings.create(
                input=text,
                model=MODEL
            )

            # Extract embedding values
            embedding_values = response.data[0].embedding
            
            # Convert to numpy array then to torch tensor
            arr = np.array(embedding_values, dtype=np.float32)
            vec = torch.from_numpy(arr).to(device)
            
            # Normalize the vector
            return vec / vec.norm()
            
        except Exception as e:
            print(f"Error getting embedding for text: {text[:50]}... Error: {e}")
            # Return zero vector as fallback
            return torch.zeros(DIM, dtype=torch.float32, device=device) 