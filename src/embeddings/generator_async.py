import torch
import numpy as np
import aiohttp
import asyncio
import json
import os
from dotenv import load_dotenv
from typing import List
import time

load_dotenv()

from ..config import MODEL, DIM, device

class AsyncEmbeddingGenerator:
    """Async embedding generator using aiohttp for better I/O performance."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        
        # Connection pool settings for optimal performance
        self.connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Connections per host
            keepalive_timeout=60,  # Keep connections alive for 60 seconds
            enable_cleanup_closed=True
        )
        
        # Timeout settings
        self.timeout = aiohttp.ClientTimeout(
            total=60,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=30  # Socket read timeout
        )
        
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_embedding_async(self, text: str, semaphore: asyncio.Semaphore = None) -> torch.Tensor:
        """Get a single embedding using async HTTP call."""
        if semaphore:
            async with semaphore:
                return await self._get_embedding_internal(text)
        else:
            return await self._get_embedding_internal(text)
    
    async def _get_embedding_internal(self, text: str) -> torch.Tensor:
        """Internal method to get embedding."""
        try:
            payload = {
                "input": text,
                "model": MODEL
            }
            
            async with self.session.post(
                "https://api.openai.com/v1/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    embedding_values = data['data'][0]['embedding']
                    vec = torch.from_numpy(np.array(embedding_values, dtype=np.float32)).to(device)
                    return vec / vec.norm()
                else:
                    error_text = await response.text()
                    print(f"API Error {response.status}: {error_text}")
                    return torch.zeros(DIM, dtype=torch.float32, device=device)
                    
        except asyncio.TimeoutError:
            print(f"Timeout error getting embedding for text: {text[:50]}...")
            return torch.zeros(DIM, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"Error getting embedding for text: {text[:50]}... Error: {e}")
            return torch.zeros(DIM, dtype=torch.float32, device=device)
    
    async def get_embeddings_batch_async(self, texts: List[str], batch_size: int = 50, max_concurrent: int = 10) -> List[torch.Tensor]:
        """
        Get embeddings for multiple texts using batch API calls with concurrency control.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch API call (OpenAI supports up to 2048)
            max_concurrent: Maximum concurrent requests
        """
        if not texts:
            return []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Split texts into batches for API calls
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        print(f"      Processing {len(texts)} texts in {len(batches)} batches (max {max_concurrent} concurrent)")
        
        # Process batches concurrently
        tasks = []
        for batch_idx, batch in enumerate(batches):
            task = self._process_batch_async(batch, batch_idx, len(batches), semaphore)
            tasks.append(task)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        all_embeddings = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                print(f"      ❌ Batch failed: {batch_result}")
                # Add zero embeddings for failed batch
                all_embeddings.extend([torch.zeros(DIM, dtype=torch.float32, device=device)] * batch_size)
            else:
                all_embeddings.extend(batch_result)
        
        # Trim to original length (in case last batch was smaller)
        return all_embeddings[:len(texts)]
    
    async def _process_batch_async(self, batch: List[str], batch_idx: int, total_batches: int, semaphore: asyncio.Semaphore) -> List[torch.Tensor]:
        """Process a single batch of texts."""
        async with semaphore:
            try:
                # Use OpenAI's batch embedding API
                payload = {
                    "input": batch,
                    "model": MODEL
                }
                
                start_time = time.perf_counter()
                async with self.session.post(
                    "https://api.openai.com/v1/embeddings",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embeddings = []
                        
                        for embedding_data in data['data']:
                            embedding_values = embedding_data['embedding']
                            vec = torch.from_numpy(np.array(embedding_values, dtype=np.float32)).to(device)
                            embeddings.append(vec / vec.norm())
                        
                        end_time = time.perf_counter()
                        print(f"      ✅ Batch {batch_idx + 1}/{total_batches} completed in {end_time - start_time:.2f}s")
                        return embeddings
                    else:
                        error_text = await response.text()
                        print(f"      ❌ Batch {batch_idx + 1} API Error {response.status}: {error_text}")
                        return [torch.zeros(DIM, dtype=torch.float32, device=device)] * len(batch)
                        
            except Exception as e:
                print(f"      ❌ Batch {batch_idx + 1} Error: {e}")
                return [torch.zeros(DIM, dtype=torch.float32, device=device)] * len(batch)
    
    async def get_embeddings_for_chunks_async(self, chunks: List[str]) -> torch.Tensor:
        """
        Get embeddings for document chunks with optimal batching.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Stacked tensor of embeddings
        """
        print(f"🔄 Generating embeddings for {len(chunks)} chunks using async processing...")
        
        start_time = time.perf_counter()
        embeddings = await self.get_embeddings_batch_async(
            chunks, 
            batch_size=50,  # Optimal batch size for OpenAI API
            max_concurrent=8  # Reasonable concurrency limit
        )
        end_time = time.perf_counter()
        
        print(f"✅ Generated {len(embeddings)} embeddings in {end_time - start_time:.2f}s")
        
        # Filter out any None values and stack
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        if not valid_embeddings:
            raise ValueError("No valid embeddings generated")
        
        return torch.stack(valid_embeddings)


# Async wrapper functions for backward compatibility
async def get_embedding_async(text: str) -> torch.Tensor:
    """Get a single embedding asynchronously."""
    async with AsyncEmbeddingGenerator() as generator:
        return await generator.get_embedding_async(text)

async def get_embeddings_in_parallel_async(texts: List[str], batch_size: int = 50, max_concurrent: int = 10) -> torch.Tensor:
    """Get embeddings for multiple texts asynchronously."""
    async with AsyncEmbeddingGenerator() as generator:
        embeddings = await generator.get_embeddings_batch_async(texts, batch_size, max_concurrent)
        return torch.stack(embeddings)

# Sync wrapper for compatibility with existing code
def get_embeddings_in_parallel_sync(texts: List[str], batch_size: int = 50, max_concurrent: int = 10) -> torch.Tensor:
    """Sync wrapper around async embedding generation."""
    return asyncio.run(get_embeddings_in_parallel_async(texts, batch_size, max_concurrent))


# Example usage and performance comparison
async def performance_test():
    """Test performance of async vs sync embedding generation."""
    test_texts = [
        f"This is test text number {i} for embedding generation performance testing."
        for i in range(100)
    ]
    
    print("🧪 Testing async embedding generation performance...")
    
    # Test async approach
    start_time = time.perf_counter()
    async with AsyncEmbeddingGenerator() as generator:
        embeddings = await generator.get_embeddings_batch_async(test_texts, batch_size=20, max_concurrent=5)
    async_time = time.perf_counter() - start_time
    
    print(f"📊 Async Results:")
    print(f"   • Time: {async_time:.2f}s")
    print(f"   • Embeddings: {len(embeddings)}")
    print(f"   • Rate: {len(embeddings) / async_time:.1f} embeddings/sec")

if __name__ == "__main__":
    asyncio.run(performance_test())
