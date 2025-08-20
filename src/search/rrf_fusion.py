"""
Reciprocal Rank Fusion (RRF) for combining FAISS and BM25 search results
"""

from typing import List, Tuple, Dict, Any, Set
import concurrent.futures
import time


class RRFFusion:
    """Reciprocal Rank Fusion for combining multiple search result rankings"""
    
    def __init__(self, k_const: int = 60):
        """
        Initialize RRF fusion
        
        Args:
            k_const: RRF constant parameter (typically 60)
        """
        self.k_const = k_const
    
    def fuse_rankings(
        self,
        bm25_results: List[Dict[str, Any]],   # BM25 search results
        faiss_results: List[Dict[str, Any]],  # FAISS search results
        top_n: int = 20,
        normalize_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fuse BM25 and FAISS rankings using Reciprocal Rank Fusion
        
        Args:
            bm25_results: List of BM25 search results with 'chunk_index', 'score', 'text'
            faiss_results: List of FAISS search results with 'chunk_index', 'score', 'text'
            top_n: Number of top results to return
            normalize_scores: Whether to normalize original scores
            
        Returns:
            List of fused results with RRF scores and ranking info
        """
        # Convert to (chunk_id, score) tuples and build rank maps
        bm25_ranked = [(r['chunk_index'], r['score']) for r in bm25_results]
        faiss_ranked = [(r['chunk_index'], r['score']) for r in faiss_results]
        
        # Build rank maps (1-based ranking)
        bm25_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_ranked, start=1)}
        faiss_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(faiss_ranked, start=1)}
        
        # Create lookup maps for original results
        bm25_lookup = {r['chunk_index']: r for r in bm25_results}
        faiss_lookup = {r['chunk_index']: r for r in faiss_results}
        
        # Union of candidate chunk indices
        candidate_chunks = set(bm25_rank.keys()) | set(faiss_rank.keys())
        
        fused_results = []
        
        for chunk_id in candidate_chunks:
            rrf_score = 0.0
            fusion_info = {
                'chunk_index': chunk_id,
                'bm25_rank': None,
                'faiss_rank': None,
                'bm25_score': None,
                'faiss_score': None,
                'sources': []
            }
            
            # Add BM25 contribution
            if chunk_id in bm25_rank:
                bm25_contribution = 1.0 / (self.k_const + bm25_rank[chunk_id])
                rrf_score += bm25_contribution
                fusion_info['bm25_rank'] = bm25_rank[chunk_id]
                fusion_info['bm25_score'] = bm25_lookup[chunk_id]['score']
                fusion_info['sources'].append('bm25')
            
            # Add FAISS contribution
            if chunk_id in faiss_rank:
                faiss_contribution = 1.0 / (self.k_const + faiss_rank[chunk_id])
                rrf_score += faiss_contribution
                fusion_info['faiss_rank'] = faiss_rank[chunk_id]
                fusion_info['faiss_score'] = faiss_lookup[chunk_id]['score']
                fusion_info['sources'].append('faiss')
            
            # Get text content (prefer FAISS expanded text, fallback to BM25)
            if chunk_id in faiss_lookup:
                text_content = faiss_lookup[chunk_id]['text']
            elif chunk_id in bm25_lookup:
                text_content = bm25_lookup[chunk_id]['text']
            else:
                text_content = ""
            
            # Create fused result
            fused_result = {
                'chunk_index': chunk_id,
                'rrf_score': rrf_score,
                'text': text_content,
                'fusion_info': fusion_info
            }
            
            fused_results.append(fused_result)
        
        # Sort by RRF score (descending) and take top_n
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return fused_results[:top_n]
    
    def fuse_with_weights(
        self,
        bm25_results: List[Dict[str, Any]],
        faiss_results: List[Dict[str, Any]],
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5,
        top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fuse rankings with custom weights for each method
        
        Args:
            bm25_results: BM25 search results
            faiss_results: FAISS search results
            bm25_weight: Weight for BM25 contributions (0.0 to 1.0)
            faiss_weight: Weight for FAISS contributions (0.0 to 1.0)
            top_n: Number of top results to return
            
        Returns:
            List of weighted fused results
        """
        # Normalize weights
        total_weight = bm25_weight + faiss_weight
        if total_weight > 0:
            bm25_weight = bm25_weight / total_weight
            faiss_weight = faiss_weight / total_weight
        
        # Convert to ranked tuples
        bm25_ranked = [(r['chunk_index'], r['score']) for r in bm25_results]
        faiss_ranked = [(r['chunk_index'], r['score']) for r in faiss_results]
        
        # Build rank maps
        bm25_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_ranked, start=1)}
        faiss_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(faiss_ranked, start=1)}
        
        # Create lookup maps
        bm25_lookup = {r['chunk_index']: r for r in bm25_results}
        faiss_lookup = {r['chunk_index']: r for r in faiss_results}
        
        # Union of candidates
        candidate_chunks = set(bm25_rank.keys()) | set(faiss_rank.keys())
        
        weighted_results = []
        
        for chunk_id in candidate_chunks:
            weighted_score = 0.0
            fusion_info = {
                'chunk_index': chunk_id,
                'bm25_rank': None,
                'faiss_rank': None,
                'bm25_score': None,
                'faiss_score': None,
                'bm25_weight': bm25_weight,
                'faiss_weight': faiss_weight,
                'sources': []
            }
            
            # Weighted BM25 contribution
            if chunk_id in bm25_rank:
                bm25_contribution = bm25_weight * (1.0 / (self.k_const + bm25_rank[chunk_id]))
                weighted_score += bm25_contribution
                fusion_info['bm25_rank'] = bm25_rank[chunk_id]
                fusion_info['bm25_score'] = bm25_lookup[chunk_id]['score']
                fusion_info['sources'].append('bm25')
            
            # Weighted FAISS contribution
            if chunk_id in faiss_rank:
                faiss_contribution = faiss_weight * (1.0 / (self.k_const + faiss_rank[chunk_id]))
                weighted_score += faiss_contribution
                fusion_info['faiss_rank'] = faiss_rank[chunk_id]
                fusion_info['faiss_score'] = faiss_lookup[chunk_id]['score']
                fusion_info['sources'].append('faiss')
            
            # Get text content
            if chunk_id in faiss_lookup:
                text_content = faiss_lookup[chunk_id]['text']
            elif chunk_id in bm25_lookup:
                text_content = bm25_lookup[chunk_id]['text']
            else:
                text_content = ""
            
            weighted_result = {
                'chunk_index': chunk_id,
                'weighted_rrf_score': weighted_score,
                'text': text_content,
                'fusion_info': fusion_info
            }
            
            weighted_results.append(weighted_result)
        
        # Sort by weighted score and return top_n
        weighted_results.sort(key=lambda x: x['weighted_rrf_score'], reverse=True)
        
        return weighted_results[:top_n]


class MMRPostProcessor:
    """Maximal Marginal Relevance post-processing for diversity"""
    
    def __init__(self, diversity_lambda: float = 0.5):
        """
        Initialize MMR processor
        
        Args:
            diversity_lambda: Balance between relevance and diversity (0.0 = pure relevance, 1.0 = pure diversity)
        """
        self.diversity_lambda = diversity_lambda
    
    def apply_mmr(
        self,
        fused_results: List[Dict[str, Any]],
        max_results: int = 10,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Apply MMR to reduce redundancy in results
        
        Args:
            fused_results: RRF fused results
            max_results: Maximum number of results to return
            similarity_threshold: Threshold for considering documents similar
            
        Returns:
            MMR-filtered results with diversity
        """
        if not fused_results:
            return []
        
        selected = []
        remaining = fused_results.copy()
        
        # Always select the top result first
        selected.append(remaining.pop(0))
        
        while len(selected) < max_results and remaining:
            best_candidate = None
            best_mmr_score = -float('inf')
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # Relevance score (normalized RRF score)
                relevance = candidate.get('rrf_score', candidate.get('weighted_rrf_score', 0))
                
                # Diversity score (1 - max similarity to selected documents)
                max_similarity = 0.0
                for selected_doc in selected:
                    similarity = self._compute_text_similarity(candidate['text'], selected_doc['text'])
                    max_similarity = max(max_similarity, similarity)
                
                diversity = 1.0 - max_similarity
                
                # MMR score
                mmr_score = (1 - self.diversity_lambda) * relevance + self.diversity_lambda * diversity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return selected
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple text similarity (Jaccard similarity of words)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class HybridSearchEngine:
    """Combines FAISS and BM25 search with RRF fusion"""
    
    def __init__(self, k_const: int = 60, diversity_lambda: float = 0.5):
        """
        Initialize hybrid search engine
        
        Args:
            k_const: RRF constant parameter
            diversity_lambda: MMR diversity parameter
        """
        self.rrf_fusion = RRFFusion(k_const=k_const)
        self.mmr_processor = MMRPostProcessor(diversity_lambda=diversity_lambda)
    
    def hybrid_search(
        self,
        faiss_engine,
        bm25_engine,
        query: str,
        top_k_per_engine: int = 20,
        final_top_k: int = 10,
        use_mmr: bool = True,
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining FAISS and BM25 with RRF fusion
        
        Args:
            faiss_engine: FAISS search engine instance
            bm25_engine: BM25 search engine instance
            query: Search query
            top_k_per_engine: Results to get from each engine
            final_top_k: Final number of results after fusion
            use_mmr: Whether to apply MMR for diversity
            bm25_weight: Weight for BM25 results
            faiss_weight: Weight for FAISS results
            
        Returns:
            Dictionary with hybrid search results and metadata
        """
        search_start = time.time()
        
        # Run both searches in parallel
        def run_faiss():
            return faiss_engine.find_relevant_chunks(query, top_k_per_engine)
        
        def run_bm25():
            return bm25_engine.search(query, top_k_per_engine)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            faiss_future = executor.submit(run_faiss)
            bm25_future = executor.submit(run_bm25)
            
            faiss_results = faiss_future.result()
            bm25_results = bm25_future.result()
        
        parallel_time = time.time() - search_start
        
        # Fuse results using RRF
        fusion_start = time.time()
        if bm25_weight != 0.5 or faiss_weight != 0.5:
            # Use weighted fusion
            fused_results = self.rrf_fusion.fuse_with_weights(
                bm25_results, faiss_results, bm25_weight, faiss_weight, final_top_k * 2
            )
        else:
            # Use standard RRF
            fused_results = self.rrf_fusion.fuse_rankings(
                bm25_results, faiss_results, final_top_k * 2
            )
        
        fusion_time = time.time() - fusion_start
        
        # Apply MMR for diversity if requested
        final_results = fused_results
        mmr_time = 0.0
        
        if use_mmr and len(fused_results) > final_top_k:
            mmr_start = time.time()
            final_results = self.mmr_processor.apply_mmr(fused_results, final_top_k)
            mmr_time = time.time() - mmr_start
        else:
            final_results = fused_results[:final_top_k]
        
        total_time = time.time() - search_start
        
        return {
            'results': final_results,
            'metadata': {
                'query': query,
                'faiss_results_count': len(faiss_results),
                'bm25_results_count': len(bm25_results),
                'fused_results_count': len(fused_results),
                'final_results_count': len(final_results),
                'timing': {
                    'parallel_search': parallel_time,
                    'fusion': fusion_time,
                    'mmr': mmr_time,
                    'total': total_time
                },
                'parameters': {
                    'top_k_per_engine': top_k_per_engine,
                    'final_top_k': final_top_k,
                    'use_mmr': use_mmr,
                    'bm25_weight': bm25_weight,
                    'faiss_weight': faiss_weight,
                    'k_const': self.rrf_fusion.k_const,
                    'diversity_lambda': self.mmr_processor.diversity_lambda
                }
            }
        }
