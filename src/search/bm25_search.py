import os
import pickle
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import math
import re
from ..config import TOP_K


class BM25SearchEngine:
    """BM25 search implementation with parallel processing and disk storage capabilities"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 search engine
        
        Args:
            k1: Controls term frequency normalization (default 1.2)
            b: Controls length normalization (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.document_chunks = []
        self.document_name = None
        
        # BM25 specific data structures
        self.term_frequencies = []  # TF for each document
        self.document_frequencies = {}  # DF for each term
        self.inverse_document_frequencies = {}  # IDF for each term
        self.document_lengths = []  # Length of each document
        self.average_document_length = 0.0
        self.vocabulary = set()
        
        # For faster lookup during search
        self.term_to_docs = defaultdict(list)  # Maps term to list of doc indices
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing, lowercasing, and removing stopwords
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove punctuation and split into tokens
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Simple stopword removal (can be enhanced with NLTK or spaCy)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their'
        }
        
        return [token for token in tokens if token not in stopwords and len(token) > 2]
    
    def _compute_term_frequencies(self, documents: List[str]) -> List[Dict[str, int]]:
        """
        Compute term frequencies for all documents in parallel
        
        Args:
            documents: List of document texts
            
        Returns:
            List of term frequency dictionaries for each document
        """
        print(f"      📊 Computing term frequencies for {len(documents)} documents...")
        
        def process_document(doc_text: str) -> Dict[str, int]:
            tokens = self._preprocess_text(doc_text)
            return Counter(tokens)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            term_freq_results = list(executor.map(process_document, documents))
        
        return term_freq_results
    
    def _build_vocabulary_and_df(self, term_frequencies: List[Dict[str, int]]) -> None:
        """
        Build vocabulary and compute document frequencies
        
        Args:
            term_frequencies: List of term frequency dictionaries
        """
        print(f"      📚 Building vocabulary and computing document frequencies...")
        
        self.vocabulary = set()
        self.document_frequencies = defaultdict(int)
        
        for doc_tf in term_frequencies:
            for term in doc_tf.keys():
                self.vocabulary.add(term)
                self.document_frequencies[term] += 1
    
    def _compute_idf(self, total_documents: int) -> None:
        """
        Compute inverse document frequencies for all terms
        
        Args:
            total_documents: Total number of documents in the collection
        """
        print(f"      🧮 Computing IDF values for {len(self.vocabulary)} terms...")
        
        for term in self.vocabulary:
            df = self.document_frequencies[term]
            # IDF = log((N - df + 0.5) / (df + 0.5))
            self.inverse_document_frequencies[term] = math.log(
                (total_documents - df + 0.5) / (df + 0.5)
            )
    
    def _build_term_to_docs_mapping(self) -> None:
        """Build mapping from terms to document indices for faster search"""
        print(f"      🗺️  Building term-to-documents mapping...")
        
        self.term_to_docs = defaultdict(list)
        for doc_idx, doc_tf in enumerate(self.term_frequencies):
            for term in doc_tf.keys():
                self.term_to_docs[term].append(doc_idx)
    
    def build_index(self, document_chunks: List[str], document_name: str) -> None:
        """
        Build BM25 index from document chunks
        
        Args:
            document_chunks: List of document text chunks
            document_name: Name identifier for the document
        """
        print(f"      🔄 Building BM25 index for {len(document_chunks)} chunks...")
        index_start = time.time()
        
        self.document_chunks = document_chunks
        self.document_name = document_name
        
        # Step 1: Compute term frequencies in parallel
        tf_start = time.time()
        self.term_frequencies = self._compute_term_frequencies(document_chunks)
        tf_time = time.time() - tf_start
        print(f"      ✅ Term frequency computation: {tf_time:.2f}s")
        
        # Step 2: Compute document lengths
        self.document_lengths = [sum(tf.values()) for tf in self.term_frequencies]
        self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)
        
        # Step 3: Build vocabulary and document frequencies
        vocab_start = time.time()
        self._build_vocabulary_and_df(self.term_frequencies)
        vocab_time = time.time() - vocab_start
        print(f"      ✅ Vocabulary building: {vocab_time:.2f}s")
        
        # Step 4: Compute IDF values
        idf_start = time.time()
        self._compute_idf(len(document_chunks))
        idf_time = time.time() - idf_start
        print(f"      ✅ IDF computation: {idf_time:.2f}s")
        
        # Step 5: Build term-to-docs mapping for faster search
        mapping_start = time.time()
        self._build_term_to_docs_mapping()
        mapping_time = time.time() - mapping_start
        print(f"      ✅ Term mapping: {mapping_time:.2f}s")
        
        total_time = time.time() - index_start
        print(f"      🏁 Total BM25 index building: {total_time:.2f}s")
        print(f"      📊 Index stats: {len(self.vocabulary)} unique terms, avg doc length: {self.average_document_length:.1f}")
    
    def _compute_bm25_score(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a document given query terms
        
        Args:
            query_terms: List of preprocessed query terms
            doc_idx: Document index to score
            
        Returns:
            BM25 score for the document
        """
        score = 0.0
        doc_tf = self.term_frequencies[doc_idx]
        doc_length = self.document_lengths[doc_idx]
        
        for term in query_terms:
            if term in doc_tf:
                tf = doc_tf[term]
                idf = self.inverse_document_frequencies.get(term, 0)
                
                # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d| / avgdl)))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.average_document_length)
                )
                
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using BM25 scoring
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores and text
        """
        if not self.document_chunks:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        
        search_start = time.time()
        
        # Preprocess query
        query_terms = self._preprocess_text(query)
        if not query_terms:
            return []
        
        # Find candidate documents (documents that contain at least one query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.term_to_docs:
                candidate_docs.update(self.term_to_docs[term])
        
        if not candidate_docs:
            return []
        
        # Score candidate documents in parallel
        def score_document(doc_idx: int) -> Tuple[int, float]:
            score = self._compute_bm25_score(query_terms, doc_idx)
            return doc_idx, score
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            doc_scores = list(executor.map(score_document, candidate_docs))
        
        # Sort by score (descending) and take top_k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = doc_scores[:top_k]
        
        # Build result objects
        results = []
        for doc_idx, score in top_results:
            # Expand context similar to FAISS implementation
            expanded_text = self._build_expanded_text(doc_idx, window=1)
            results.append({
                "chunk_index": doc_idx,
                "text": expanded_text,
                "score": score
            })
        
        search_time = time.time() - search_start
        print(f"      🔍 BM25 search completed in {search_time:.3f}s, found {len(results)} results")
        
        return results
    
    def _build_expanded_text(self, center_index: int, window: int = 1) -> str:
        """Return text that includes the chunk at center_index and its neighbors within the window."""
        if not self.document_chunks:
            return ""
        start_index = max(0, center_index - window)
        end_index = min(len(self.document_chunks) - 1, center_index + window)
        # Join with blank lines to preserve readability between chunks
        return "\n\n".join(self.document_chunks[start_index:end_index + 1])
    
    def save_index(self, index_path: str) -> None:
        """
        Save BM25 index to disk for future loading
        
        Args:
            index_path: Path to save the index file
        """
        print(f"      💾 Saving BM25 index to {index_path}...")
        save_start = time.time()
        
        # Prepare index data for serialization
        index_data = {
            'document_chunks': self.document_chunks,
            'document_name': self.document_name,
            'term_frequencies': self.term_frequencies,
            'document_frequencies': dict(self.document_frequencies),
            'inverse_document_frequencies': self.inverse_document_frequencies,
            'document_lengths': self.document_lengths,
            'average_document_length': self.average_document_length,
            'vocabulary': list(self.vocabulary),
            'term_to_docs': dict(self.term_to_docs),
            'k1': self.k1,
            'b': self.b
        }
        
        # Save using pickle for efficient serialization
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        save_time = time.time() - save_start
        print(f"      ✅ BM25 index saved in {save_time:.2f}s")
    
    def load_index(self, index_path: str) -> bool:
        """
        Load BM25 index from disk
        
        Args:
            index_path: Path to the saved index file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(index_path):
            print(f"      ❌ BM25 index file not found: {index_path}")
            return False
        
        print(f"      📂 Loading BM25 index from {index_path}...")
        load_start = time.time()
        
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Restore index data
            self.document_chunks = index_data['document_chunks']
            self.document_name = index_data['document_name']
            self.term_frequencies = index_data['term_frequencies']
            self.document_frequencies = defaultdict(int, index_data['document_frequencies'])
            self.inverse_document_frequencies = index_data['inverse_document_frequencies']
            self.document_lengths = index_data['document_lengths']
            self.average_document_length = index_data['average_document_length']
            self.vocabulary = set(index_data['vocabulary'])
            self.term_to_docs = defaultdict(list, index_data['term_to_docs'])
            self.k1 = index_data.get('k1', 1.2)
            self.b = index_data.get('b', 0.75)
            
            load_time = time.time() - load_start
            print(f"      ✅ BM25 index loaded in {load_time:.2f}s")
            print(f"      📊 Loaded index: {len(self.vocabulary)} terms, {len(self.document_chunks)} chunks")
            
            return True
            
        except Exception as e:
            print(f"      ❌ Error loading BM25 index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current BM25 index
        
        Returns:
            Dictionary with index statistics
        """
        if not self.document_chunks:
            return {"status": "No index loaded"}
        
        return {
            "document_name": self.document_name,
            "total_chunks": len(self.document_chunks),
            "vocabulary_size": len(self.vocabulary),
            "average_document_length": self.average_document_length,
            "k1_parameter": self.k1,
            "b_parameter": self.b
        }
