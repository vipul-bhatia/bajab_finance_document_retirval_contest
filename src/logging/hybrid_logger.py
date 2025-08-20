"""
Hybrid Search Logger for detailed tracking of FAISS + BM25 + RRF pipeline
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any
import logging


class HybridSearchLogger:
    """Comprehensive logger for hybrid search operations"""
    
    def __init__(self, log_dir: str = "hybrid_search_logs"):
        """
        Initialize hybrid search logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create separate log files for different aspects
        self.setup_loggers()
        
        # Current session data
        self.current_session = None
        self.session_start_time = None
    
    def setup_loggers(self):
        """Setup different loggers for various aspects of hybrid search"""
        
        # Main hybrid search logger
        self.hybrid_logger = self._create_file_logger(
            'hybrid_search',
            os.path.join(self.log_dir, 'hybrid_search_detailed.log')
        )
        
        # Search results comparison logger
        self.comparison_logger = self._create_file_logger(
            'search_comparison',
            os.path.join(self.log_dir, 'search_results_comparison.log')
        )
        
        # Performance metrics logger
        self.performance_logger = self._create_file_logger(
            'performance_metrics',
            os.path.join(self.log_dir, 'performance_metrics.log')
        )
        
        # RRF fusion logger
        self.rrf_logger = self._create_file_logger(
            'rrf_fusion',
            os.path.join(self.log_dir, 'rrf_fusion_details.log')
        )
    
    def _create_file_logger(self, name: str, filepath: str) -> logging.Logger:
        """Create a file logger with specific format"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler
        handler = logging.FileHandler(filepath, mode='a', encoding='utf-8')
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.propagate = False
        
        return logger
    
    def start_session(self, document_name: str, total_questions: int):
        """Start a new hybrid search session"""
        self.session_start_time = time.time()
        self.current_session = {
            'session_id': f"{document_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'document_name': document_name,
            'total_questions': total_questions,
            'start_time': datetime.now().isoformat(),
            'questions': []
        }
        
        session_info = (
            f"=== NEW HYBRID SEARCH SESSION ===\n"
            f"Session ID: {self.current_session['session_id']}\n"
            f"Document: {document_name}\n"
            f"Total Questions: {total_questions}\n"
            f"Timestamp: {self.current_session['start_time']}\n"
            f"=================================="
        )
        
        self.hybrid_logger.info(session_info)
        self.comparison_logger.info(session_info)
        self.performance_logger.info(session_info)
        self.rrf_logger.info(session_info)
    
    def log_document_info(self, doc_info: Dict[str, Any]):
        """Log document information and search engine status"""
        doc_summary = (
            f"DOCUMENT INFORMATION:\n"
            f"  • Chunks: {doc_info.get('chunk_count', 'Unknown')}\n"
            f"  • FAISS Loaded: {doc_info.get('faiss_loaded', False)}\n"
            f"  • BM25 Loaded: {doc_info.get('bm25_loaded', False)}\n"
        )
        
        if doc_info.get('bm25_stats'):
            bm25_stats = doc_info['bm25_stats']
            doc_summary += (
                f"  • BM25 Vocabulary: {bm25_stats.get('vocabulary_size', 'N/A')} terms\n"
                f"  • BM25 Avg Doc Length: {bm25_stats.get('average_document_length', 'N/A'):.1f}\n"
                f"  • BM25 Parameters: k1={bm25_stats.get('k1_parameter', 'N/A')}, b={bm25_stats.get('b_parameter', 'N/A')}"
            )
        
        self.hybrid_logger.info(doc_summary)
    
    def log_question_start(self, question_index: int, question: str):
        """Log the start of processing a question"""
        question_header = (
            f"\n--- QUESTION {question_index + 1} ---\n"
            f"Query: '{question}'\n"
            f"Started: {datetime.now().strftime('%H:%M:%S')}"
        )
        
        self.hybrid_logger.info(question_header)
        self.comparison_logger.info(question_header)
        
        # Initialize question data
        question_data = {
            'index': question_index,
            'question': question,
            'start_time': time.time(),
            'faiss_results': [],
            'bm25_results': [],
            'fused_results': [],
            'final_answer': '',
            'timing': {}
        }
        
        if self.current_session:
            self.current_session['questions'].append(question_data)
        
        return question_data
    
    def log_faiss_results(self, question_index: int, faiss_results: List[Dict[str, Any]], search_time: float):
        """Log FAISS search results in detail"""
        faiss_summary = (
            f"FAISS SEARCH RESULTS (Semantic):\n"
            f"  • Search Time: {search_time:.3f}s\n"
            f"  • Results Found: {len(faiss_results)}\n"
        )
        
        for i, result in enumerate(faiss_results, 1):
            faiss_summary += (
                f"  {i}. Chunk {result['chunk_index']}: Score={result['score']:.4f}\n"
                f"     Text: {result['text'][:80]}...\n"
            )
        
        self.hybrid_logger.info(faiss_summary)
        self.comparison_logger.info(f"FAISS: {len(faiss_results)} results, best score: {faiss_results[0]['score']:.4f}" if faiss_results else "FAISS: No results")
        
        # Store in session data
        if self.current_session and question_index < len(self.current_session['questions']):
            self.current_session['questions'][question_index]['faiss_results'] = faiss_results
            self.current_session['questions'][question_index]['timing']['faiss_search'] = search_time
    
    def log_bm25_results(self, question_index: int, bm25_results: List[Dict[str, Any]], search_time: float):
        """Log BM25 search results in detail"""
        bm25_summary = (
            f"BM25 SEARCH RESULTS (Keyword):\n"
            f"  • Search Time: {search_time:.3f}s\n"
            f"  • Results Found: {len(bm25_results)}\n"
        )
        
        for i, result in enumerate(bm25_results, 1):
            bm25_summary += (
                f"  {i}. Chunk {result['chunk_index']}: Score={result['score']:.4f}\n"
                f"     Text: {result['text'][:80]}...\n"
            )
        
        self.hybrid_logger.info(bm25_summary)
        self.comparison_logger.info(f"BM25: {len(bm25_results)} results, best score: {bm25_results[0]['score']:.4f}" if bm25_results else "BM25: No results")
        
        # Store in session data
        if self.current_session and question_index < len(self.current_session['questions']):
            self.current_session['questions'][question_index]['bm25_results'] = bm25_results
            self.current_session['questions'][question_index]['timing']['bm25_search'] = search_time
    
    def log_rrf_fusion(self, question_index: int, fusion_metadata: Dict[str, Any], fusion_time: float):
        """Log RRF fusion process in detail"""
        rrf_summary = (
            f"RRF FUSION PROCESS:\n"
            f"  • Fusion Time: {fusion_time:.3f}s\n"
            f"  • FAISS Input: {fusion_metadata.get('faiss_results_count', 0)} chunks\n"
            f"  • BM25 Input: {fusion_metadata.get('bm25_results_count', 0)} chunks\n"
            f"  • Fused Output: {fusion_metadata.get('final_results_count', 0)} chunks\n"
            f"  • MMR Applied: {fusion_metadata.get('used_mmr', False)}\n"
        )
        
        if 'parameters' in fusion_metadata:
            params = fusion_metadata['parameters']
            rrf_summary += (
                f"  • RRF Parameters:\n"
                f"    - k_constant: {params.get('k_const', 'N/A')}\n"
                f"    - BM25 weight: {params.get('bm25_weight', 'N/A')}\n"
                f"    - FAISS weight: {params.get('faiss_weight', 'N/A')}\n"
                f"    - Diversity lambda: {params.get('diversity_lambda', 'N/A')}\n"
            )
        
        self.rrf_logger.info(rrf_summary)
        
        # Store timing data
        if self.current_session and question_index < len(self.current_session['questions']):
            self.current_session['questions'][question_index]['timing']['rrf_fusion'] = fusion_time
    
    def log_final_results(self, question_index: int, final_results: List[Dict[str, Any]], answer: str):
        """Log final fused results that go to answer generation"""
        final_summary = (
            f"FINAL HYBRID RESULTS (After RRF + MMR):\n"
            f"  • Total Results: {len(final_results)}\n"
        )
        
        for i, result in enumerate(final_results, 1):
            fusion_info = result.get('fusion_info', {})
            sources = fusion_info.get('sources', [])
            rrf_score = result.get('rrf_score', result.get('weighted_rrf_score', 0))
            
            final_summary += (
                f"  {i}. Chunk {result['chunk_index']}: RRF={rrf_score:.4f} (Sources: {', '.join(sources)})\n"
                f"     BM25 rank: {fusion_info.get('bm25_rank', 'N/A')}, FAISS rank: {fusion_info.get('faiss_rank', 'N/A')}\n"
                f"     Text: {result['text'][:80]}...\n"
            )
        
        final_summary += f"\nGENERATED ANSWER ({len(answer)} chars):\n{answer[:200]}..."
        
        self.hybrid_logger.info(final_summary)
        
        # Create detailed comparison
        self.log_search_comparison(question_index, final_results)
        
        # Store in session data
        if self.current_session and question_index < len(self.current_session['questions']):
            self.current_session['questions'][question_index]['fused_results'] = final_results
            self.current_session['questions'][question_index]['final_answer'] = answer
    
    def log_search_comparison(self, question_index: int, final_results: List[Dict[str, Any]]):
        """Log detailed comparison between search engines"""
        if not final_results:
            return
        
        # Analyze source distribution
        faiss_only = sum(1 for r in final_results if r.get('fusion_info', {}).get('sources') == ['faiss'])
        bm25_only = sum(1 for r in final_results if r.get('fusion_info', {}).get('sources') == ['bm25'])
        both_sources = sum(1 for r in final_results if len(r.get('fusion_info', {}).get('sources', [])) == 2)
        
        comparison = (
            f"SEARCH ENGINE COMPARISON:\n"
            f"  • FAISS-only results: {faiss_only}\n"
            f"  • BM25-only results: {bm25_only}\n"
            f"  • Both engines: {both_sources}\n"
            f"  • Coverage: {((faiss_only + bm25_only + both_sources) / len(final_results) * 100):.1f}%\n"
        )
        
        # Top result analysis
        top_result = final_results[0]
        top_sources = top_result.get('fusion_info', {}).get('sources', [])
        comparison += f"  • Top result from: {', '.join(top_sources)}\n"
        
        self.comparison_logger.info(comparison)
    
    def log_performance_metrics(self, question_index: int, total_time: float):
        """Log performance metrics for the question"""
        if not (self.current_session and question_index < len(self.current_session['questions'])):
            return
        
        question_data = self.current_session['questions'][question_index]
        timing = question_data['timing']
        
        metrics = (
            f"PERFORMANCE METRICS - Question {question_index + 1}:\n"
            f"  • FAISS Search: {timing.get('faiss_search', 0):.3f}s\n"
            f"  • BM25 Search: {timing.get('bm25_search', 0):.3f}s\n"
            f"  • RRF Fusion: {timing.get('rrf_fusion', 0):.3f}s\n"
            f"  • Total Time: {total_time:.3f}s\n"
            f"  • FAISS Results: {len(question_data['faiss_results'])}\n"
            f"  • BM25 Results: {len(question_data['bm25_results'])}\n"
            f"  • Final Results: {len(question_data['fused_results'])}\n"
        )
        
        self.performance_logger.info(metrics)
        
        # Store total time
        question_data['timing']['total'] = total_time
    
    def end_session(self):
        """End the current session and create summary"""
        if not self.current_session:
            return
        
        session_end_time = time.time()
        total_session_time = session_end_time - self.session_start_time
        
        # Create session summary
        successful_questions = sum(1 for q in self.current_session['questions'] if q.get('final_answer'))
        avg_time_per_question = total_session_time / len(self.current_session['questions']) if self.current_session['questions'] else 0
        
        summary = (
            f"\n=== SESSION SUMMARY ===\n"
            f"Session ID: {self.current_session['session_id']}\n"
            f"Total Time: {total_session_time:.2f}s\n"
            f"Questions Processed: {len(self.current_session['questions'])}\n"
            f"Successful Answers: {successful_questions}\n"
            f"Average Time/Question: {avg_time_per_question:.2f}s\n"
            f"Document: {self.current_session['document_name']}\n"
            f"=======================\n"
        )
        
        # Log to all loggers
        self.hybrid_logger.info(summary)
        self.comparison_logger.info(summary)
        self.performance_logger.info(summary)
        self.rrf_logger.info(summary)
        
        # Save session data as JSON
        self.save_session_json()
        
        # Reset session
        self.current_session = None
        self.session_start_time = None
    
    def save_session_json(self):
        """Save detailed session data as JSON file"""
        if not self.current_session:
            return
        
        json_filename = f"{self.current_session['session_id']}_detailed.json"
        json_filepath = os.path.join(self.log_dir, json_filename)
        
        # Add end time
        self.current_session['end_time'] = datetime.now().isoformat()
        self.current_session['total_duration'] = time.time() - self.session_start_time
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Detailed session data saved to: {json_filepath}")
    
    def get_log_files_info(self):
        """Get information about created log files"""
        if not os.path.exists(self.log_dir):
            return "No log directory found"
        
        log_files = []
        for file in os.listdir(self.log_dir):
            if file.endswith(('.log', '.json')):
                filepath = os.path.join(self.log_dir, file)
                size = os.path.getsize(filepath)
                log_files.append(f"  • {file} ({size} bytes)")
        
        return f"Log files in {self.log_dir}:\n" + "\n".join(log_files)


# Global logger instance
_hybrid_logger = None

def get_hybrid_logger() -> HybridSearchLogger:
    """Get the global hybrid search logger instance"""
    global _hybrid_logger
    if _hybrid_logger is None:
        _hybrid_logger = HybridSearchLogger()
    return _hybrid_logger
