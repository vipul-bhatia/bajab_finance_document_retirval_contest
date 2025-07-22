#!/usr/bin/env python3
"""
Document Retrieval System - Main Interface
Uses modular architecture for document processing, embedding generation, and search.
"""

from src.document_manager import DocumentManager
from src.database import DatabaseManager
from src.config import DOCUMENT_PATH


def main():
    """Main REPL interface for the document retrieval system"""
    print("📚 Document Embeddings with SQLite (Modular Version)")
    print("=" * 55)
    
    document_manager = DocumentManager()
    
    while True:
        print("\nOptions:")
        print("1. Load/Create document embeddings")
        print("2. List existing documents")
        print("3. Delete document database")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\n--- Load/Create Document Embeddings ---")
            
            # Get document file path
            document_file = input("Enter document path (or press Enter for default): ").strip()
            if not document_file:
                document_file = DOCUMENT_PATH
            
            # Get custom document name for database
            doc_name = input("Enter document name for database (or press Enter for auto-generated): ").strip()
            if not doc_name:
                doc_name = None
            
            try:
                success, document_name = document_manager.initialize_document(document_file, doc_name)
                if not success:
                    print("Failed to initialize. Please try again.")
                    continue
            except Exception as e:
                print(f"Error during initialization: {e}")
                continue
            
            # Get document info
            doc_info = document_manager.get_document_info()
            print(f"\n✅ Document Search Ready! Loaded {doc_info['chunk_count']} chunks for document '{doc_info['name']}'.")
            print("\n🎯 Enter your query to get the best answer:")
            print("The system will use Gemini to decompose your query, search in parallel, and return the single most relevant result.")
            print("\nEnter your query (blank to go back to menu):")
            
            # Search loop
            while True:
                query = input("\n> ").strip()
                if not query:
                    break
                
                try:
                    # Always get the best answer
                    document_manager.get_best_answer(query)
                except Exception as e:
                    print(f"Error during search: {e}")
        
        elif choice == "2":
            print("\n--- List Existing Documents ---")
            DatabaseManager.list_documents()
        
        elif choice == "3":
            print("\n--- Delete Document Database ---")
            documents = DatabaseManager.list_documents()
            if documents:
                try:
                    doc_num = int(input("Enter document number to delete: ")) - 1
                    if 0 <= doc_num < len(documents):
                        doc_name = documents[doc_num][0]
                        confirm = input(f"Are you sure you want to delete '{doc_name}'? (y/N): ").strip().lower()
                        if confirm == 'y':
                            DatabaseManager.delete_document(doc_name)
                        else:
                            print("Deletion cancelled.")
                    else:
                        print("Invalid document number.")
                except ValueError:
                    print("Please enter a valid number.")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-4.")


if __name__ == "__main__":
    main() 