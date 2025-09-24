# fortune_setup.py
"""
Initial setup script to load Chinese fortune telling books into the RAG system
and configure specialized prompts for fortune telling.
"""

import os
import json
from datetime import datetime
from api.db_utils import get_db_connection, insert_document_record
from api.chroma_utils import index_document_to_chroma

# Configuration
FORTUNE_BOOKS_DIR = "fortune_books"
if not os.path.exists(FORTUNE_BOOKS_DIR):
    os.makedirs(FORTUNE_BOOKS_DIR)

# List of fortune telling books to include
FORTUNE_BOOKS = [
    {
        "filename": "san_ming_tong_hui.pdf",
        "title": "三命通会",
        "description": "A comprehensive text on BaZi (八字) and destiny analysis"
    },
    {
        "filename": "di_tian_sui.pdf",
        "title": "滴天髓",
        "description": "Essential teachings on BaZi interpretation"
    },
    {
        "filename": "zi_ping_zhen_quan.pdf", 
        "title": "子平真诠",
        "description": "Detailed explanation of destiny calculation methods"
    }
]

# Create metadata for the books
BOOKS_METADATA = {
    "fortune_books": FORTUNE_BOOKS,
    "last_updated": datetime.now().isoformat()
}

# Save metadata
with open(os.path.join(FORTUNE_BOOKS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(BOOKS_METADATA, f, ensure_ascii=False, indent=4)

def load_fortune_books():
    """Load the fortune telling books into the RAG system."""
    print("Loading fortune telling books into the RAG system...")
    
    for book in FORTUNE_BOOKS:
        filename = book["filename"]
        file_path = os.path.join(FORTUNE_BOOKS_DIR, filename)
        
        # Check if the file exists (you'll need to obtain these files)
        if not os.path.exists(file_path):
            print(f"Warning: {filename} does not exist. Please add it to the {FORTUNE_BOOKS_DIR} directory.")
            continue
        
        # Add the book to the document store
        file_id = insert_document_record(filename)
        
        # Index the book in Chroma
        success = index_document_to_chroma(file_path, file_id)
        
        if success:
            print(f"Successfully loaded {book['title']} ({filename}) into the RAG system.")
        else:
            print(f"Failed to load {book['title']} ({filename}).")
    
    print("Fortune telling books loading complete.")

if __name__ == "__main__":
    load_fortune_books()