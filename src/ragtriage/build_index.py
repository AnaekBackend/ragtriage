#!/usr/bin/env python3
"""
Build vector index from Freshdesk Solutions.json
Extracts articles, chunks them with token awareness, and builds Chroma index with metadata.
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import json
import os
import re
import hashlib
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Load .env file if present
try:
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key, value)
except FileNotFoundError:
    pass

# Simple HTML tag stripper
def strip_html(html: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def count_tokens(text: str) -> int:
    """Rough token count (1 token ≈ 4 chars for English)."""
    return len(text) // 4

def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by token count."""
    if not text:
        return []
    
    # Rough character limit based on tokens
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        
        # Try to break at sentence boundary
        chunk_text = text[start:end]
        last_period = max(
            chunk_text.rfind('.'),
            chunk_text.rfind('!'),
            chunk_text.rfind('?'),
            chunk_text.rfind('\n')
        )
        
        if last_period > max_chars * 0.5:  # Only use if we're not cutting too short
            end = start + last_period + 1
        else:
            # Try word boundary
            last_space = chunk_text.rfind(' ')
            if last_space > max_chars * 0.8:
                end = start + last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap_chars
    
    return chunks

def extract_articles_from_json(filepath: str) -> List[Dict]:
    """Extract all articles from Solutions.json with platform metadata."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    articles = []
    
    platform_map = {
        'Slack': 'slack',
        'Microsoft Teams': 'msteams',
        'General': 'general',
        'Hangouts': 'hangouts',
        'AB Web': 'web',
        'Support': 'general',
        'Gusto': 'general',
        'Jira': 'general'
    }
    
    for category_item in data:
        category = category_item.get('category', {})
        category_name = category.get('name', 'Unknown')
        platform = platform_map.get(category_name, 'general')
        
        for folder in category.get('all_folders', []):
            folder_name = folder.get('name', '')
            
            for article in folder.get('articles', []):
                # Use desc_un_html if available (cleaner), else strip description
                content = article.get('desc_un_html', '')
                if not content:
                    content = strip_html(article.get('description', ''))
                
                if not content or not article.get('title'):
                    continue
                
                # Build article URL
                article_id = article.get('id')
                article_url = f"https://attendancebot.freshdesk.com/support/solutions/articles/{article_id}"
                
                articles.append({
                    'id': str(article_id),
                    'title': article.get('title', ''),
                    'content': content,
                    'platform': platform,
                    'category': category_name,
                    'folder': folder_name,
                    'url': article_url,
                    'updated_at': article.get('updated_at', '')
                })
    
    return articles

def create_chunks(articles: List[Dict], max_tokens: int = 512, overlap: int = 50) -> List[Dict]:
    """Create overlapping chunks from articles with metadata."""
    chunks = []
    
    for article in articles:
        # Prepend title to content for context
        full_text = f"{article['title']}\n\n{article['content']}"
        
        # If entire article fits in one chunk, keep it together
        if count_tokens(full_text) <= max_tokens:
            chunk_id = hashlib.md5(f"{article['id']}-0".encode()).hexdigest()
            chunks.append({
                'id': chunk_id,
                'text': full_text,
                'article_id': article['id'],
                'title': article['title'],
                'platform': article['platform'],
                'url': article['url'],
                'chunk_index': 0,
                'total_chunks': 1
            })
        else:
            # Split into chunks
            text_chunks = chunk_text(article['content'], max_tokens - 50, overlap)  # Reserve 50 tokens for title
            
            for i, text_chunk in enumerate(text_chunks):
                chunk_text_with_title = f"{article['title']}\n\n{text_chunk}"
                chunk_id = hashlib.md5(f"{article['id']}-{i}".encode()).hexdigest()
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text_with_title,
                    'article_id': article['id'],
                    'title': article['title'],
                    'platform': article['platform'],
                    'url': article['url'],
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                })
    
    return chunks

def build_index(chunks: List[Dict], persist_dir: str = "chroma_db"):
    """Build and persist Chroma vector index."""
    # Initialize Chroma client
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="attendancebot_docs",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize OpenAI client for embeddings
    openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    
    # Process in batches
    batch_size = 100
    total = len(chunks)
    
    print(f"Building index with {total} chunks...")
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        
        # Generate embeddings
        texts = [c['text'] for c in batch]
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [e.embedding for e in response.data]
        
        # Prepare for Chroma
        ids = [c['id'] for c in batch]
        documents = [c['text'] for c in batch]
        metadatas = [{
            'article_id': c['article_id'],
            'title': c['title'],
            'platform': c['platform'],
            'url': c['url'],
            'chunk_index': c['chunk_index'],
            'total_chunks': c['total_chunks']
        } for c in batch]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"  Progress: {min(i+batch_size, total)}/{total}")
    
    print(f"\nIndex built successfully at {persist_dir}/")
    print(f"Total chunks: {total}")

if __name__ == "__main__":
    import sys
    
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Extract articles
    print("Extracting articles from Solutions.json...")
    articles = extract_articles_from_json('Solutions.json')
    print(f"Found {len(articles)} articles")
    
    # Count by platform
    platform_counts = {}
    for a in articles:
        p = a['platform']
        platform_counts[p] = platform_counts.get(p, 0) + 1
    print(f"By platform: {platform_counts}")
    
    # Create chunks
    print("\nCreating chunks...")
    chunks = create_chunks(articles, max_tokens=512, overlap=50)
    print(f"Created {len(chunks)} chunks")
    
    # Build index
    print("\nBuilding vector index...")
    build_index(chunks)
    
    print("\nDone! Run 'python app.py' to start the server.")
