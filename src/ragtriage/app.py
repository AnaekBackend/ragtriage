#!/usr/bin/env python3
"""
Modernized AttendanceBot RAG API
- Direct OpenAI client (no LangChain)
- Single vector DB with metadata filtering
- Structured outputs for classification
- Simple, maintainable codebase
"""
import os
import json
from typing import List, Dict, Optional
from flask import Flask, jsonify, request
from openai import OpenAI
import chromadb
from chromadb.config import Settings

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

# Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable required")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(
    path="chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_collection("attendancebot_docs")

app = Flask(__name__)

def get_query_embedding(text: str) -> List[float]:
    """Generate embedding for query text."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_docs(
    query: str, 
    platform: str = 'slack',
    n_results: int = 5
) -> List[Dict]:
    """Search vector DB for relevant documents."""
    query_embedding = get_query_embedding(query)
    
    # Build where clause for platform filtering
    # Platform 'slack' should also include 'general' articles
    if platform == 'slack':
        where_clause = {"$or": [{"platform": "slack"}, {"platform": "general"}, {"platform": "hangouts"}]}
    elif platform == 'msteams':
        where_clause = {"$or": [{"platform": "msteams"}, {"platform": "general"}]}
    else:
        where_clause = {"platform": platform}
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_clause
    )
    
    documents = []
    for i in range(len(results['ids'][0])):
        documents.append({
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    
    return documents

def classify_query(query: str) -> Dict:
    """
    Classify if query is specific enough to answer.
    Returns: {'continue': bool, 'reason': str}
    """
    system_prompt = """You are a query classifier for AttendanceBot support.
Determine if the user's question is SPECIFIC enough to answer with documentation.

SPECIFIC questions:
- Ask about particular features, settings, or workflows
- Mention concrete actions like "upload certificate", "approve timesheet", "set up shift"
- Include error scenarios or specific situations

GENERAL questions (should be rejected):
- "what do I do", "how can I use this", "how to set up my account"
- "help me", "what is this", "how does it work" (without specifics)
- Greetings, chitchat, or vague requests

Respond with JSON: {"continue": true/false, "reason": "brief explanation"}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    result = json.loads(response.choices[0].message.content)
    return result

def generate_answer(
    query: str, 
    documents: List[Dict],
    platform: str = 'slack'
) -> Dict:
    """Generate answer from retrieved documents."""
    
    # Build context from documents
    context_parts = []
    sources = []
    
    for i, doc in enumerate(documents[:3], 1):  # Use top 3
        metadata = doc['metadata']
        context_parts.append(f"[{i}] {doc['text']}")
        sources.append({
            'title': metadata['title'],
            'url': metadata['url'],
            'relevance': 1 - doc['distance']  # Convert distance to relevance score
        })
    
    context = "\n\n".join(context_parts)
    
    # Build URL reference list for the prompt
    url_list = "\n".join([f"[{i+1}] {s['url']}" for i, s in enumerate(sources)])
    
    system_prompt = f"""You are AttendanceBot's helpful support assistant. Answer user questions based on the provided documentation.

CRITICAL RULES:
1. Answer completely using the provided context
2. Be concise - use less than 150 tokens for the main answer
3. Use numbered steps when explaining procedures
4. At the end, on a new line, output ONLY the raw URL - no markdown, no link text, just the URL itself
5. ONLY use URLs from this list - DO NOT make up or hallucinate URLs:
{url_list}

6. If the user's question is in a language other than English, ask them to chat in English only
7. If you cannot find the answer in the context, say you don't have enough information and suggest opening a support ticket by typing 'ticket'
8. For broad questions like "tell me about feature X", ask for specific details
9. For questions unrelated to AttendanceBot, state your limitations

Platform context: {platform}

Documentation context:
{context}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=250
    )
    
    answer = response.choices[0].message.content
    
    return {
        'answer': answer,
        'sources': sources
    }

@app.route("/")
def ping():
    return "pong"

@app.route("/qa/", methods=['POST'])
def qa():
    """Two-stage QA: classify then answer."""
    data = request.get_json() or request.form
    user_query = data.get('query_string', '')
    platform = data.get('platform', 'slack')
    
    if not user_query:
        return jsonify({"ok": False, "error": "Missing query_string"}), 400
    
    # Normalize platform
    if platform != 'msteams':
        platform = 'slack'
    
    # Stage 1: Classify
    classification = classify_query(user_query)
    
    if not classification.get('continue', False):
        return jsonify({
            "ok": False, 
            "reason": classification.get('reason', 'Query too general')
        })
    
    # Stage 2: Retrieve and answer
    documents = search_docs(user_query, platform)
    
    if not documents:
        return jsonify({
            "ok": False,
            "reason": "No relevant documentation found"
        })
    
    result = generate_answer(user_query, documents, platform)
    
    return jsonify({
        "ok": True,
        "answer": result['answer'],
        "sources": result['sources']
    })

@app.route("/assist/", methods=['POST'])
def assist():
    """Direct answer without classification step."""
    data = request.get_json() or request.form
    user_query = data.get('query_string', '')
    platform = data.get('platform', 'slack')
    
    if not user_query:
        return jsonify({"ok": False, "error": "Missing query_string"}), 400
    
    # Normalize platform
    if platform != 'msteams':
        platform = 'slack'
    
    # Retrieve and answer
    documents = search_docs(user_query, platform, n_results=5)
    
    if not documents:
        return jsonify({
            "ok": False,
            "reason": "No relevant documentation found"
        })
    
    result = generate_answer(user_query, documents, platform)
    
    return jsonify({
        "ok": True,
        "answer": result['answer'],
        "sources": result['sources']
    })

@app.route("/health/")
def health():
    """Health check endpoint."""
    count = collection.count()
    return jsonify({
        "status": "ok",
        "indexed_documents": count
    })

if __name__ == "__main__":
    # Verify collection exists and has data
    try:
        count = collection.count()
        print(f"Loaded index with {count} documents")
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Run 'python build_index.py' first to build the index")
        exit(1)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
