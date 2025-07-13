import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify, request
import sys

# Initialize Flask app
app = Flask(__name__)

# Configuration
CONFIG = {
    "index_name": "starter-index",
    "dimension": 384,  # all-MiniLM-L6-v2 embedding size
    "metric": "cosine",
    "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
    "embedding_model": "all-MiniLM-L6-v2"
}

# Global variables for initialized components
pc = None
model = None
index = None

def initialize_components():
    """Initialize Pinecone and embedding model"""
    global pc, model
    try:
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        pc = Pinecone(api_key=api_key)
        model = SentenceTransformer(CONFIG["embedding_model"])
        return pc, model
        
    except Exception as e:
        app.logger.error(f"Initialization failed: {str(e)}")
        sys.exit(1)

def setup_index():
    """Create or connect to Pinecone index"""
    global index
    try:
        if CONFIG["index_name"] in pc.list_indexes().names():
            index_info = pc.describe_index(CONFIG["index_name"])
            if index_info.dimension != CONFIG["dimension"]:
                pc.delete_index(CONFIG["index_name"])
                app.logger.info("Deleted old index with incorrect dimension")
                
        if CONFIG["index_name"] not in pc.list_indexes().names():
            pc.create_index(
                name=CONFIG["index_name"],
                dimension=CONFIG["dimension"],
                metric=CONFIG["metric"],
                spec=CONFIG["spec"]
            )
            app.logger.info(f"Created new index: {CONFIG['index_name']}")
            
        index = pc.Index(CONFIG["index_name"])
        return index
        
    except Exception as e:
        app.logger.error(f"Index setup failed: {str(e)}")
        sys.exit(1)

@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "index": CONFIG["index_name"],
        "model": CONFIG["embedding_model"]
    })

@app.route('/upsert', methods=['POST'])
def upsert_documents():
    """Insert documents with embeddings"""
    try:
        documents = request.json
        vectors = [
            (
                doc["id"],
                model.encode(doc["text"]).tolist(),
                {"text": doc["text"], "category": doc.get("category", "")}
            ) for doc in documents
        ]
        
        index.upsert(vectors=vectors)
        return jsonify({
            "status": "success",
            "inserted": len(vectors)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """Query the index"""
    try:
        data = request.json
        results = index.query(
            vector=model.encode(data["query"]).tolist(),
            top_k=data.get("top_k", 1),
            include_metadata=True
        )
        
        return jsonify({
            "results": [
                {
                    "id": match.id,
                    "score": float(match.score),
                    "text": match.metadata["text"],
                    "category": match.metadata.get("category", "")
                } for match in results.matches
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def initialize_app():
    """Initialize all components"""
    initialize_components()
    setup_index()
    
    # Sample data (optional)
    sample_docs = [
        {"id": "doc1", "text": "Einstein's theory of relativity", "category": "science"},
        {"id": "doc2", "text": "Roman aqueducts were engineering marvels", "category": "history"}
    ]
    index.upsert(vectors=[
        (doc["id"], model.encode(doc["text"]).tolist(), {"text": doc["text"], "category": doc["category"]})
        for doc in sample_docs
    ])

if __name__ == '__main__':
    initialize_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)