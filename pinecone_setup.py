import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import sys

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "index_name": "starter-index",
    "dimension": 384,  # all-MiniLM-L6-v2 embedding size
    "metric": "cosine",
    "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
    "embedding_model": "all-MiniLM-L6-v2"
}

def initialize_components():
    """Initialize Pinecone and embedding model with error handling"""
    try:
        # Validate API key
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
        
        pc = Pinecone(api_key=api_key)
        model = SentenceTransformer(CONFIG["embedding_model"])
        return pc, model
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        sys.exit(1)

def setup_index(pc):
    """Create or connect to Pinecone index with proper dimension"""
    try:
        # Check if index exists and has correct dimension
        if CONFIG["index_name"] in pc.list_indexes().names():
            index_info = pc.describe_index(CONFIG["index_name"])
            if index_info.dimension != CONFIG["dimension"]:
                print(f"Existing index has wrong dimension ({index_info.dimension}), deleting...")
                pc.delete_index(CONFIG["index_name"])
                print("Deleted old index with incorrect dimension")
                
        # Create new index if needed
        if CONFIG["index_name"] not in pc.list_indexes().names():
            pc.create_index(
                name=CONFIG["index_name"],
                dimension=CONFIG["dimension"],
                metric=CONFIG["metric"],
                spec=CONFIG["spec"]
            )
            print(f"Created index: {CONFIG['index_name']} with dimension {CONFIG['dimension']}")
        else:
            print(f"Using existing index: {CONFIG['index_name']}")
            
        return pc.Index(CONFIG["index_name"])
        
    except Exception as e:
        print(f"Index setup failed: {str(e)}")
        sys.exit(1)

def upsert_documents(index, model):
    """Insert sample documents with embeddings"""
    documents = [
        {
            "id": "doc1",
            "text": "Einstein's theory of relativity revolutionized modern physics",
            "category": "science"
        },
        {
            "id": "doc2", 
            "text": "Ancient Roman aqueducts were engineering marvels",
            "category": "history"
        },
        {
        "id": "doc3",
        "text": "Quantum mechanics describes subatomic particles",
        "category": "physics"
        },
       {
        "id": "doc4",
        "text": "The Colosseum was an ancient Roman amphitheater",
        "category": "history"
       }
    ]
    
    try:
        vectors = [
            (
                doc["id"],
                model.encode(doc["text"]).tolist(),
                {"text": doc["text"], "category": doc["category"]}
            ) for doc in documents
        ]
        
        index.upsert(vectors=vectors)
        print(f"Inserted {len(documents)} documents")
        return vectors
        
    except Exception as e:
        print(f"Document upload failed: {str(e)}")
        sys.exit(1)

def query_index(index, model, query_text, top_k=1):
    """Query the index and display results"""
    try:
        results = index.query(
            vector=model.encode(query_text).tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"\nTop {top_k} result(s) for '{query_text}':")
        if not results.matches:
            print("No matches found")
        else:
            for i, match in enumerate(results.matches, 1):
                print(f"\nMatch {i}:")
                print(f"ID: {match.id}")
                print(f"Score: {match.score:.4f}")
                print(f"Category: {match.metadata['category']}")
                print(f"Content: {match.metadata['text']}")
                
    except Exception as e:
        print(f"Query failed: {str(e)}")

def main():
    """Main execution flow"""
    # Initialize
    pc, model = initialize_components()
    
    # Setup index (now handles dimension mismatch)
    index = setup_index(pc)
    
    # Insert data
    upsert_documents(index, model)
    
    # Query examples
    query_index(index, model, "scientific theories")
    query_index(index, model, "ancient engineering", top_k=2)

if __name__ == "__main__":
    main()