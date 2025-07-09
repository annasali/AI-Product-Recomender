"""
Embedding Service for Product Recommender
Handles OpenAI embeddings and ChromaDB operations
"""

from openai import OpenAI
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any
import logging
from products_db import get_all_products, get_product_embedding_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, openai_api_key: str, chroma_db_path: str = "./chroma_db"):
        """
        Initialize the embedding service
        
        Args:
            openai_api_key: OpenAI API key
            chroma_db_path: Path to ChromaDB storage
        """
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collection name for products
        self.collection_name = "product_embeddings"
        
        # Initialize or get collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding for given text using OpenAI
        
        Args:
            text: Text to embed
            model: OpenAI embedding model (using latest model)
            
        Returns:
            List of float values representing the embedding
        """
        try:
            # Clean and prepare text
            text = text.replace("\n", " ").strip()
            
            # Get embedding from OpenAI using new API
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def embed_products(self) -> bool:
        """
        Embed all products and store in ChromaDB
        
        Returns:
            True if successful, False otherwise
        """
        try:
            products = get_all_products()
            
            # Prepare data for batch embedding
            documents = []
            metadatas = []
            ids = []
            
            for product in products:
                # Create embedding text
                embedding_text = get_product_embedding_text(product)
                documents.append(embedding_text)
                
                # Prepare metadata
                metadata = {
                    "product_id": product["id"],
                    "name": product["name"],
                    "category": product["category"],
                    "price": product["price"],
                    "currency": product["currency"],
                    "tags": ",".join(product["tags"]),
                    "emotion_context": ",".join(product["emotion_context"]),
                    "lifestyle_hints": ",".join(product["lifestyle_hints"]),
                    "special_offer": product["special_offer"],
                    "follow_up_tip": product["follow_up_tip"],
                    "stock": product["stock"]
                }
                metadatas.append(metadata)
                ids.append(product["id"])
            
            # Get embeddings for all documents
            logger.info("Getting embeddings from OpenAI...")
            embeddings = []
            for doc in documents:
                embedding = self.get_embedding(doc)
                embeddings.append(embedding)
            
            # Store in ChromaDB
            logger.info("Storing embeddings in ChromaDB...")
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully embedded {len(products)} products")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding products: {e}")
            return False
    
    def search_products(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for products based on query
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of product matches with metadata
        """
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "product_id": results['metadatas'][0][i]['product_id'],
                    "name": results['metadatas'][0][i]['name'],
                    "category": results['metadatas'][0][i]['category'],
                    "price": results['metadatas'][0][i]['price'],
                    "currency": results['metadatas'][0][i]['currency'],
                    "tags": results['metadatas'][0][i]['tags'].split(','),
                    "emotion_context": results['metadatas'][0][i]['emotion_context'].split(','),
                    "lifestyle_hints": results['metadatas'][0][i]['lifestyle_hints'].split(','),
                    "special_offer": results['metadatas'][0][i]['special_offer'],
                    "follow_up_tip": results['metadatas'][0][i]['follow_up_tip'],
                    "stock": results['metadatas'][0][i]['stock'],
                    "similarity_score": results['distances'][0][i],
                    "document": results['documents'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
    
    def get_collection_count(self) -> int:
        """Get the number of items in the collection"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all items)"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

# Initialize embedding service function
def initialize_embedding_service(openai_api_key: str) -> EmbeddingService:
    """Initialize and return embedding service"""
    return EmbeddingService(openai_api_key)

# Main function for testing
if __name__ == "__main__":
    # Test the embedding service
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize service
    service = initialize_embedding_service(os.getenv("OPENAI_API_KEY"))
    
    # Embed products if collection is empty
    if service.get_collection_count() == 0:
        print("Embedding products...")
        success = service.embed_products()
        if success:
            print(f"Successfully embedded {service.get_collection_count()} products")
        else:
            print("Failed to embed products")
    
    # Test search
    test_query = "I'm feeling stressed and need something relaxing under RM100"
    results = service.search_products(test_query)
    
    print(f"\nSearch results for: '{test_query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']} - {result['currency']}{result['price']}")
        print(f"   Special Offer: {result['special_offer']}")
        print(f"   Tip: {result['follow_up_tip']}")