"""
Flask Backend API for AI Product Recommender
Handles HTTP requests and integrates all services
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import uuid
import logging
from datetime import datetime
from ai_recommender import get_personalized_recommendations, CRMLogger
from embedding_service import initialize_embedding_service
from products_db import get_all_products, get_product_by_id
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['SECRET_KEY'] = os.urandom(24)  # Generate random secret key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY environment variable is required")
    print("🔧 SOLUTION:")
    print("1. Get your API key from: https://platform.openai.com/api-keys")
    print("2. Create a .env file in your project directory with:")
    print("   OPENAI_API_KEY=your_api_key_here")
    print("3. Or set it as an environment variable:")
    print("   export OPENAI_API_KEY=your_api_key_here")
    print("\n💡 For testing, you can also run:")
    print("   OPENAI_API_KEY=your_key python3 app.py")
    exit(1)

# Initialize services
embedding_service = initialize_embedding_service(OPENAI_API_KEY)
crm_logger = CRMLogger()

# Ensure products are embedded on startup
def initialize_database():
    """Initialize the product database with embeddings"""
    try:
        if embedding_service.get_collection_count() == 0:
            logger.info("Initializing product embeddings...")
            success = embedding_service.embed_products()
            if success:
                logger.info(f"Successfully embedded {embedding_service.get_collection_count()} products")
            else:
                logger.error("Failed to embed products")
        else:
            logger.info(f"Found {embedding_service.get_collection_count()} products in database")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Initialize database when the module loads
initialize_database()

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "products_count": embedding_service.get_collection_count()
    })

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Get product recommendations based on user input
    
    Expected JSON payload:
    {
        "message": "user input text",
        "user_id": "optional user identifier",
        "session_id": "optional session identifier"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' field in request body"
            }), 400
        
        user_input = data['message'].strip()
        if not user_input:
            return jsonify({
                "error": "Message cannot be empty"
            }), 400
        
        # Extract optional parameters
        user_id = data.get('user_id', str(uuid.uuid4()))
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Get recommendations
        result = get_personalized_recommendations(
            user_input=user_input,
            openai_api_key=OPENAI_API_KEY,
            user_id=user_id,
            session_id=session_id
        )
        
        # Format response
        response = {
            "success": True,
            "user_input": user_input,
            "response": result["personalized_response"],
            "recommendations": [
                {
                    "id": rec["product_id"],
                    "name": rec["name"],
                    "category": rec["category"],
                    "price": rec["price"],
                    "currency": rec["currency"],
                    "special_offer": rec["special_offer"],
                    "follow_up_tip": rec["follow_up_tip"],
                    "stock": rec["stock"],
                    "tags": rec["tags"]
                }
                for rec in result["recommendations"]
            ],
            "analysis": {
                "emotion": result["analysis"].get("primary_emotion"),
                "intent": result["analysis"].get("intent"),
                "budget": result["analysis"].get("detected_budget_range"),
                "requirements": result["analysis"].get("specific_requirements", [])
            },
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "total_matches": result.get("total_matches", 0),
                "filtered_matches": result.get("filtered_matches", 0)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error occurred",
            "message": "Please try again later"
        }), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products or filter by category/price"""
    try:
        category = request.args.get('category')
        min_price = request.args.get('min_price', type=float)
        max_price = request.args.get('max_price', type=float)
        
        products = get_all_products()
        
        # Apply filters
        if category:
            products = [p for p in products if p['category'].lower() == category.lower()]
        
        if min_price is not None:
            products = [p for p in products if p['price'] >= min_price]
        
        if max_price is not None:
            products = [p for p in products if p['price'] <= max_price]
        
        return jsonify({
            "success": True,
            "products": products,
            "count": len(products)
        })
        
    except Exception as e:
        logger.error(f"Error in products endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch products"
        }), 500

@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """Get a specific product by ID"""
    try:
        product = get_product_by_id(product_id)
        
        if not product:
            return jsonify({
                "success": False,
                "error": "Product not found"
            }), 404
        
        return jsonify({
            "success": True,
            "product": product
        })
        
    except Exception as e:
        logger.error(f"Error in product endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch product"
        }), 500

@app.route('/api/user/history/<user_id>', methods=['GET'])
def get_user_history(user_id):
    """Get user interaction history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        history = crm_logger.get_user_history(user_id, limit)
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "history": history,
            "count": len(history)
        })
        
    except Exception as e:
        logger.error(f"Error in user history endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch user history"
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all product categories"""
    try:
        products = get_all_products()
        categories = list(set(product['category'] for product in products))
        categories.sort()
        
        return jsonify({
            "success": True,
            "categories": categories,
            "count": len(categories)
        })
        
    except Exception as e:
        logger.error(f"Error in categories endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch categories"
        }), 500

@app.route('/api/search', methods=['POST'])
def search_products():
    """Direct product search using vector similarity"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request body"
            }), 400
        
        query = data['query'].strip()
        limit = data.get('limit', 5)
        
        if not query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        # Search using embedding service
        results = embedding_service.search_products(query, limit)
        
        return jsonify({
            "success": True,
            "query": query,
            "results": [
                {
                    "id": result["product_id"],
                    "name": result["name"],
                    "category": result["category"],
                    "price": result["price"],
                    "currency": result["currency"],
                    "similarity_score": result["similarity_score"],
                    "special_offer": result["special_offer"],
                    "tags": result["tags"]
                }
                for result in results
            ],
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Search failed"
        }), 500

@app.route('/api/admin/reset-embeddings', methods=['POST'])
def reset_embeddings():
    """Admin endpoint to reset and rebuild embeddings"""
    try:
        # Reset collection
        success = embedding_service.reset_collection()
        if not success:
            return jsonify({
                "success": False,
                "error": "Failed to reset collection"
            }), 500
        
        # Rebuild embeddings
        success = embedding_service.embed_products()
        if not success:
            return jsonify({
                "success": False,
                "error": "Failed to rebuild embeddings"
            }), 500
        
        return jsonify({
            "success": True,
            "message": f"Successfully reset and rebuilt embeddings for {embedding_service.get_collection_count()} products"
        })
        
    except Exception as e:
        logger.error(f"Error in reset embeddings endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to reset embeddings"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# Run the application
if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )