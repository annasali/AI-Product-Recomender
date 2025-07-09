"""
Products Database for AI Recommender Engine
Contains product information with detailed descriptions for embedding
"""

PRODUCTS_DATA = [
    {
        "id": "P001",
        "name": "Aromatherapy Essential Oil Diffuser",
        "category": "Wellness & Relaxation",
        "price": 89.99,
        "currency": "RM",
        "description": "Ultrasonic essential oil diffuser with 7 LED colors, perfect for stress relief and relaxation. Creates a calming atmosphere with soothing mist and ambient lighting. Ideal for meditation, sleep, and reducing anxiety.",
        "tags": ["relaxation", "stress relief", "aromatherapy", "sleep", "meditation", "wellness"],
        "emotion_context": ["stressed", "anxious", "tired", "overwhelmed"],
        "lifestyle_hints": ["busy professional", "student", "parent", "wellness enthusiast"],
        "special_offer": "Comes with 3 starter essential oils and 30-day money-back guarantee",
        "follow_up_tip": "Did you know lavender oil can reduce stress by up to 60%? Try it 30 minutes before bedtime!",
        "stock": 25
    },
    {
        "id": "P002",
        "name": "Nike Air Zoom Pegasus 40 (Wide Fit)",
        "category": "Athletic Footwear",
        "price": 459.00,
        "currency": "RM",
        "description": "Premium running shoe designed for flat feet with enhanced arch support and wide fit. Features responsive Zoom Air cushioning and breathable mesh upper. Stylish design suitable for both running and casual wear.",
        "tags": ["running", "flat feet", "wide fit", "stylish", "athletic", "comfortable"],
        "emotion_context": ["motivated", "energetic", "health-conscious"],
        "lifestyle_hints": ["runner", "fitness enthusiast", "active lifestyle", "athlete"],
        "special_offer": "Free gait analysis and 90-day comfort guarantee",
        "follow_up_tip": "Perfect for flat feet! The wide fit and arch support will improve your running form by 25%.",
        "stock": 15
    },
    {
        "id": "P003",
        "name": "Weighted Blanket - Bamboo Cooling",
        "category": "Sleep & Comfort",
        "price": 129.99,
        "currency": "RM",
        "description": "15lb weighted blanket made from bamboo fiber for natural cooling. Promotes deeper sleep and reduces anxiety through gentle pressure therapy. Hypoallergenic and machine washable.",
        "tags": ["sleep", "anxiety relief", "comfort", "weighted", "cooling", "relaxation"],
        "emotion_context": ["stressed", "anxious", "restless", "tired"],
        "lifestyle_hints": ["poor sleeper", "anxiety sufferer", "comfort seeker"],
        "special_offer": "7-day trial period - return if not satisfied",
        "follow_up_tip": "Weighted blankets can improve sleep quality by 42% and reduce cortisol levels naturally!",
        "stock": 18
    },
    {
        "id": "P004",
        "name": "Portable Bluetooth Speaker - Waterproof",
        "category": "Electronics & Audio",
        "price": 79.99,
        "currency": "RM",
        "description": "Compact waterproof Bluetooth speaker with 12-hour battery life. Perfect for outdoor adventures, beach trips, or relaxing at home. Crystal clear sound quality with deep bass.",
        "tags": ["music", "portable", "waterproof", "outdoor", "bluetooth", "entertainment"],
        "emotion_context": ["happy", "excited", "social", "adventurous"],
        "lifestyle_hints": ["outdoor enthusiast", "music lover", "social person", "traveler"],
        "special_offer": "Buy 2 get 1 free carrying case",
        "follow_up_tip": "Perfect for creating the right mood anywhere! Music can boost happiness by 89%.",
        "stock": 30
    },
    {
        "id": "P005",
        "name": "Mindfulness Journal & Pen Set",
        "category": "Self-Care & Wellness",
        "price": 34.99,
        "currency": "RM",
        "description": "Guided mindfulness journal with daily prompts for gratitude, reflection, and stress management. Includes premium pen and bookmark. Perfect for building healthy mental habits.",
        "tags": ["mindfulness", "journaling", "self-care", "mental health", "gratitude", "reflection"],
        "emotion_context": ["stressed", "overwhelmed", "seeking clarity", "anxious"],
        "lifestyle_hints": ["busy professional", "student", "self-improvement seeker"],
        "special_offer": "Free digital meditation app subscription (3 months)",
        "follow_up_tip": "Just 5 minutes of daily journaling can reduce stress by 30% and improve focus!",
        "stock": 40
    },
    {
        "id": "P006",
        "name": "Ergonomic Office Chair - Lumbar Support",
        "category": "Office & Workspace",
        "price": 299.99,
        "currency": "RM",
        "description": "Professional ergonomic chair with adjustable lumbar support and breathable mesh back. Reduces back pain and improves posture during long work sessions. Modern design fits any office.",
        "tags": ["ergonomic", "office", "back support", "comfortable", "professional", "posture"],
        "emotion_context": ["uncomfortable", "productive", "professional"],
        "lifestyle_hints": ["office worker", "remote worker", "student", "professional"],
        "special_offer": "Free desk assessment and 5-year warranty",
        "follow_up_tip": "Proper ergonomics can increase productivity by 40% and reduce back pain significantly!",
        "stock": 12
    },
    {
        "id": "P007",
        "name": "Herbal Tea Sampler Set",
        "category": "Beverages & Wellness",
        "price": 42.00,
        "currency": "RM",
        "description": "Collection of 12 premium herbal teas for relaxation, energy, and wellness. Includes chamomile, peppermint, ginger, and exotic blends. Perfect for stress relief and healthy hydration.",
        "tags": ["herbal tea", "relaxation", "wellness", "natural", "caffeine-free", "variety"],
        "emotion_context": ["stressed", "seeking comfort", "health-conscious"],
        "lifestyle_hints": ["tea lover", "wellness enthusiast", "natural health seeker"],
        "special_offer": "Free tea infuser and brewing guide",
        "follow_up_tip": "Chamomile tea can reduce anxiety by 50% - perfect for your evening routine!",
        "stock": 35
    },
    {
        "id": "P008",
        "name": "Resistance Bands Workout Set",
        "category": "Fitness & Exercise",
        "price": 65.99,
        "currency": "RM",
        "description": "Complete home workout set with 5 resistance levels. Perfect for strength training, rehabilitation, and staying fit at home. Includes door anchor and exercise guide.",
        "tags": ["fitness", "home workout", "strength training", "portable", "resistance", "exercise"],
        "emotion_context": ["motivated", "energetic", "health-conscious"],
        "lifestyle_hints": ["fitness enthusiast", "home workout", "busy schedule", "beginner"],
        "special_offer": "Free personal training session (virtual) and nutrition guide",
        "follow_up_tip": "Resistance training can build muscle and boost metabolism 24/7 - perfect for busy schedules!",
        "stock": 28
    },
    {
        "id": "P009",
        "name": "Smart Plant Care Monitor",
        "category": "Home & Garden",
        "price": 89.00,
        "currency": "RM",
        "description": "IoT device that monitors soil moisture, light, and temperature for your plants. Sends smartphone alerts and care tips. Perfect for plant lovers and beginners.",
        "tags": ["smart home", "plants", "gardening", "technology", "care", "monitoring"],
        "emotion_context": ["nurturing", "curious", "tech-savvy"],
        "lifestyle_hints": ["plant parent", "tech enthusiast", "home decorator", "beginner gardener"],
        "special_offer": "Free plant care app premium subscription (6 months)",
        "follow_up_tip": "Caring for plants can reduce stress by 68% and improve air quality naturally!",
        "stock": 20
    },
    {
        "id": "P010",
        "name": "Cozy Reading Nook Light",
        "category": "Home & Lighting",
        "price": 95.50,
        "currency": "RM",
        "description": "Adjustable LED reading light with warm/cool settings and touch control. Perfect for creating a cozy reading atmosphere. Reduces eye strain and provides comfortable lighting.",
        "tags": ["reading", "lighting", "cozy", "adjustable", "eye-friendly", "comfort"],
        "emotion_context": ["relaxed", "peaceful", "bookish"],
        "lifestyle_hints": ["book lover", "student", "night reader", "comfort seeker"],
        "special_offer": "Free bookmark set and book recommendation list",
        "follow_up_tip": "Good lighting can improve reading comprehension by 35% and reduce eye strain!",
        "stock": 22
    }
]

def get_all_products():
    """Return all products in the database"""
    return PRODUCTS_DATA

def get_product_by_id(product_id):
    """Get a specific product by ID"""
    for product in PRODUCTS_DATA:
        if product["id"] == product_id:
            return product
    return None

def get_products_by_category(category):
    """Get products by category"""
    return [product for product in PRODUCTS_DATA if product["category"] == category]

def get_products_by_price_range(min_price, max_price):
    """Get products within a price range"""
    return [product for product in PRODUCTS_DATA if min_price <= product["price"] <= max_price]

def get_product_embedding_text(product):
    """Create text for embedding that includes all relevant product information"""
    return f"""
    Product: {product['name']}
    Category: {product['category']}
    Price: {product['currency']} {product['price']}
    Description: {product['description']}
    Tags: {', '.join(product['tags'])}
    Emotional Context: {', '.join(product['emotion_context'])}
    Lifestyle: {', '.join(product['lifestyle_hints'])}
    Special Offer: {product['special_offer']}
    """.strip()