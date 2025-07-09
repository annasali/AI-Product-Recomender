# AI Product Recommender

This project is an AI-powered product recommendation system built with Python. It leverages embeddings and a simple web interface to suggest products based on user input.

## Features
- Embedding-based product recommendations
- Simple web interface for user interaction
- Modular codebase for easy extension

## Project Structure
```
ai_recommender.py         # Core recommendation logic
app.py                    # Web application (Flask or similar)
embedding_service.py      # Embedding generation and management
products_db.py            # Product database and utilities
requirements.txt          # Python dependencies
templates/
    index.html            # Main web interface
```

## Setup Instructions
1. **Clone the repository:**
   ```powershell
   git clone <repo-url>
   cd AI-Product-Recomender
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```powershell
   python app.py
   ```
4. **Open your browser:**
   Navigate to `http://localhost:5000` to use the recommender.

## Usage
- Enter your preferences or product description in the web interface.
- The system will recommend the most relevant products from the database.

## Customization
- Add or modify products in `products_db.py`.
- Adjust embedding logic in `embedding_service.py`.
- Update the web UI in `templates/index.html`.

