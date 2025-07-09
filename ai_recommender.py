"""
AI Recommender Service
Handles LLM-based intent detection, emotion analysis, and recommendation generation
"""

from openai import OpenAI
import re
import json
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
from embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIRecommenderService:
    def __init__(self, openai_api_key: str, embedding_service: EmbeddingService):
        """
        Initialize AI Recommender Service
        
        Args:
            openai_api_key: OpenAI API key
            embedding_service: Initialized embedding service
        """
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_service = embedding_service
        
        # Emotion patterns for detection
        self.emotion_patterns = {
            "stressed": ["stressed", "overwhelmed", "pressure", "tension", "anxious", "worried"],
            "tired": ["tired", "exhausted", "fatigue", "sleepy", "worn out", "drained"],
            "happy": ["happy", "excited", "joyful", "cheerful", "delighted", "thrilled"],
            "sad": ["sad", "depressed", "down", "blue", "melancholy", "upset"],
            "energetic": ["energetic", "motivated", "active", "pumped", "vigorous"],
            "relaxed": ["relaxed", "calm", "peaceful", "serene", "tranquil", "chill"],
            "frustrated": ["frustrated", "annoyed", "irritated", "aggravated", "fed up"],
            "confident": ["confident", "assured", "self-assured", "bold", "determined"]
        }
        
        # Budget patterns
        self.budget_patterns = {
            "under_50": ["under 50", "below 50", "less than 50", "under rm50", "below rm50"],
            "under_100": ["under 100", "below 100", "less than 100", "under rm100", "below rm100"],
            "under_200": ["under 200", "below 200", "less than 200", "under rm200", "below rm200"],
            "under_500": ["under 500", "below 500", "less than 500", "under rm500", "below rm500"],
            "no_limit": ["no limit", "unlimited", "any price", "money no object", "whatever it costs"]
        }
    
    def analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Analyze user input to extract intent, emotions, budget, and context
        
        Args:
            user_input: User's message
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Use OpenAI to analyze the input
            analysis_prompt = f"""
            Analyze the following user message for a product recommendation system:
            
            User Message: "{user_input}"
            
            Extract the following information and respond in JSON format:
            {{
                "primary_emotion": "detected emotion (stressed, happy, tired, etc.)",
                "secondary_emotions": ["list of other emotions detected"],
                "intent": "what the user is looking for",
                "budget_mentioned": "extracted budget amount or range",
                "specific_requirements": ["list of specific requirements"],
                "lifestyle_hints": ["inferred lifestyle characteristics"],
                "urgency_level": "low/medium/high",
                "product_category_preference": "preferred product category if mentioned",
                "context_clues": ["situational context from the message"]
            }}
            
            Be specific and detailed in your analysis.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing user messages for product recommendations. Always respond with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse the JSON response
            analysis = json.loads(response.choices[0].message.content)
            
            # Add additional manual analysis
            analysis["detected_budget_range"] = self._extract_budget_range(user_input)
            analysis["raw_input"] = user_input
            analysis["timestamp"] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing user input: {e}")
            # Return basic analysis if LLM fails
            return {
                "primary_emotion": self._detect_emotion_simple(user_input),
                "secondary_emotions": [],
                "intent": "product recommendation",
                "budget_mentioned": self._extract_budget_range(user_input),
                "specific_requirements": [],
                "lifestyle_hints": [],
                "urgency_level": "medium",
                "product_category_preference": "",
                "context_clues": [],
                "detected_budget_range": self._extract_budget_range(user_input),
                "raw_input": user_input,
                "timestamp": datetime.now().isoformat()
            }
    
    def _detect_emotion_simple(self, text: str) -> str:
        """Simple emotion detection fallback"""
        text_lower = text.lower()
        for emotion, patterns in self.emotion_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return emotion
        return "neutral"
    
    def _extract_budget_range(self, text: str) -> Dict[str, Any]:
        """Extract budget information from text"""
        text_lower = text.lower()
        
        # Look for specific amounts
        amount_match = re.search(r'rm\s*(\d+)|under\s*(\d+)|below\s*(\d+)|less\s*than\s*(\d+)', text_lower)
        if amount_match:
            amount = next(g for g in amount_match.groups() if g)
            return {"type": "max", "amount": int(amount), "currency": "RM"}
        
        # Look for budget patterns
        for budget_type, patterns in self.budget_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                budget_ranges = {
                    "under_50": {"type": "max", "amount": 50, "currency": "RM"},
                    "under_100": {"type": "max", "amount": 100, "currency": "RM"},
                    "under_200": {"type": "max", "amount": 200, "currency": "RM"},
                    "under_500": {"type": "max", "amount": 500, "currency": "RM"},
                    "no_limit": {"type": "unlimited", "amount": None, "currency": "RM"}
                }
                return budget_ranges.get(budget_type, {"type": "unknown", "amount": None, "currency": "RM"})
        
        return {"type": "unknown", "amount": None, "currency": "RM"}
    
    def get_recommendations(self, user_input: str, num_recommendations: int = 3) -> Dict[str, Any]:
        """
        Get product recommendations based on user input
        
        Args:
            user_input: User's message
            num_recommendations: Number of recommendations to return
            
        Returns:
            Dictionary containing recommendations and analysis
        """
        try:
            # Analyze user input
            analysis = self.analyze_user_input(user_input)
            
            # Create enhanced search query
            search_query = self._create_search_query(analysis)
            
            # Search for products
            search_results = self.embedding_service.search_products(
                query=search_query,
                n_results=num_recommendations * 2  # Get more results to filter
            )
            
            # Filter results based on budget and requirements
            filtered_results = self._filter_by_budget_and_requirements(
                search_results, 
                analysis
            )
            
            # Take top recommendations
            recommendations = filtered_results[:num_recommendations]
            
            # Generate personalized response
            response = self._generate_personalized_response(
                recommendations, 
                analysis
            )
            
            return {
                "recommendations": recommendations,
                "analysis": analysis,
                "personalized_response": response,
                "search_query_used": search_query,
                "total_matches": len(search_results),
                "filtered_matches": len(filtered_results)
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                "recommendations": [],
                "analysis": {},
                "personalized_response": "Sorry, I encountered an error while processing your request. Please try again.",
                "search_query_used": "",
                "total_matches": 0,
                "filtered_matches": 0
            }
    
    def _create_search_query(self, analysis: Dict[str, Any]) -> str:
        """Create an enhanced search query based on analysis"""
        query_parts = []
        
        # Add primary emotion
        if analysis.get("primary_emotion"):
            query_parts.append(f"feeling {analysis['primary_emotion']}")
        
        # Add intent
        if analysis.get("intent"):
            query_parts.append(analysis["intent"])
        
        # Add specific requirements
        if analysis.get("specific_requirements"):
            query_parts.extend(analysis["specific_requirements"])
        
        # Add lifestyle hints
        if analysis.get("lifestyle_hints"):
            query_parts.extend(analysis["lifestyle_hints"])
        
        # Add context clues
        if analysis.get("context_clues"):
            query_parts.extend(analysis["context_clues"])
        
        # Add category preference
        if analysis.get("product_category_preference"):
            query_parts.append(analysis["product_category_preference"])
        
        return " ".join(query_parts)
    
    def _filter_by_budget_and_requirements(self, results: List[Dict], analysis: Dict) -> List[Dict]:
        """Filter results based on budget and requirements"""
        filtered = []
        
        budget_info = analysis.get("detected_budget_range", {})
        max_budget = budget_info.get("amount")
        
        for result in results:
            # Budget filter
            if max_budget and result["price"] > max_budget:
                continue
            
            # Stock filter
            if result["stock"] <= 0:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def _generate_personalized_response(self, recommendations: List[Dict], analysis: Dict) -> str:
        """Generate a personalized response based on recommendations and analysis"""
        try:
            if not recommendations:
                return "I couldn't find any products that match your specific requirements and budget. Could you provide more details about what you're looking for?"
            
            # Create context for the LLM
            context = f"""
            User Analysis:
            - Primary Emotion: {analysis.get('primary_emotion', 'neutral')}
            - Intent: {analysis.get('intent', 'product recommendation')}
            - Budget: {analysis.get('detected_budget_range', {}).get('amount', 'not specified')}
            - Specific Requirements: {', '.join(analysis.get('specific_requirements', []))}
            - Lifestyle Hints: {', '.join(analysis.get('lifestyle_hints', []))}
            
            Recommended Products:
            """
            
            for i, rec in enumerate(recommendations, 1):
                context += f"""
                {i}. {rec['name']} - {rec['currency']}{rec['price']}
                   Special Offer: {rec['special_offer']}
                   Follow-up Tip: {rec['follow_up_tip']}
                """
            
            response_prompt = f"""
            Based on the user analysis and recommended products above, create a personalized, empathetic response that:
            
            1. Acknowledges the user's emotional state and needs
            2. Explains why these products are perfect for them
            3. Highlights special offers and tips
            4. Uses a warm, conversational tone
            5. Keeps the response concise but informative
            
            Context: {context}
            
            Write a natural, helpful response as if you're a knowledgeable friend making recommendations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a helpful, empathetic product recommendation assistant. Respond in a warm, conversational tone."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating personalized response: {e}")
            # Fallback response
            response_parts = []
            emotion = analysis.get('primary_emotion', 'neutral')
            
            if emotion == 'stressed':
                response_parts.append("I understand you're feeling stressed. Here are some products that might help you relax:")
            elif emotion == 'tired':
                response_parts.append("It sounds like you need some rest and comfort. These products might be perfect:")
            else:
                response_parts.append("Based on your needs, here are my recommendations:")
            
            for i, rec in enumerate(recommendations, 1):
                response_parts.append(f"\n{i}. **{rec['name']}** - {rec['currency']}{rec['price']}")
                response_parts.append(f"   💡 {rec['follow_up_tip']}")
                if rec['special_offer']:
                    response_parts.append(f"   🎁 {rec['special_offer']}")
            
            return " ".join(response_parts)

# CRM logging functionality
class CRMLogger:
    def __init__(self, log_file: str = "crm_interactions.json"):
        """
        Initialize CRM logger
        
        Args:
            log_file: Path to the log file
        """
        self.log_file = log_file
    
    def log_interaction(self, user_input: str, analysis: Dict, recommendations: List[Dict], 
                       response: str, user_id: str = None, session_id: str = None):
        """
        Log user interaction for CRM purposes
        
        Args:
            user_input: Original user input
            analysis: Analysis results
            recommendations: Product recommendations
            response: System response
            user_id: Optional user identifier
            session_id: Optional session identifier
        """
        try:
            interaction_log = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "session_id": session_id,
                "user_input": user_input,
                "analysis": analysis,
                "recommendations": [
                    {
                        "product_id": rec["product_id"],
                        "name": rec["name"],
                        "price": rec["price"],
                        "similarity_score": rec.get("similarity_score", 0)
                    }
                    for rec in recommendations
                ],
                "system_response": response,
                "interaction_type": "product_recommendation"
            }
            
            # Load existing logs
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []
            
            # Add new log
            logs.append(interaction_log)
            
            # Save logs
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"Logged interaction for user {user_id or 'anonymous'}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get user interaction history
        
        Args:
            user_id: User identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of user interactions
        """
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            user_logs = [log for log in logs if log.get("user_id") == user_id]
            return user_logs[-limit:]  # Return most recent interactions
            
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []

# Main recommendation function
def get_personalized_recommendations(user_input: str, openai_api_key: str, 
                                   user_id: str = None, session_id: str = None) -> Dict[str, Any]:
    """
    Main function to get personalized product recommendations
    
    Args:
        user_input: User's message
        openai_api_key: OpenAI API key
        user_id: Optional user identifier
        session_id: Optional session identifier
        
    Returns:
        Complete recommendation response
    """
    try:
        # Initialize services
        from embedding_service import initialize_embedding_service
        
        embedding_service = initialize_embedding_service(openai_api_key)
        ai_service = AIRecommenderService(openai_api_key, embedding_service)
        crm_logger = CRMLogger()
        
        # Ensure products are embedded
        if embedding_service.get_collection_count() == 0:
            logger.info("No products found in database. Embedding products...")
            embedding_service.embed_products()
        
        # Get recommendations
        result = ai_service.get_recommendations(user_input)
        
        # Log interaction
        crm_logger.log_interaction(
            user_input=user_input,
            analysis=result["analysis"],
            recommendations=result["recommendations"],
            response=result["personalized_response"],
            user_id=user_id,
            session_id=session_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in main recommendation function: {e}")
        return {
            "recommendations": [],
            "analysis": {},
            "personalized_response": "Sorry, I encountered an error. Please try again.",
            "error": str(e)
        }

# Test function
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test the recommendation system
    test_input = "I'm feeling stressed lately and I want to try something relaxing under RM100."
    
    result = get_personalized_recommendations(
        user_input=test_input,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        user_id="test_user_123"
    )
    
    print("=== AI Recommendation System Test ===")
    print(f"Input: {test_input}")
    print(f"\nResponse: {result['personalized_response']}")
    print(f"\nRecommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec['name']} - {rec['currency']}{rec['price']}")
        print(f"   {rec['special_offer']}")
        print()