"""
API routes for the Email Wizard Assistant.
"""

from flask import Blueprint, request, jsonify
import logging
import time
import json
import os
import sys

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import EmailEmbedder
from src.similarity_search import SimilaritySearch
from src.response_generator import ResponseGenerator

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize components
embedder = None
similarity_search = None
response_generator = None
emails = []

def load_emails():
    """
    Load emails from the sample dataset.
    
    Returns:
        List[Dict]: List of email data.
    """
    global emails
    try:
        with open('data/sample_emails.json', 'r') as f:
            emails = json.load(f)
        logger.info(f"Loaded {len(emails)} emails from dataset")
        return emails
    except Exception as e:
        logger.error(f"Error loading emails: {e}")
        return []

def initialize_components():
    """
    Initialize the email wizard components.
    """
    global embedder, similarity_search, response_generator, emails
    
    # Load emails if not already loaded
    if not emails:
        emails = load_emails()
    
    # Initialize embedder if not already initialized
    if embedder is None:
        embedder = EmailEmbedder()
        
        # Check if embeddings exist, if not create them
        try:
            embeddings = embedder.load_embeddings()
            if not embeddings and emails:
                logger.info("No existing embeddings found, creating new embeddings")
                embeddings = embedder.embed_emails(emails)
                embedder.save_embeddings(embeddings)
        except Exception as e:
            logger.error(f"Error initializing embedder: {e}")
            return False
    
    # Initialize similarity search if not already initialized
    if similarity_search is None:
        try:
            similarity_search = SimilaritySearch()
            embeddings = embedder.load_embeddings()
            similarity_search.build_index(embeddings, emails)
        except Exception as e:
            logger.error(f"Error initializing similarity search: {e}")
            return False
    
    # Initialize response generator if not already initialized
    if response_generator is None:
        try:
            response_generator = ResponseGenerator()
        except Exception as e:
            logger.error(f"Error initializing response generator: {e}")
            return False
    
    return True

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON: Health status.
    """
    return jsonify({"status": "healthy", "message": "Email Wizard Assistant API is running"})

@api_bp.route('/query_email', methods=['POST'])
def query_email():
    """
    Query email endpoint.
    
    Returns:
        JSON: Generated response and retrieved emails.
    """
    # Initialize components if not already initialized
    if not initialize_components():
        return jsonify({"error": "Failed to initialize components"}), 500
    
    # Get query from request
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query parameter"}), 400
    
    query = data['query']
    logger.info(f"Received query: {query}")
    
    try:
        # Record start time for performance measurement
        start_time = time.time()
        
        # Search for similar emails
        retrieved_emails = similarity_search.search(query, k=5)
        
        # Generate response
        response = response_generator.generate_response(retrieved_emails, query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response data
        response_data = {
            "response": response,
            "retrieved_emails": [
                {
                    "id": email['id'],
                    "subject": email['subject'],
                    "snippet": email['snippet'],
                    "similarity": email['similarity']
                } for email in retrieved_emails
            ],
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500
