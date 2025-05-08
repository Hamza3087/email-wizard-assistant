"""
API routes for the Email Wizard Assistant.
"""

from flask import Blueprint, request, jsonify
import logging
import time
import json
import os
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import EmailEmbedder
from src.similarity_search import SimilaritySearch
from src.simple_response_generator import SimpleResponseGenerator

# Try to import the ML-based response generator, but don't fail if it can't be loaded
try:
    from src.response_generator import ResponseGenerator
    USE_ML_GENERATOR = True
except Exception as e:
    logger.warning(f"Could not import ResponseGenerator: {e}. Will use SimpleResponseGenerator instead.")
    USE_ML_GENERATOR = False

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize components
embedder = None
similarity_search = None
response_generator = None
simple_response_generator = None
emails = []

def load_emails():
    """
    Load emails from the sample dataset.

    Returns:
        List[Dict]: List of email data.
    """
    global emails
    try:
        # Use absolute path to ensure file is found
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sample_emails.json')
        with open(file_path, 'r') as f:
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
    global embedder, similarity_search, response_generator, simple_response_generator, emails

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

    # Initialize simple response generator if not already initialized
    if simple_response_generator is None:
        try:
            simple_response_generator = SimpleResponseGenerator()
        except Exception as e:
            logger.error(f"Error initializing simple response generator: {e}")
            return False

    # Initialize ML-based response generator if enabled and not already initialized
    if USE_ML_GENERATOR and response_generator is None:
        try:
            response_generator = ResponseGenerator()
        except Exception as e:
            logger.error(f"Error initializing ML response generator: {e}. Will use simple generator instead.")
            # We can continue even if this fails, as we have the simple generator as fallback

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

        # Generate response - try ML-based generator first, fall back to simple if needed
        if USE_ML_GENERATOR and response_generator is not None:
            try:
                response = response_generator.generate_response(retrieved_emails, query)
            except Exception as e:
                logger.warning(f"ML response generation failed: {e}. Using simple generator instead.")
                response = simple_response_generator.generate_response(retrieved_emails, query)
        else:
            response = simple_response_generator.generate_response(retrieved_emails, query)

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
