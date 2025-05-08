"""
Main Flask application for the Email Wizard Assistant.
"""

from flask import Flask
from flask_cors import CORS
import logging
import os
from dotenv import load_dotenv
from routes import api_bp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('email_wizard.log')
    ]
)
logger = logging.getLogger(__name__)

def create_app():
    """
    Create and configure the Flask application.

    Returns:
        Flask: Configured Flask application.
    """
    app = Flask(__name__)

    # Enable CORS for all routes
    CORS(app)

    # Register blueprints
    app.register_blueprint(api_bp)

    # Log application startup
    logger.info("Email Wizard Assistant API started")

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
