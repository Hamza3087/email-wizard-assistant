"""
Email embedding module for the Email Wizard Assistant.
This module handles the conversion of emails into vector embeddings.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmailEmbedder:
    """
    Class for embedding emails using pre-trained models.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the EmailEmbedder with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use for embeddings.
                              Default is 'all-MiniLM-L6-v2'.
        """
        logger.info(f"Initializing EmailEmbedder with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def preprocess_email(self, email: Dict[str, Any]) -> str:
        """
        Preprocess an email for embedding.

        Args:
            email (Dict[str, Any]): Email data containing subject, body, etc.

        Returns:
            str: Preprocessed email text ready for embedding.
        """
        # Combine relevant fields for embedding
        subject = email.get('subject', '')
        body = email.get('body', '')
        sender = email.get('sender', '')

        # Create a combined text representation
        combined_text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"

        return combined_text

    def embed_email(self, email: Dict[str, Any]) -> np.ndarray:
        """
        Embed a single email into a vector representation.

        Args:
            email (Dict[str, Any]): Email data to embed.

        Returns:
            np.ndarray: Vector embedding of the email.
        """
        preprocessed_text = self.preprocess_email(email)
        embedding = self.model.encode(preprocessed_text)
        return embedding

    def embed_emails(self, emails: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Embed a list of emails into vector representations.

        Args:
            emails (List[Dict[str, Any]]): List of email data to embed.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping email IDs to their embeddings.
        """
        logger.info(f"Embedding {len(emails)} emails")
        email_embeddings = {}

        for email in emails:
            email_id = email.get('id', str(hash(email.get('subject', '') + email.get('body', ''))))
            embedding = self.embed_email(email)
            email_embeddings[email_id] = embedding

        logger.info(f"Completed embedding {len(email_embeddings)} emails")
        return email_embeddings

    def save_embeddings(self, embeddings: Dict[str, np.ndarray], file_path: str = None) -> None:
        """
        Save email embeddings to a file.

        Args:
            embeddings (Dict[str, np.ndarray]): Dictionary of email embeddings.
            file_path (str): Path to save the embeddings file.
        """
        if file_path is None:
            # Use default path relative to the current file
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'embeddings.pkl')

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Saved {len(embeddings)} embeddings to {file_path}")

    def load_embeddings(self, file_path: str = None) -> Dict[str, np.ndarray]:
        """
        Load email embeddings from a file.

        Args:
            file_path (str): Path to the embeddings file.

        Returns:
            Dict[str, np.ndarray]: Dictionary of email embeddings.
        """
        if file_path is None:
            # Use default path relative to the current file
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'embeddings.pkl')

        if not os.path.exists(file_path):
            logger.warning(f"Embeddings file {file_path} not found")
            return {}

        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
        return embeddings
