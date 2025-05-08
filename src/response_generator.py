"""
Response generator module for the Email Wizard Assistant.
This module handles the generation of responses based on retrieved emails.
"""

from typing import List, Dict, Any
import logging
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Class for generating responses based on retrieved emails using a RAG approach.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize the ResponseGenerator with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use for response generation.
                              Default is "google/flan-t5-base".
        """
        logger.info(f"Initializing ResponseGenerator with model: {model_name}")
        self.model_name = model_name
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create a text generation pipeline
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=150
        )
    
    def prepare_context(self, retrieved_emails: List[Dict[str, Any]], query: str) -> str:
        """
        Prepare context from retrieved emails for response generation.
        
        Args:
            retrieved_emails (List[Dict[str, Any]]): List of retrieved emails with similarity scores.
            query (str): Original query string.
            
        Returns:
            str: Formatted context for the model.
        """
        context = f"Query: {query}\n\nRelevant Email Information:\n"
        
        for i, email in enumerate(retrieved_emails):
            email_data = email.get('email', {})
            subject = email_data.get('subject', 'No Subject')
            body = email_data.get('body', 'No Content')
            sender = email_data.get('sender', 'Unknown')
            date = email_data.get('date', 'Unknown Date')
            
            # Add email information to context
            context += f"Email {i+1}:\n"
            context += f"Subject: {subject}\n"
            context += f"From: {sender}\n"
            context += f"Date: {date}\n"
            context += f"Content: {body[:200]}...\n\n"
        
        return context
    
    def generate_response(self, retrieved_emails: List[Dict[str, Any]], query: str) -> str:
        """
        Generate a response based on retrieved emails and the original query.
        
        Args:
            retrieved_emails (List[Dict[str, Any]]): List of retrieved emails with similarity scores.
            query (str): Original query string.
            
        Returns:
            str: Generated response.
        """
        if not retrieved_emails:
            logger.warning("No retrieved emails provided for response generation")
            return "I couldn't find any relevant information in your emails to answer this query."
        
        # Prepare context from retrieved emails
        context = self.prepare_context(retrieved_emails, query)
        
        # Create prompt for the model
        prompt = f"{context}\n\nBased on the above emails, provide a concise answer to the query: {query}"
        
        # Generate response
        logger.info(f"Generating response for query: {query}")
        response = self.generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        
        return response
