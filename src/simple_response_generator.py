"""
Simple response generator module for the Email Wizard Assistant.
This is a lightweight alternative that doesn't require downloading large models.
"""

from typing import List, Dict, Any
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleResponseGenerator:
    """
    Class for generating simple responses based on retrieved emails without using ML models.
    """
    
    def __init__(self):
        """
        Initialize the SimpleResponseGenerator.
        """
        logger.info("Initializing SimpleResponseGenerator")
    
    def extract_key_sentences(self, text: str, query_terms: List[str], max_sentences: int = 3) -> str:
        """
        Extract key sentences from text that contain query terms.
        
        Args:
            text (str): Text to extract sentences from.
            query_terms (List[str]): List of query terms to look for.
            max_sentences (int): Maximum number of sentences to extract.
            
        Returns:
            str: Extracted sentences joined together.
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences based on query terms
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for term in query_terms if term.lower() in sentence.lower())
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
        
        # If no sentences match, take the first few sentences
        if not top_sentences and sentences:
            top_sentences = sentences[:max_sentences]
        
        return ' '.join(top_sentences)
    
    def generate_response(self, retrieved_emails: List[Dict[str, Any]], query: str) -> str:
        """
        Generate a simple response based on retrieved emails and the original query.
        
        Args:
            retrieved_emails (List[Dict[str, Any]]): List of retrieved emails with similarity scores.
            query (str): Original query string.
            
        Returns:
            str: Generated response.
        """
        if not retrieved_emails:
            logger.warning("No retrieved emails provided for response generation")
            return "I couldn't find any relevant information in your emails to answer this query."
        
        # Extract query terms (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'about', 'is', 'are', 'was', 'were'}
        query_terms = [term for term in query.lower().split() if term not in common_words]
        
        # Get the most relevant email
        most_relevant_email = retrieved_emails[0]['email']
        subject = most_relevant_email.get('subject', 'No Subject')
        body = most_relevant_email.get('body', '')
        
        # Extract key sentences from the email body
        key_content = self.extract_key_sentences(body, query_terms)
        
        # Create a response
        response = f"Based on your emails, I found relevant information in an email titled '{subject}'. "
        
        if key_content:
            response += key_content
        else:
            response += f"Here's a snippet: {body[:150]}..."
        
        # If we have multiple relevant emails, add a bit more context
        if len(retrieved_emails) > 1:
            second_email = retrieved_emails[1]['email']
            second_subject = second_email.get('subject', 'No Subject')
            response += f" I also found another relevant email titled '{second_subject}'."
        
        return response
