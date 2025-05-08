"""
Similarity search module for the Email Wizard Assistant.
This module handles the retrieval of relevant emails based on query similarity.
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimilaritySearch:
    """
    Class for performing similarity search on email embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the SimilaritySearch with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use for query embedding.
                              Default is 'all-MiniLM-L6-v2'.
        """
        logger.info(f"Initializing SimilaritySearch with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.email_ids = []
        self.emails = []
        
    def build_index(self, embeddings: Dict[str, np.ndarray], emails: List[Dict[str, Any]]) -> None:
        """
        Build a FAISS index for fast similarity search.
        
        Args:
            embeddings (Dict[str, np.ndarray]): Dictionary mapping email IDs to their embeddings.
            emails (List[Dict[str, Any]]): List of email data.
        """
        if not embeddings:
            logger.warning("No embeddings provided to build index")
            return
            
        # Extract embeddings and corresponding email IDs
        self.email_ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[email_id] for email_id in self.email_ids]).astype('float32')
        
        # Store emails for later retrieval
        self.emails = {email.get('id'): email for email in emails}
        
        # Get embedding dimension
        dimension = embedding_matrix.shape[1]
        
        # Create and train the index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embedding_matrix)
        
        logger.info(f"Built FAISS index with {len(self.email_ids)} emails and dimension {dimension}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string into a vector representation.
        
        Args:
            query (str): Query string to embed.
            
        Returns:
            np.ndarray: Vector embedding of the query.
        """
        embedding = self.model.encode(query)
        return embedding.astype('float32').reshape(1, -1)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the k most similar emails to the query.
        
        Args:
            query (str): Query string to search for.
            k (int): Number of similar emails to retrieve.
            
        Returns:
            List[Dict[str, Any]]: List of the k most similar emails with similarity scores.
        """
        if self.index is None:
            logger.error("Index not built. Call build_index() first.")
            return []
        
        # Embed the query
        query_embedding = self.embed_query(query)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.email_ids):
                email_id = self.email_ids[idx]
                email = self.emails.get(email_id, {})
                
                # Calculate similarity score (convert distance to similarity)
                similarity = 1.0 / (1.0 + distances[0][i])
                
                results.append({
                    'id': email_id,
                    'subject': email.get('subject', ''),
                    'snippet': email.get('body', '')[:100] + '...' if len(email.get('body', '')) > 100 else email.get('body', ''),
                    'similarity': float(similarity),
                    'email': email
                })
        
        logger.info(f"Found {len(results)} similar emails for query: {query}")
        return results
