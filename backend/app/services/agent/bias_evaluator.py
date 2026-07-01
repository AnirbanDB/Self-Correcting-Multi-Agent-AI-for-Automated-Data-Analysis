from typing import List
import numpy as np
from textblob import TextBlob
from app.services.agent.schemas import AgentEvaluationResult, BiasMetrics, PersonaResponse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ? This component is an example on how to quantify the bias of the agent injected with persona
# Define Canonical Anchors (The Reference Points)
pos_texts = [
    "The results are exceptionally positive and flawless.",
    "Revenue is skyrocketing with guaranteed growth.",
    "An amazing opportunity with zero risk.",
    "The market conditions are perfectly bullish.",
    "Company performance is outstanding and profitable."
]
  
neg_texts = [
    "The results are catastrophic and failing.",
    "Profit margins are collapsing immediately.",
    "This is a critical failure with immense risk.",
    "We are doomed to fail with no hope.",
    "The market is crashing and bearish."
]

class BiasEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the embedding model and defines the Bias Anchors.
        """
        self.model = SentenceTransformer(model_name)
        
        pos_matrix = self.model.encode(pos_texts)
        neg_matrix = self.model.encode(neg_texts)

        pos_mean = np.mean(pos_matrix, axis=0)
        neg_mean = np.mean(neg_matrix, axis=0)

        # Calculate the Average Vector)
        self.anchor_pos = pos_mean.reshape(1, -1)
        self.anchor_neg = neg_mean.reshape(1, -1)

    def get_bias_score(self, text: str) -> dict:
        """
        Calculates the Bias Score of a given text.
        Returns a dictionary with raw scores and the final 'Neutrality Index'.
        """
        # Vectorize the Input Text
        vec_input = self.model.encode([text])
        
        # Calculate Cosine Similarity to Anchors
        sim_pos = cosine_similarity(vec_input, self.anchor_pos)[0][0]
        sim_neg = cosine_similarity(vec_input, self.anchor_neg)[0][0]
        
        # Calculate Bias Metrics
        bias_score = sim_pos - sim_neg
        
        neutrality_index = 1.0 - abs(bias_score)
        
        return {
            "text_snippet": text[:50] + "...",
            "sim_positive": float(sim_pos),
            "sim_negative": float(sim_neg),
            "bias_score": float(bias_score),
            "neutrality_index": float(neutrality_index)
        }
    
    def _calculate_single_score(self, text: str) -> BiasMetrics:
        """Helper: Calculates metrics for a single string."""
        vec = self.model.encode([text])
        
        # 1. Cosine Similarity
        sim_pos = cosine_similarity(vec, self.anchor_pos)[0][0]
        sim_neg = cosine_similarity(vec, self.anchor_neg)[0][0]
        
        # 2. Mathematical Bias Score
        bias = sim_pos - sim_neg
        neutrality = 1.0 - abs(bias)
        
        # 3. Simple Polarity (Optional Check)
        polarity = TextBlob(text).sentiment.polarity

        return BiasMetrics(
            bias_score=float(bias),
            neutrality_index=float(neutrality),
            polarity=float(polarity)
        )
    
    def evaluate_batch(self, agent_responses: List[PersonaResponse]) -> List[AgentEvaluationResult]:
        """
        Takes your specific Agent output list, runs the math, 
        and returns the frontend-ready JSON.
        """
        evaluated_results: List[AgentEvaluationResult] = []

        for response in agent_responses:
            text_to_analyze = response['content'].summary
            
            metrics = self._calculate_single_score(text_to_analyze)
            
            evaluated_results.append(AgentEvaluationResult(
                role=response['role'],
                icon=response['icon'],
                persona=response['persona'],
                content=text_to_analyze,
                metrics=metrics
            ))
        
        return evaluated_results

