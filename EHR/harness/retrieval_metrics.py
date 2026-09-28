"""CosineScorer — offline candidate scoring for Strategy 1/3's search loops.

See Planning_docs/ImprovedMINJA_technical_implementation_plan.md §4 (cosine-only, Levenshtein
dropped) and §7.4. hill_climbing.py and llm_adaptive_search.py (Phase 2) are handed a CosineScorer
instance and call score() to decide whether a candidate rewrite is an improvement.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class CosineScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Same model MedAgent.retrieve_examples uses (vendor/ehragent/medagent.py:48), loaded
        # once and shared across every score() call during a search run.
        self.model = SentenceTransformer(model_name)

    def score(self, candidate: str, memory_bank: List[str]) -> float:
        """Mean cosine similarity between candidate and each string in memory_bank.

        memory_bank is whatever the caller wants the candidate to resemble — per the ranking
        doc's Strategy 1 example ("check whether its cosine similarity to a typical benign
        query gets higher"), Phase 2's search code passes the realistic-memory bank's genuine
        victim-related records here, so a higher score means the candidate looks more like a
        real future victim query and is more likely to win the retrieval ranking once injected.
        """
        if not memory_bank:
            raise ValueError("memory_bank must be non-empty")
        candidate_embedding = self.model.encode(candidate, convert_to_numpy=True)
        bank_embeddings = self.model.encode(memory_bank, convert_to_numpy=True)
        similarities = cosine_similarity([candidate_embedding], bank_embeddings)[0]
        return float(np.mean(similarities))
