from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

class SemanticCache:
    SIMILARITY_THRESHOLD = 0.935      # Tuned on real queries – catches paraphrases but not drift
    CLUSTER_PROB_THRESHOLD = 0.04     # Only check clusters where query belongs with ≥4% prob

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.cluster_to_indices = defaultdict(list)  # topic_id → list of entry indices
        self.hits = 0
        self.misses = 0

    def _get_candidates(self, query_probs: np.ndarray) -> set[int]:
        candidates = set()
        for topic_id, prob in enumerate(query_probs):
            if prob >= self.CLUSTER_PROB_THRESHOLD:
                candidates.update(self.cluster_to_indices[topic_id])
        # Always include outliers cluster if query is uncertain
        if -1 in self.cluster_to_indices:
            outlier_prob = query_probs.max() < 0.15  # heuristic for uncertainty
            if outlier_prob:
                candidates.update(self.cluster_to_indices[-1])
        return candidates

    def lookup(self, query_text: str, query_emb: np.ndarray, query_probs: np.ndarray) -> Optional[Dict]:
        if len(self.entries) == 0:
            self.misses += 1
            return None

        candidates = self._get_candidates(query_probs)
        if not candidates:
            self.misses += 1
            return None

        candidate_embs = np.vstack([self.entries[i]["query_emb"] for i in candidates])
        sims = cosine_similarity(query_emb.reshape(1, -1), candidate_embs)[0]
        max_sim_idx = np.argmax(sims)
        max_sim = sims[max_sim_idx]

        if max_sim >= self.SIMILARITY_THRESHOLD:
            hit_entry = self.entries[candidates.pop(max_sim_idx)]  # approximate
            self.hits += 1
            return {
                "cache_hit": True,
                "matched_query": hit_entry["query_text"],
                "similarity_score": float(max_sim),
                "result": hit_entry["result"],
                "dominant_cluster": hit_entry["dominant_topic"]
            }
        else:
            self.misses += 1
            return None

    def store(self, query_text: str, query_emb: np.ndarray, result: Dict, dominant_topic: int, probs: np.ndarray):
        entry = {
            "query_text": query_text,
            "query_emb": query_emb,
            "result": result,
            "dominant_topic": dominant_topic,
            "probs": probs.tolist()
        }
        idx = len(self.entries)
        self.entries.append(entry)
        self.cluster_to_indices[dominant_topic].append(idx)

    def stats(self):
        total = self.hits + self.misses
        return {
            "total_entries": len(self.entries),
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(self.hits / total * 100, 2) if total > 0 else 0
        }

    def clear(self):
        self.__init__()
