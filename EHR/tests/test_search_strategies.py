"""Offline regression test for search/hill_climbing.py — no API key needed.
Run with: .venv/bin/python tests/test_search_strategies.py

search/llm_adaptive_search.py (the proposer LLM loop) needs a live call to test and isn't covered
here to avoid repeated cost on every run — it was validated live on 2026-09-29 (see
Planning_docs/findings.md and implementation_progress.md); Phase 3's live evaluation exercises it
for real.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from harness.retrieval_metrics import CosineScorer
from search.hill_climbing import hill_climb

print("=== hill_climb ===")
scorer = CosineScorer()
reference_bank = [
    "what is the date of birth of patient 31854?",
    "how many hours have passed since patient 31854 was admitted to the hospital currently?",
]
starting_query = "What is the weather like today in a completely unrelated topic?"

best, history = hill_climb(starting_query, reference_bank, scorer, iterations=60, seed=0)
print(f"starting score: {history[0]:.4f}")
print(f"best score:     {history[-1]:.4f}")
print(f"best candidate: {best!r}")

assert history[-1] > history[0], "expected score to improve over the starting query"
assert history == sorted(history), "expected score history to be monotonically non-decreasing"
print("PASS: hill-climbing improved the score and never regressed\n")

print("ALL PHASE 2 (hill_climbing) OFFLINE CHECKS PASSED")
