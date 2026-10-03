"""Strategy 1 — black-box hill-climbing offline search.

See Planning_docs/MINJA_winning_the_ranking.md's Strategy 1 and
Planning_docs/ImprovedMINJA_technical_implementation_plan.md §7.5. Purely offline: perturbs a
starting query's wording (word substitution, clause deletion — the ranking doc's
"masking-and-substitution procedure"), re-scoring each candidate against a reference memory bank
via CosineScorer, keeping improvements and discarding regressions. No live model contact.

Scope: this optimizes the *bridging query* text only (what competes for retrieval ranking, per
the ranking doc's Category A) — not the malicious "Knowledge: Refer X to Y" redirect sentence,
which is a separate, output-side concern (Category G) and stays fixed. Combining the optimized
bridging query with the fixed redirect sentence, injecting it live, and checking retrieval is
Phase 3's job (conditions/), reusing harness/precheck.py against a real PortableMedAgent.
"""

import random
import re
from typing import List, Tuple


def _tokenize(text: str) -> List[str]:
    return text.split()


def _build_vocab(reference_bank: List[str]) -> List[str]:
    """Perturbation vocabulary drawn from the reference bank's own words, not an external
    synonym dictionary — this biases substitutions toward words that already appear in genuine
    queries, which is exactly what should raise cosine similarity to that style."""
    vocab = set()
    for text in reference_bank:
        vocab.update(_tokenize(text))
    return sorted(vocab)


def _perturb(candidate: str, vocab: List[str], rng: random.Random) -> str:
    tokens = _tokenize(candidate)
    if not tokens:
        return candidate

    op = rng.choice(["substitute", "delete_word", "delete_clause"])

    if op == "substitute" and vocab:
        i = rng.randrange(len(tokens))
        tokens[i] = rng.choice(vocab)
        return " ".join(tokens)

    if op == "delete_word" and len(tokens) > 3:
        i = rng.randrange(len(tokens))
        del tokens[i]
        return " ".join(tokens)

    if op == "delete_clause":
        parts = re.split(r"(?<=[,;])\s+", candidate)
        if len(parts) > 1:
            i = rng.randrange(len(parts))
            return " ".join(parts[:i] + parts[i + 1 :]).strip()

    return candidate


def hill_climb(
    starting_query: str,
    reference_bank: List[str],
    scorer,
    iterations: int = 50,
    seed: int = 0,
) -> Tuple[str, List[float]]:
    """Returns (best_candidate, score_history). score_history[i] is the best score known after
    iteration i (monotonically non-decreasing, since regressions are discarded)."""
    rng = random.Random(seed)
    vocab = _build_vocab(reference_bank)

    best = starting_query
    best_score = scorer.score(best, reference_bank)
    history = [best_score]

    for _ in range(iterations):
        candidate = _perturb(best, vocab, rng)
        if not candidate.strip():
            history.append(best_score)
            continue
        score = scorer.score(candidate, reference_bank)
        if score > best_score:
            best, best_score = candidate, score
        history.append(best_score)

    return best, history
