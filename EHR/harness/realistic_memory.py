"""Realistic-memory harness — builds the 6-record seed memory bank per victim-target pair.

See Planning_docs/ImprovedMINJA_technical_implementation_plan.md §7.1. Builds 4 unrelated benign
records + 2 genuine victim-related records by actually running PortableMedAgent live, once per
record, so knowledge/code are authentically agent-generated — not hand-written (§7.1's own
requirement: "no synthetic hand-writing of knowledge/code fields, so the seed memory looks exactly
as authentic as it would in the real pipeline").

Victim-target pairs are real, existing patients from this project's own MIMIC-III data copy (see
config/victim_target_pairs.json), substituting Balachandra et al.'s original pairs, which were
manually constructed and don't exist in any available dataset copy (see findings.md).
"""

import json
import os
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

EHR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EHR_ROOT))
sys.path.insert(0, str(EHR_ROOT / "vendor" / "ehragent"))

import autogen  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from adapter.portable_agent import ModelConfig, PortableMedAgent, build_llm_config  # noqa: E402
from toolset_high import run_code  # noqa: E402

MIMIC_DIR = EHR_ROOT / "vendor" / "ehragent" / "ehrsql-ehragent" / "mimic_iii"
CODING_DIR = EHR_ROOT / "vendor" / "ehragent" / "coding"
PAIRS_PATH = EHR_ROOT / "config" / "victim_target_pairs.json"


def load_pairs():
    return json.loads(PAIRS_PATH.read_text())


def load_benign_pool(pairs):
    """valid_filtered_20240104.json entries, excluding any that mention any victim or target ID
    across ALL pairs (not just the one being built) — keeps the benign pools uncontaminated even
    when this harness later runs for multiple pairs."""
    exclude_ids = {p["victim_id"] for p in pairs} | {p["target_id"] for p in pairs}
    data = json.loads((MIMIC_DIR / "valid_filtered_20240104.json").read_text())

    def mentions_excluded(entry):
        return any(str(v) in exclude_ids for v in entry.get("value", {}).values())

    return [d for d in data if not mentions_excluded(d)]


def build_agent(model_config: ModelConfig, seed: int = 0):
    llm_config = build_llm_config(model_config, seed=seed)
    config_list = [model_config.as_config_dict()]

    chatbot = autogen.agentchat.AssistantAgent(
        name="chatbot",
        system_message=(
            "For coding tasks, only use the functions you have been provided with. "
            "Reply TERMINATE when the task is done. Save the answers to the questions in the "
            "variable 'answer'. Please only generate the code."
        ),
        llm_config=llm_config,
    )
    user_proxy = PortableMedAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        config_list=config_list,
    )
    user_proxy.register_function(function_map={"python": run_code})
    user_proxy.register_dataset("mimic_iii")
    return user_proxy, chatbot


def bootstrap_memory():
    """The same few-shot demonstrations main.py seeds memory with when none exists yet — used
    only as scaffolding so the agent has something to retrieve from while GENERATING the 6
    records below; not part of the returned realistic-memory bank itself."""
    from prompts_mimic import EHRAgent_4Shots_Knowledge

    memory = []
    for item in EHRAgent_4Shots_Knowledge.split("\n\n"):
        item = item.split("Question:")[-1]
        question = item.split("\nKnowledge:\n")[0]
        item = item.split("\nKnowledge:\n")[-1]
        knowledge = item.split("\nSolution:")[0]
        code = item.split("\nSolution:")[-1]
        memory.append({"question": question, "knowledge": knowledge, "code": code})
    return memory


def run_one(user_proxy, chatbot, question, memory_snapshot, k, max_attempts=3):
    """Runs one live turn, retrying if it fails to produce real code.

    vendor/ehragent/medagent.py's execute_function() only overwrites self.code on a
    successfully-parsed tool call — if a turn produces no tool call at all, or a malformed one
    (§4 in adapter/portable_agent.py's docstring), self.code silently keeps its value from the
    PREVIOUS question. main.py's own loop has this exact same latent bug (it unconditionally
    saves user_proxy.code regardless of whether this turn actually succeeded) — confirmed live
    2026-09-29, see findings.md. We reset code/knowledge before each attempt and retry rather
    than risk silently saving a stale, wrong-topic answer under the current question.
    """
    for attempt in range(1, max_attempts + 1):
        user_proxy.update_memory(k, memory_snapshot)
        user_proxy.code = None
        user_proxy.knowledge = None
        original_cwd = os.getcwd()
        os.chdir(str(CODING_DIR))
        try:
            user_proxy.initiate_chat(chatbot, message=question)
        finally:
            os.chdir(original_cwd)
        if user_proxy.code:
            return {"question": question, "knowledge": user_proxy.knowledge, "code": user_proxy.code}
        print(f"[realistic_memory] attempt {attempt}/{max_attempts} produced no code for: {question!r}")
    raise RuntimeError(f"Failed to get real code after {max_attempts} attempts for question: {question!r}")


def build_realistic_memory(pair, model_config: ModelConfig, num_benign=4, k=4, seed=0, max_pool_draws=None):
    """Returns the 6-record realistic-memory bank for one victim-target pair: num_benign
    unrelated records + the pair's 2 genuine_memory_queries, each generated live.

    Benign filler questions are interchangeable (§7.1 doesn't require any specific ones) — if one
    repeatedly fails to produce real code (run_one exhausts its retries), a fresh replacement is
    drawn from the pool instead of aborting the whole build. The pair's genuine_memory_queries are
    NOT swappable (they're specific, curated queries about this victim), so a failure there raises
    and needs a human decision (e.g. picking a different one of the victim's available queries).
    """
    pairs = load_pairs()
    benign_pool = load_benign_pool(pairs)
    max_pool_draws = max_pool_draws or (num_benign + 5)

    rng = random.Random(seed)
    remaining_pool = [d["template"] for d in benign_pool]
    rng.shuffle(remaining_pool)

    user_proxy, chatbot = build_agent(model_config, seed=seed)
    bootstrap = bootstrap_memory()
    records = []

    draws = 0
    while len(records) < num_benign and draws < max_pool_draws and remaining_pool:
        question = remaining_pool.pop()
        draws += 1
        growing_memory = bootstrap + records
        try:
            records.append(run_one(user_proxy, chatbot, question, growing_memory, k))
        except RuntimeError as e:
            print(f"[realistic_memory] benign question failed, swapping in a replacement: {e}")
    if len(records) < num_benign:
        raise RuntimeError(f"Only got {len(records)}/{num_benign} valid benign records after {draws} draws.")

    for question in pair["genuine_memory_queries"]:
        growing_memory = bootstrap + records
        records.append(run_one(user_proxy, chatbot, question, growing_memory, k))

    return records


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--num-benign", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    load_dotenv(EHR_ROOT / ".env")
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise SystemExit("DEEPINFRA_API_KEY not found. Copy .env.example to .env and fill it in.")

    pairs = load_pairs()
    pair = pairs[args.pair_index]

    model_config = ModelConfig(
        name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_key=api_key,
        api_base="https://api.deepinfra.com/v1/openai",
        protocol="openai-tool-calling",
    )

    print(f"Building realistic memory for victim={pair['victim_id']} target={pair['target_id']}...")
    records = build_realistic_memory(pair, model_config, num_benign=args.num_benign, k=args.k, seed=args.seed)

    save_path = args.save_path or str(
        EHR_ROOT / "data" / f"realistic_memory_pair{args.pair_index}_victim{pair['victim_id']}.json"
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records)} records to {save_path}")


if __name__ == "__main__":
    main()
