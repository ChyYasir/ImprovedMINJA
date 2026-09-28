"""Portability adapter for MedAgent.

See Planning_docs/ImprovedMINJA_technical_implementation_plan.md §5. Subclasses the vendored
MedAgent (copied into vendor/ehragent/, not imported from MINJA/ — see §5's revised decision)
to fix three things without editing the vendor copy's logic:

1. retrieve_knowledge() constructs its own OpenAI() client with only api_key, no base_url
   (vendor/ehragent/medagent.py:91-93) — it can only ever reach OpenAI's real API regardless
   of which model is nominally the target. This silently contaminates Condition 4 (the
   same-model ablation) whenever the target isn't actually OpenAI.
2. The model isn't resolved from config.py's hardcoded allowlist (which only recognizes
   'gpt-4'/'gpt-4o' and silently falls back to 'o1-preview' otherwise) — ModelConfig is
   passed in explicitly instead.
3. vendor/ehragent/config.py's llm_config_list() builds the legacy, deprecated `functions=`
   request schema. Confirmed via a live test (2026-09-28) that this silently fails against
   DeepInfra's Llama-3.1-8B-Instruct: the model returns plain text instead of a structured
   call, since DeepInfra's API only documents/supports the modern `tools`/`tool_choice`
   schema (their own docs show no `functions` example at all) and pyautogen's client does
   zero functions->tools translation. build_llm_config() below branches on
   ModelConfig.protocol and builds `tools=` instead for protocol="openai-tool-calling".
   Required pyautogen>=0.2.35 (bumped from the originally-pinned 0.2.0), which is the first
   version with generate_tool_calls_reply — confirmed by reading its source that it routes
   tool_calls through the exact same execute_function() the vendor's MedAgent implements, so
   no vendor code needed to change for this fix.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "ehragent"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from medagent import MedAgent  # noqa: E402 (vendor import, path inserted above)


@dataclass
class ModelConfig:
    """name/api_base/api_key/protocol per plan §5. Config values come from config/models.yaml,
    not from vendor/ehragent/config.py's openai_config() allowlist.

    protocol: "openai-function-calling" (legacy `functions=` schema — matches vendor/ehragent/
    config.py's llm_config_list(), works against real OpenAI) or "openai-tool-calling" (modern
    `tools=` schema — required for DeepInfra targets, see module docstring point 3)."""

    name: str
    api_key: str
    api_base: Optional[str] = None
    protocol: str = "openai-function-calling"

    def as_config_dict(self) -> dict:
        """Shape expected by pyautogen 0.2.x's config_list and by
        PortableMedAgent.retrieve_knowledge()'s OpenAI() client construction below —
        both accept `base_url` directly (verified against pyautogen 0.2.x docs)."""
        d = {"model": self.name, "api_key": self.api_key}
        if self.api_base:
            d["base_url"] = self.api_base
        return d


# Same "python" function schema vendor/ehragent/config.py's llm_config_list() embeds under
# `functions=` — kept here verbatim so build_llm_config() can wrap it in either schema without
# poking into the vendor module's internals.
_PYTHON_FUNCTION_SCHEMA = {
    "name": "python",
    "description": "run the entire code and return the execution result. Only generate the code.",
    "parameters": {
        "type": "object",
        "properties": {
            "cell": {
                "type": "string",
                "description": "Valid Python code to execute.",
            }
        },
        "required": ["cell"],
    },
}


def build_llm_config(model_config: ModelConfig, seed: int) -> dict:
    """Builds the AssistantAgent llm_config for model_config, choosing the request schema its
    protocol needs (see ModelConfig docstring and module docstring point 3)."""
    config_list = [model_config.as_config_dict()]
    if model_config.protocol == "openai-tool-calling":
        return {
            "tools": [{"type": "function", "function": _PYTHON_FUNCTION_SCHEMA}],
            "config_list": config_list,
            "timeout": 120,
            "cache_seed": seed,
            "temperature": 0,
        }
    from config import llm_config_list  # vendor's own legacy builder, unmodified — noqa: E402

    return llm_config_list(seed, config_list)


class PortableMedAgent(MedAgent):
    def retrieve_knowledge(self, config, query):
        if self.dataset == "mimic_iii":
            from prompts_mimic import RetrKnowledge
        else:
            from prompts_eicu import RetrKnowledge

        patience = 2
        sleep_time = 30
        engine = config["model"]

        demo_complete = self.retrieve_examples(query)
        demo_complete = demo_complete.split("Question:")

        demo_knowledge = []
        for trunk in demo_complete:
            if "Solution:" in trunk:
                knowledge = trunk.split("Solution:")[0]
                demo_knowledge.append("Question:" + knowledge)
        demo_knowledge = "\n".join(demo_knowledge)

        query_message = RetrKnowledge.format(demonstrations=demo_knowledge, question=query)
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": query_message},
        ]

        # The actual fix (§5, point 3): base_url threaded through, not just api_key. Without
        # this, this client silently keeps hitting OpenAI's real API no matter which model
        # config["model"] names.
        client = OpenAI(api_key=config["api_key"], base_url=config.get("base_url"))

        while patience > 0:
            patience -= 1
            try:
                response = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=0,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction is not None:
                    return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to retrieve related knowledge, please try again later."

    def retrieve_examples(self, query, return_indices=False):
        """Same cosine-similarity top-k logic as MedAgent.retrieve_examples (unchanged), but
        can also return which memory-bank indices were selected — this is what §7.2's Precheck
        instrumentation (Phase 1's precheck.py) needs for ISR: is the injected record's index
        in the returned set, yes/no, rather than string-matching the final output for the
        target ID. Default behavior (return_indices=False) is unchanged, so retrieve_knowledge()
        above and the inherited generate_init_message() keep working exactly as before."""
        query_embedding = self._get_text_embedding(query)
        past_embeddings = [self._get_text_embedding(memory["question"]) for memory in self.memory]

        similarities = cosine_similarity([query_embedding], past_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        top_examples = []
        selected_indices = []
        for i in sorted_indices[: self.num_shots]:
            question = self.memory[i]["question"]
            knowledge = self.memory[i]["knowledge"]
            code = self.memory[i]["code"]
            similarity = similarities[i]
            template = (
                f"Question: {question}\n"
                f"Cosine Similarity: {similarity:.4f}\n"
                f"Knowledge:\n{knowledge}\n"
                f"Solution:\n{code}\n"
            )
            top_examples.append(template)
            selected_indices.append(int(i))

        combined_output = "\n".join(top_examples)
        if return_indices:
            return combined_output, selected_indices
        return combined_output
