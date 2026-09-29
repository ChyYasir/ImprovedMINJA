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
4. execute_function() is overridden to normalize a malformed tool-call shape observed live
   (2026-09-29) from Llama-3.1-8B-Instruct via DeepInfra: instead of `{"cell": "..."}`, it
   sometimes echoes the whole function schema back as arguments, e.g.
   `{"type": "function", "name": "python", "parameters": {"cell": "..."}}`. vendor/ehragent/
   medagent.py's execute_function() expects `arguments["cell"]` directly and raises KeyError
   on this shape. The override flattens it before delegating to the vendored logic unchanged.
5. The dominant failure mode, found 2026-09-29 (~2/3 of attempts in one run): Llama-3.1-8B-
   Instruct via DeepInfra sometimes emits its tool call in Llama's own native prompt-template
   syntax, `<function=NAME>{...}`, as plain message *content*, instead of populating the
   structured `tool_calls` API field. pyautogen's generate_tool_calls_reply only looks at
   `tool_calls`, so when this happens no execution is attempted at all — the turn silently
   produces nothing. _pseudo_tool_call_reply() is registered ahead of the normal reply chain
   to detect this pattern and route it through execute_function() same as a real tool call;
   it's a no-op (defers immediately) whenever tool_calls/function_call are actually present.
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from autogen.agentchat import Agent
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


def _extract_cell(parsed):
    """Finds a "cell" value inside a possibly-malformed tool-call arguments structure — see
    PortableMedAgent.execute_function's docstring for the observed shapes this handles."""
    if isinstance(parsed, dict):
        if "cell" in parsed:
            return parsed["cell"]
        for v in parsed.values():
            found = _extract_cell(v)
            if found is not None:
                return found
    elif isinstance(parsed, list) and parsed:
        return _extract_cell(parsed[0])
    return None


def build_llm_config(model_config: ModelConfig, seed: int) -> dict:
    """Builds the AssistantAgent llm_config for model_config, choosing the request schema its
    protocol needs (see ModelConfig docstring and module docstring point 3)."""
    config_list = [model_config.as_config_dict()]
    if model_config.protocol == "openai-tool-calling":
        return {
            "tools": [{"type": "function", "function": _PYTHON_FUNCTION_SCHEMA}],
            "config_list": config_list,
            "timeout": 120,
            # cache_seed intentionally omitted (disables pyautogen's response cache): confirmed
            # live (2026-09-29) that a fixed cache_seed made an identical-input retry replay the
            # exact same malformed tool-call response instead of attempting a fresh generation,
            # defeating the point of retrying a failed turn.
            "temperature": 0,
        }
    from config import llm_config_list  # vendor's own legacy builder, unmodified — noqa: E402

    return llm_config_list(seed, config_list)


_FUNCTION_TAG_RE = re.compile(r"<function=(\w+)>\s*(\{.*?\})\s*(?:</function>)?\s*$", re.DOTALL)


class PortableMedAgent(MedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Registered last so it's checked FIRST (pyautogen's register_reply: later registration
        # = earlier check, at the default position=0) — see point 5 in the module docstring.
        self.register_reply([Agent, None], PortableMedAgent._pseudo_tool_call_reply)

    def _pseudo_tool_call_reply(self, messages=None, sender=None, config=None):
        """Handles a fifth observed failure mode (2026-09-29, the dominant one — ~2/3 of
        benign-record attempts in one run): Llama-3.1-8B-Instruct via DeepInfra sometimes emits
        its tool call in Llama's own native prompt-template syntax, `<function=NAME>{...}`, as
        plain message *content*, instead of the structured `tool_calls` API field pyautogen's
        generate_tool_calls_reply expects. When that happens, tool_calls is empty, no execution
        is attempted at all, and the turn silently produces no code. This reply function is
        registered to run first; it only acts when tool_calls/function_call are absent AND the
        content matches this pattern — otherwise it defers (returns False, None) to the normal
        handlers, so correctly-formed calls are completely unaffected."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if message.get("tool_calls") or message.get("function_call"):
            return False, None
        content = message.get("content") or ""
        match = _FUNCTION_TAG_RE.search(content.strip())
        if not match:
            return False, None
        func_name, raw_args = match.group(1), match.group(2)
        _, func_return = self.execute_function({"name": func_name, "arguments": raw_args})
        return True, func_return

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

    def execute_function(self, func_call):
        """Normalizes the malformed tool-call shapes described in the module docstring (point 4)
        before delegating to MedAgent's own execute_function, unchanged. Observed live shapes
        (2026-09-29, Llama-3.1-8B-Instruct via DeepInfra): the whole schema echoed back with the
        real args nested under "parameters", and arguments wrapped in a list.

        Rather than special-case every shape this occasionally-non-conforming 8B model might
        produce, delegation is wrapped defensively: a shape neither _extract_cell nor the
        vendored execute_function can handle is reported back to the model as a corrective
        error (matching vendor's own error-reporting convention, e.g. its error_debugger path)
        instead of crashing the run — the model gets a chance to self-correct on its next turn,
        same as it already does for real code-execution errors."""
        raw_args = func_call.get("arguments", "{}")
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if not (isinstance(parsed, dict) and "cell" in parsed):
            cell = _extract_cell(parsed)
            if cell is not None:
                func_call = dict(func_call)
                func_call["arguments"] = json.dumps({"cell": cell})
        try:
            return super().execute_function(func_call)
        except (KeyError, TypeError, IndexError) as e:
            print(f"[PortableMedAgent] malformed tool-call arguments, raw={raw_args!r}: {e}")
            return False, {
                "name": func_call.get("name", ""),
                "role": "function",
                "content": (
                    f"Error: {e}. The arguments must be a plain JSON object with only a "
                    'single "cell" key, e.g. {"cell": "<python code>"} — no other keys, no '
                    "list wrapping, no repeating the function schema."
                ),
            }
