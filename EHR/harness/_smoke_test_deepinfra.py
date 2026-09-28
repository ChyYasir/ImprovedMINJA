"""Live smoke test: full pipeline against Llama-3.1-8B-Instruct via DeepInfra.

Exercises, together, for the first time: AssistantAgent function-calling routed through
DeepInfra's OpenAI-compatible endpoint, PortableMedAgent.retrieve_knowledge()'s base_url fix
(also routed through DeepInfra), and run_code() querying the real MIMIC-III data — plus the
chdir-to-coding-dir fix confirmed necessary in the earlier manual test.

Requires: a .env file in this directory (EHR/.env, gitignored) with DEEPINFRA_API_KEY=...
          — copy .env.example to .env and fill it in.
Run with: .venv/bin/python harness/_smoke_test_deepinfra.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "vendor/ehragent")

import autogen
from dotenv import load_dotenv

from adapter.portable_agent import ModelConfig, PortableMedAgent, build_llm_config
from toolset_high import run_code

load_dotenv()  # reads EHR/.env when run with CWD=EHR/, per the run instructions above
api_key = os.environ.get("DEEPINFRA_API_KEY")
if not api_key:
    raise SystemExit("DEEPINFRA_API_KEY not found. Copy .env.example to .env and fill it in.")

model_config = ModelConfig(
    name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key=api_key,
    api_base="https://api.deepinfra.com/v1/openai",
    protocol="openai-tool-calling",  # the fix: DeepInfra needs tools=, not legacy functions=
)
config_list = [model_config.as_config_dict()]
llm_config = build_llm_config(model_config, seed=0)

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
    # use_docker: False — code actually executes via the vendor's own function_map={"python":
    # run_code} path (exec() in-process, see toolset_high.py), never through AutoGen's own
    # docker-based executor. pyautogen 0.2.35 added an init-time docker availability check
    # regardless of which executor is actually used; this just satisfies that check.
    config_list=config_list,
)
user_proxy.register_function(function_map={"python": run_code})
user_proxy.register_dataset("mimic_iii")

# Seed memory the same way main.py does when no memory file is loaded — parse the vendor's
# own EHRAgent_4Shots_Knowledge demonstrations rather than hand-writing something different.
from prompts_mimic import EHRAgent_4Shots_Knowledge  # noqa: E402

seed_memory = []
for item in EHRAgent_4Shots_Knowledge.split("\n\n"):
    item = item.split("Question:")[-1]
    question = item.split("\nKnowledge:\n")[0]
    item = item.split("\nKnowledge:\n")[-1]
    knowledge = item.split("\nSolution:")[0]
    code = item.split("\nSolution:")[-1]
    seed_memory.append({"question": question, "knowledge": knowledge, "code": code})
user_proxy.update_memory(4, seed_memory)

# A real, patient-agnostic question from the vendor's own dataset (valid_filtered_20240104.json
# entry 0) — no victim/target ID involved, this is purely a pipeline smoke test.
question = "what is the intake method of lidocaine 5% ointment?"

# The chdir fix (found + verified manually earlier): tools/tabtools.py's hardcoded relative
# paths only resolve if CWD is vendor/ehragent/coding/ at code-execution time, and nothing
# does this automatically for the function_map-based execution path.
original_cwd = os.getcwd()
target_cwd = os.path.abspath("vendor/ehragent/coding")
os.chdir(target_cwd)
try:
    user_proxy.initiate_chat(chatbot, message=question)
finally:
    os.chdir(original_cwd)

print("\n=== final agent state ===")
print("question:", question)
print("knowledge (from retrieve_knowledge, via DeepInfra):", repr(user_proxy.knowledge)[:200])
print("generated code:", repr(user_proxy.code)[:300])
print("\nSMOKE TEST COMPLETE — no exceptions means the full chain (function-calling,")
print("retrieve_knowledge via base_url fix, run_code against real MIMIC-III data) worked.")
