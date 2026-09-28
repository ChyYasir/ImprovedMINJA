import time
from typing import Dict, List, Optional, Union, Callable, Literal
import logging
import asyncio
import openai
import json
from openai import OpenAI
from autogen.agentchat import Agent, UserProxyAgent, ConversableAgent
from termcolor import colored
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken

logger = logging.getLogger(__name__)

class MedAgent(UserProxyAgent):
    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, Literal[False]]] = False,
        system_message: Optional[Union[str, List]] = "",
        config_list: Optional[List[Dict]] = None,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )
        self.config_list = config_list
        self.question = ''
        self.code = ''
        self.knowledge = ''

        # Initialize SentenceTransformer model for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your preferred model
        #all-MiniLM-L6-v2
        #facebook-dpr-ctx_encoder-single-nq-base
        #msmarco-roberta-base-ance-firstp
        
        # Token counting
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text using the tiktoken library.
        """
        encoding = tiktoken.encoding_for_model("gpt-4")  # Replace with your model
        return len(encoding.encode(text))
        
    def retrieve_knowledge(self, config, query):
        if self.dataset == 'mimic_iii':
            from prompts_mimic import RetrKnowledge
        else:
            from prompts_eicu import RetrKnowledge

        patience = 2
        sleep_time = 30
        
        openai.api_key = config["api_key"]
        engine = config["model"]

        demo_complete = self.retrieve_examples(query)
        demo_complete = demo_complete.split('Question:')
        
        demo_knowledge = []
        for trunk in demo_complete:
            if 'Solution:' in trunk:
                knowledge = trunk.split('Solution:')[0]
                demo_knowledge.append('Question:' + knowledge)
                
        demo_knowledge = '\n'.join(demo_knowledge)
        
        query_message = RetrKnowledge.format(demonstrations=demo_knowledge, question=query)

        messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": query_message}]
        client = OpenAI(
            api_key=config["api_key"],
        )

        while patience > 0:
            patience -= 1
            try:
                response = client.chat.completions.create(
                    model=engine,
                    messages = messages,
                    temperature=0,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                    )
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction is not None:
                    return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to retrieve related knowledge, please try again later."

    def _get_text_embedding(self, text):
        """
        Generate a text embedding for the given text using SentenceTransformer.
        """
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(384)  # Return a zero vector for fallback

    def retrieve_examples(self, query):
        # Extract embeddings for the query and past questions
        query_embedding = self._get_text_embedding(query)
        past_embeddings = [self._get_text_embedding(memory["question"]) for memory in self.memory]

        # Calculate cosine similarities between the query and each past question
        similarities = cosine_similarity(
            [query_embedding],  # Query embedding as a 2D array
            past_embeddings     # Past question embeddings as a 2D array
        )[0]  # Extract the 1D array of similarities

        # Sort by similarity (highest similarity first)
        sorted_indices = np.argsort(similarities)[::-1]

        # Prepare a list to store the top N most similar examples
        top_examples = []
        for idx, i in enumerate(sorted_indices[:self.num_shots]):
            question = self.memory[i]["question"]
            knowledge = self.memory[i]["knowledge"]
            code = self.memory[i]["code"]
            similarity = similarities[i]

            # Format the top examples with similarity score
            template = (
                f"Question: {question}\n"
                f"Cosine Similarity: {similarity:.4f}\n"
                f"Knowledge:\n{knowledge}\n"
                f"Solution:\n{code}\n"
            )
            top_examples.append(template)

        # Combine the top examples into a single string
        combined_output = '\n'.join(top_examples)
        return combined_output

    def generate_init_message(self, **context):
        if self.dataset == 'mimic_iii':
            from prompts_mimic import EHRAgent_Message_Prompt
        else:
            from prompts_eicu import EHRAgent_Message_Prompt
        self.question = context["message"]
        knowledge = self.retrieve_knowledge(self.config_list[0], context["message"])
        self.knowledge = knowledge

        examples = self.retrieve_examples(context["message"])

        init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=knowledge, question=context["message"])
        return init_message

    def send(self, message: Union[Dict, str], recipient: Agent, request_reply: Optional[bool]=None, silent: Optional[bool]=False):
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def initiate_chat(self, recipient: "ConversableAgent", clear_history: Optional[bool]=True, silent: Optional[bool]=False, **context):
        self._prepare_chat(recipient, clear_history)
        
        # Generate and count tokens in the input message
        init_message = self.generate_init_message(**context)
        self.total_input_tokens += self._count_tokens(init_message)
        
        self.send(self.generate_init_message(**context), recipient, silent=silent)

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        self._process_received_message(message, sender, silent)
        
        # Count tokens in the response
        if isinstance(message, str):
            self.total_output_tokens += self._count_tokens(message)
        elif isinstance(message, dict) and "content" in message:
            self.total_output_tokens += self._count_tokens(message["content"])
        
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)
            
    def get_total_tokens(self) -> Dict[str, int]:
        """
        Return the total input and output tokens.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens
        }

    def error_debugger(self, config, code, error_info):
        if self.dataset == 'mimic_iii':
            from prompts_mimic import CodeDebugger
        else:
            from prompts_eicu import CodeDebugger

        patience = 2
        sleep_time = 30
        query_message = CodeDebugger.format(question=self.question, code=code, error_info=error_info)

        messages = [{"role": "system",
                     "content": "You are an AI assistant that helps people debug their code. Only list one most possible reason to the errors."},
                    {"role": "user", "content": query_message}]

        while patience > 0:
            patience -= 1
            try:
                # Assume we have some function to generate debugging responses
                prediction = "Debugging response here"  # Replace with the actual interaction
                if prediction != "" and prediction is not None:
                    return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to diagnose the reasons to the errors."

    def execute_function(self, func_call):
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = {"cell": func_call["arguments"].split(': "')[-1].split('", ')[0]}
                content = f"Error: {e}\n There might be compilation errors in the code. Please check the code and try again."

            if arguments is not None:
                print(
                    colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"),
                    flush=True,
                )
                self.code = arguments["cell"]
                try:
                    content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."
        if "error" in content or "Error" in content:
            reasons = self.error_debugger(self.config_list[0], self.code, content)
            content = content + '\nPotential Reasons: ' + reasons

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }
    
    def update_memory(self, num_shots, memory):
        self.num_shots = num_shots
        self.memory = memory

    def register_dataset(self, dataset):
        self.dataset = dataset
