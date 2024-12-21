import inspect
from collections.abc import Awaitable, Callable, Sequence
from functools import update_wrapper
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, cast, overload

from magentic.backend import get_chat_model
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    Placeholder,
    SystemMessage,
    UserMessage,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.logger import logger
from magentic.prompt_function import BasePromptFunction, AsyncPromptFunction, PromptFunction
from magentic.streaming import async_iter, azip

import random
import re
import numpy as np
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Tuple
from copy import deepcopy

P = ParamSpec("P")
R = TypeVar("R")

class LLMCall(Generic[P, R]):
    """Represents a call to an LLM and allows for structural modifications."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_root = True  # Flag to indicate if this is the starting node

    def insert_before(self, new_call: "LLMCall") -> "LLMCall":
        """Inserts a new LLMCall before the current one."""
        if self._parents:
            for parent in list(self._parents): # Iterate over a copy
                parent.remove_child(self)
                parent.add_child(new_call)
        else:
            self._is_root = False # No longer the root if inserting before
            new_call._is_root = True

        new_call.add_child(self)
        new_call._op = self._op # Inherit operation if applicable
        return new_call # Return the new root if this was the root

    def insert_after(self, new_call: "LLMCall") -> None:
        """Inserts a new LLMCall after the current one."""
        for child in list(self._children): # Iterate over a copy
            self.remove_child(child)
            new_call.add_child(child)
            new_call._op = child._op # Inherit operation

        self.add_child(new_call)
        return None

    def replace_with(self, new_call: "LLMCall") -> "LLMCall" :
        """Replaces the current LLMCall with a new one in the parent's children."""
        if self._parents:
            for parent in list(self._parents):
                parent.remove_child(self)
                parent.add_child(new_call)
        else:
            new_call._is_root = True
        return new_call if not self._parents else next(iter(self._parents)) # Return the new root or the parent

    def execute_flow(self) -> Any:
        """Executes the LLMCall and its children recursively."""
        self.execute()
        for child in self._children:
            child.execute_flow() # Or potentially process the output
        return self.output

    def deepcopy(self) -> "LLMCall":
        """Returns a deep copy of the LLMCall object and its connections."""
        return deepcopy(self)

    # ... (rest of the LLMCall class remains mostly the same, but ensure deepcopy works correctly)

def suggest_intelligent_modifications(
    current_call: LLMCall, error_signal: float, llama_intention: LLamaIntention, text_chunker: TextChunker
) -> List[LLMCall]:
    """Suggests intelligent modifications to the LLMCall."""
    modifications: List[LLMCall] = []

    # 1. Template Modifications (if error is high)
    if error_signal > 0.5:
        modifier = TemplateModifier(current_call.template)
        for mod_type, mod_kwargs in modifier.suggest_modifications():
            modified_call = current_call.deepcopy()
            modified_call.modify_template(mod_type, **mod_kwargs)
            modifications.append(modified_call)

    # 2. Structural Modifications based on LlamaAbstractor
    if isinstance(current_call.output, str) and llama_intention:
        abstraction_ranges = llama_intention.execute(current_call.output)
        for start, end in abstraction_ranges:
            abstracted_text = current_call.output[start:end]
            new_call_before = LLMCall(
                template="Process this abstraction: '{text}'",
                llama_intention=llama_intention,
                text_chunker=text_chunker,
                format_variables={"text": abstracted_text},
            )
            modified_call_insert_before = current_call.deepcopy()
            inserted_root = modified_call_insert_before.insert_before(new_call_before)
            modifications.append(inserted_root)

            new_call_after = LLMCall(
                template="Further process: '{text}'",
                llama_intention=llama_intention,
                text_chunker=text_chunker,
                format_variables={"text": current_call.output},
            )
            modified_call_insert_after = current_call.deepcopy()
            modified_call_insert_after.insert_after(new_call_after)
            modifications.append(modified_call_insert_after.deepcopy()) # Ensure the modified structure is captured

    # 3. Consider replacing the current call (radical reflection)
    replacement_call = LLMCall(
        template="Let's try a different approach: {text}",
        llama_intention=llama_intention,
        text_chunker=text_chunker,
        format_variables=current_call.format_variables,
    )
    modified_call_replace = current_call.deepcopy()
    replaced_root = modified_call_replace.replace_with(replacement_call)
    modifications.append(replaced_root)

    return modifications

async def main():
    # Initialize the LLM, LlamaIntention, and TextChunker
    llm = Llama(
        model_path="/path/to/your/model.gguf",  # Add path to model
        n_ctx=8192,
        n_gpu_layers=17,
        seed=42,
    )
    llama_intention = LLamaIntention(llm)
    text_chunker = TextChunker(global_threshold=0.6, relative_threshold=0.2)
    text_chunker.train(["This is a sentence. This is another one.", "Short text."])

    # Initialize the root LLM call
    llm_call1 = LLMCall(
        template="Translate '{text}' to French.",
        llama_intention=llama_intention,
        text_chunker=text_chunker,
        format_variables={"text": "{input_text}"}
    )
    llm_call2 = LLMCall(
        template="Summarize this text: {text}.",
        llama_intention=llama_intention,
        text_chunker=text_chunker,
        format_variables={"text": "{input_text}"}
    )
    llm_call3 = LLMCall(
        template="What is the sentiment of this text: {text}?",
        llama_intention=llama_intention,
        text_chunker=text_chunker,
        format_variables={"text": "{input_text}"}
    )

    initial_call = llm_call1 + llm_call2 / llm_call3 # Example initial program

    beam_size = 3
    active_beams = [initial_call]  # Start with the initial program

    input_text = "This is a test. This is more text to test the chunker. And even more."

    for iteration in range(3):  # Limit iterations for demonstration
        print(f"Iteration {iteration + 1}")
        next_beams = []

        for current_program_root in active_beams:
            # Execute the current program
            result_root = current_program_root.deepcopy() # Execute on a copy
            result_root(input_text=input_text).execute_flow() # Execute with input
            error_signal = result_root.evaluate_output() # Evaluate the output of the root
            print(f"Beam Program Output: {result_root.output}, Error: {error_signal:.2f}")

            # Suggest modifications
            modifications = suggest_intelligent_modifications(
                result_root, error_signal, llama_intention, text_chunker
            )
            next_beams.extend(modifications)

        # Prune the beam
        next_beams.sort(key=lambda call: call.evaluate_output() if call.output is not None else float('inf')) # Lower error is better
        active_beams = next_beams[:beam_size]

        if not active_beams:
            break

    if active_beams:
        best_program = active_beams[0]
        print(f"Best Program Output after search: {best_program.output}")
    else:
        print("No promising programs found.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
