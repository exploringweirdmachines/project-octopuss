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

P = ParamSpec("P")
# TODO: Make `R` type Union of all possible return types except FunctionCall ?
# Then `R | FunctionCall[FuncR]` will separate FunctionCall from other return types.
# Can then use `FuncR` to make typechecker check `functions` argument to `prompt`
# `Not` type would solve this - https://github.com/python/typing/issues/801
R = TypeVar("R")

class LLMCall(Generic[P, R]):
    """Represents a call to an LLM with a prompt template and functions.

    This class allows for the chaining of LLM calls using arithmetic operators.
    """

    def __init__(
        self,
        template: str,
        functions: list[Callable[..., Any]] | None = None,
        model: ChatModel | None = None,
        output_type: type[R] | None = None,
        format_variables: dict[str, Any] | None = None,
        template_modifiers: list[Callable[[str], str]] | None = None,
    ):
        self.template = template
        self.functions = functions or []
        self.model = model or get_chat_model()
        self.output_type = output_type or str
        self.format_variables = format_variables or {}
        self.output: str | None = None  # Store the output of the LLM call
        self.input: str | None = None  # Store the formatted input prompt
        self.error_signal: float | None = None  # Placeholder for error signals during execution
        self._parents: list[LLMCall] = []
        self._op: str | None = None
        self._children: list[LLMCall] = []
        self.version: int = 0
        self.template_modifiers = template_modifiers or []

    def _get_prompt_function(self) -> PromptFunction[P, R]:
        """Create a PromptFunction using the `prompt` decorator."""
        return prompt(
            self.template, functions=self.functions, model=self.model, output_type=self.output_type
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "LLMCall[P, R]":
        """Call the LLM with the formatted prompt and arguments."""
        # Update format variables with the new arguments
        self.format_variables.update(kwargs)

        # Format the template string with the current format variables
        formatted_template = self.template.format(**self.format_variables)

        # Create a prompt function with the formatted template and call it
        prompt_function = self._get_prompt_function()
        self.output = prompt_function(formatted_template, **self.format_variables)
        self.input = formatted_template
        return self

    def execute(self) -> Any:
        """Executes the LLM call and returns the output."""
        if self.output is None:
            self.__call__()

        if self._op == "+":
            # Concatenate outputs of parents
            self.output = " ".join(str(parent.execute()) for parent in self._parents)
        elif self._op == "*":
            # Intersection for lists
            if all(isinstance(parent.output, list) for parent in self._parents):
                self.output = list(set(self._parents[0].output) & set(self._parents[1].output))
            else:
                raise TypeError("Multiplication is only defined for list outputs.")
        elif self._op == "-":
            # Subtraction: Return first parent's output for now
            self.output = self._parents[0].execute()
        elif self._op == "/":
            # Division: Implement your logic here
            # For now, just return the first parent's output as a placeholder
            self.output = self._parents[0].execute()
        else:
            return self.output

        return self.output

    def __add__(self, other: "LLMCall") -> "LLMCall":
        return self.combine(other, "+")

    def __sub__(self, other: "LLMCall") -> "LLMCall":
        return self.combine(other, "-")

    def __mul__(self, other: "LLMCall") -> "LLMCall":
        return self.combine(other, "*")

    def __truediv__(self, other: "LLMCall") -> "LLMCall":
        return self.combine(other, "/")

    def combine(self, other: "LLMCall", op: str) -> "LLMCall":
        combined = LLMCall(
            template=self.template,
            functions=self.functions,
            model=self.model,
            output_type=self.output_type,
            format_variables=self.format_variables,
        )
        combined._parents = [self, other]
        combined._op = op

        self._children.append(combined)
        if isinstance(other, LLMCall):
            other._children.append(combined)

        return combined

    def _filter_text_by_concepts(self, text: str, concepts: List[str]) -> str:
        """Placeholder for filtering text based on concepts."""
        # Implement your logic here
        return text

    def update_template(self, new_template: str):
        """Updates the template and increments the version."""
        self.template = new_template
        self.version += 1

    def modify_template(self, modification_type: str, **kwargs):
        """Modifies the template based on the modification type."""
        if modification_type == "add_instruction":
            self.template += " " + kwargs["instruction"]
        elif modification_type == "replace_keyword":
            self.template = self.template.replace(
                kwargs["old_keyword"], kwargs["new_keyword"]
            )
        # Add more modification types as needed
        self.version += 1

    def evaluate_output(self) -> float:
        """Placeholder for evaluating the output and returning an error signal."""
        # Implement your evaluation logic here
        # This could be based on external feedback, internal consistency checks, etc.
        # For now, we just return a random error signal
        return random.uniform(0, 1)

    def backward(self, error_signal: Any = None):
        """Backpropagation logic to adjust based on error signals."""
        logger.info(f"Backpropagating through {self._op} operation, error_signal: {error_signal}")

        if error_signal is None:
            error_signal = self.evaluate_output()

        if self._op:
            if self._op == "+":
                if error_signal > 0.5:
                    self.update_template(self.template + " Be more concise and factual.")
                    self.output = self._get_prompt_function()(**self.format_variables)
                    logger.info(f"Re-executed prompt with updated template. New output: {self.output}")

            elif self._op == "/":
                if error_signal > 0.5:
                    if len(self._parents) == 2:
                        numerator_output = self._parents[0].execute()
                        denominator_output = self._parents[1].execute()

                        if isinstance(numerator_output, str) and isinstance(denominator_output, list):
                            filtered_output = self._filter_text_by_concepts(numerator_output, denominator_output)
                            self.output = filtered_output
                            logger.info(f"Filtered output based on concepts. New output: {self.output}")
                        else:
                            logger.warning("Division operation not defined for current output types.")

        for parent in self._parents:
            parent.backward(error_signal)

    def __repr__(self) -> str:
        return f"<LLMCall - Output: {self.output}>"

class TemplateModifier:
    def __init__(self, base_template: str):
        self.base_template = base_template

    def suggest_modifications(self) -> list[tuple[str, dict[str, str]]]:
        """Suggests possible modifications to the template."""
        modifications = []

        # Example modifications:
        # 1. Add an instruction
        modifications.append(
            (
                "add_instruction",
                {"instruction": " Be more concise and factual."},
            )
        )
        modifications.append(
            (
                "add_instruction",
                {"instruction": " Use simple and direct language."},
            )
        )

        # 2. Replace a keyword (example)
        modifications.append(
            (
                "replace_keyword",
                {"old_keyword": "{text}", "new_keyword": "{input_text}"},
            )
        )

        return modifications

    def apply_modification(
        self, template: str, modification_type: str, **kwargs: Any
    ) -> str:
        """Applies a given modification to the template."""
        if modification_type == "add_instruction":
            return template + kwargs["instruction"]
        elif modification_type == "replace_keyword":
            return template.replace(kwargs["old_keyword"], kwargs["new_keyword"])
        else:
            raise ValueError(f"Unknown modification type: {modification_type}")

# Example usage with asyncio
async def main():
    # Initialize the LLM and the first prompt template
    llm_call1 = LLMCall("Translate '{text}' to French.")
    llm_call2 = LLMCall("Summarize this text: {text}.")
    llm_call3 = LLMCall("What is the sentiment of this text: {text}?")

    # Combine LLM calls using arithmetic operators
    combined_call = (llm_call1 + llm_call2) * llm_call3

    # Initial input
    input_text = "This is a test."

    # Create a template modifier
    modifier = TemplateModifier(llm_call1.template)

    # Main loop
    for iteration in range(3):
        print(f"Iteration {iteration + 1}")

        # Execute the combined call with specific inputs
        result = combined_call(
            text=input_text,
            summary="Provide a concise summary.",
            sentiment="Positive",
        )
        result.execute()

        print(f"Output: {result.output}")

        # Simulate an error signal based on the output or further downstream tasks
        error_signal = random.uniform(0, 1)

        # Perform backpropagation
        combined_call.backward(error_signal)

        # Randomly suggest and apply modifications
        if error_signal > 0.5:
            modifications = modifier.suggest_modifications()
            mod_type, mod_kwargs = random.choice(modifications)

            # Randomly pick a call to modify
            call_to_modify = random.choice([llm_call1, llm_call2, llm_call3])

            call_to_modify.modify_template(mod_type, **mod_kwargs)
            print(f"Modified template of {call_to_modify} with: {mod_type} - New template: {call_to_modify.template}")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
