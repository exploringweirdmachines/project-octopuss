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

P = ParamSpec("P")
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
        llama_intention: "LLamaIntention" | None = None,
        text_chunker: "TextChunker" | None = None,  # Add TextChunker
    ):
        self.template = template
        self.functions = functions or []
        self.model = model or get_chat_model()
        self.output_type = output_type or str
        self.format_variables = format_variables or {}
        self.output: str | None = None  # Store the output of the LLM call
        self.input: str | None = None  # Store the formatted input prompt
        self.error_signal: float | None = None  # Placeholder for error signals during execution
        self._parents: list["LLMCall"] = []
        self._op: str | None = None
        self._children: list["LLMCall"] = []
        self.version: int = 0
        self.template_modifiers = template_modifiers or []
        self.llama_intention = llama_intention
        self.text_chunker = text_chunker

    def _get_prompt_function(self) -> PromptFunction[P, R]:
        """Create a PromptFunction using the `prompt` decorator."""
        return prompt(
            self.template, functions=self.functions, model=self.model, output_type=self.output_type
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "LLMCall[P, R]":
        """Call the LLM with the formatted prompt and arguments."""
        logger.info(f"Entering __call__ for template: {self.template}, version: {self.version}")
        self.format_variables.update(kwargs)

        formatted_template = self.template.format(**self.format_variables)

        # Apply text chunking if a chunker is provided and the input text exists
        if self.text_chunker and "text" in self.format_variables:
            chunk_result = self.text_chunker.chunk(self.format_variables["text"])
            self.format_variables["chunks"] = chunk_result["chunks"]  # Add chunks to format variables
            formatted_template = self.template.format(**self.format_variables)  # Re-format

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
            # Division: Use LlamaIntention to filter concepts
            if len(self._parents) == 2 and isinstance(self._parents[0].output, str) and isinstance(self._parents[1].output, list):
                if self.llama_intention:
                    self.output = self.llama_intention.filter_text_by_concepts(self._parents[0].execute(), self._parents[1].execute())
                else:
                    logger.warning("Division operation not defined for current output types or no LlamaIntention provided.")
            else:
                raise ValueError("Division operation not defined for current output types or requires LlamaIntention.")
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
            llama_intention=self.llama_intention,
            text_chunker=self.text_chunker,
        )
        combined._parents = [self, other]
        combined._op = op

        self._children.append(combined)
        if isinstance(other, LLMCall):
            other._children.append(combined)

        return combined

    def _filter_text_by_concepts(self, text: str, concepts: List[str]) -> str:
        """Filters text based on the presence of given concepts."""
        filtered_text = []
        for sentence in text.split("."):
            sentence = sentence.strip()
            if any(concept.lower() in sentence.lower() for concept in concepts):
                filtered_text.append(sentence)
        return ". ".join(filtered_text)

    def update_template(self, new_template: str):
        """Updates the template and increments the version."""
        self.template = new_template
        self.version += 1

    def modify_template(self, modification_type: str, **kwargs: Any):
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
                    # Re-execute the prompt with an updated template
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

        # Propagate the error signal to parent nodes
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

import abc
import numpy as np

class LLamaIntention(abc.ABC):
    """
    A base class for defining intentions to be executed using a Llama language model.
    This class provides the core functionality for interacting with the model and
    analyzing its output to achieve a specific task.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = 17,
        seed: int = 42
    ):
        """
        Initializes the LLamaIntention with a language model.

        Args:
            model_path: Path to the Llama model file (e.g., .gguf).
            n_ctx: The maximum context window size for the model.
            n_gpu_layers: The number of layers to offload to the GPU.
            seed: Random seed for reproducibility.
        """
        self.llm = Llama(
            model_path=model_path,
            logits_all=True,  # Need logits for all positions
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed
        )
        self._token_to_id = {
            self.llm.detokenize([i]).decode("utf-8", errors='ignore'): i
            for i in range(self.llm.n_vocab())
        }

    @abc.abstractmethod
    def format_prompt(self, text: str) -> str:
        """
        Formats the input text into a prompt suitable for the language model.
        Subclasses should implement this method to define the specific prompt
        structure required for their intention.

        Args:
            text: The input text to be processed.

        Returns:
            The formatted prompt string.
        """
        pass

    @abc.abstractmethod
    def get_target_token(self) -> str:
        """
        Returns the target token whose log probabilities will be analyzed.
        Subclasses should implement this method to specify the token relevant
        to their intention (e.g., a split token, an end-of-sentence token, etc.).

        Returns:
            The target token string.
        """
        pass

    def get_target_token_id(self) -> int:
        """
        Retrieves the vocabulary ID of the target token.

        Returns:
            The integer ID of the target token in the model's vocabulary.
        """
        target_token = self.get_target_token()
        if target_token in self._token_to_id:
            return self._token_to_id[target_token]
        raise ValueError(f"Could not find '{target_token}' in model vocabulary")

    def get_logprobs(self, text: str) -> np.ndarray:
        """
        Gets the log probabilities of the target token at each position in the
        model's output after processing the formatted prompt.

        Args:
            text: The input text to be processed.

        Returns:
            A NumPy array of log probabilities for the target token at each position.
        """
        prompt = self.format_prompt(text)
        input_tokens = self.llm.tokenize(prompt.encode("utf-8"), special=True)

        # Add the target token to complete the sequence for logprob calculation
        target_token = self.get_target_token()
        target_token_input = self.llm.tokenize(target_token.encode("utf-8"), special=True)
        input_tokens.extend(target_token_input)

        self.llm.eval(input_tokens)

        target_token_id = self.get_target_token_id()
        logprobs = []
        for i in range(len(input_tokens)):
            scores = self.llm.scores[i]
            log_probs = self.llm.logits_to_logprobs(scores)
            logprobs.append(float(log_probs[target_token_id]))

        return np.array(logprobs)

    def normalize_logprobs(self, logprobs: np.ndarray, window: int = 100) -> np.ndarray:
        """
        Normalizes log probabilities by subtracting a local rolling mean. This can
        help to identify significant peaks or drops in probability relative to the
        surrounding context.

        Args:
            logprobs: The array of log probabilities to normalize.
            window: The size of the rolling window for calculating the mean.

        Returns:
            A NumPy array of normalized log probabilities.
        """
        if len(logprobs) < window:
            return logprobs - np.mean(logprobs)

        kernel_size = min(window, len(logprobs))
        kernel = np.ones(kernel_size) / kernel_size
        rolling_mean = np.convolve(logprobs, kernel, mode='same')
        return logprobs - rolling_mean

    @abc.abstractmethod
    def analyze_logprobs(self, text: str, logprobs: np.ndarray) -> Any:
        """
        Analyzes the log probabilities to determine the outcome of the intention.
        Subclasses should implement this method to interpret the log probabilities
        in the context of their specific task.

        Args:
            text: The original input text.
            logprobs: The array of log probabilities for the target token.

        Returns:
            The result of the analysis, which can be of any type depending on the
            intention (e.g., a list of split points, a summary, etc.).
        """
        pass

    def execute(self, text: str) -> Any:
        """
        Executes the defined intention on the given text. This involves formatting
        the prompt, getting log probabilities, and analyzing them to produce the
        final result.

        Args:
            text: The input text to be processed.

        Returns:
            The result of executing the intention.
        """
        if not text.strip():
            return self.handle_empty_input()

        logprobs = self.get_logprobs(text)
        return self.analyze_logprobs(text, logprobs)

    def handle_empty_input(self) -> Any:
        """
        Handles the case where the input text is empty. Subclasses can override
        this method to provide specific behavior for empty input.

        Returns:
            The result to return when the input is empty.
        """
        return None

from scipy.signal import find_peaks

class LlamaAbstractor(LLamaIntention):
    """
    Identifies potential abstract functions (ideas) in text based on log probabilities.
    """

    def __init__(
        self,
        model_path: str = "resources/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        n_ctx: int = 8192,
        n_gpu_layers: int = 17,
        seed: int = 42,
        target_token: str = "\n",  # Analyze logprobs of newlines or sentence ends
        min_tokens: int = 10,  # Minimum tokens for an idea to be considered
        logprob_threshold: float = 0.3,  # Threshold for logprob peak
        min_distance: int = 5  # Minimum token distance between potential abstractions
    ):
        self.target_token = target_token
        self.min_tokens = min_tokens
        self.logprob_threshold = logprob_threshold
        self.min_distance = min_distance
        self.prompt_format = """<|start_header_id|>system<|end_header_id|>
You are an expert text analyst. Your task is to identify sections of the following text that represent distinct ideas or concepts that could be considered as abstract functions or modules.

Analyze the provided text and indicate potential boundaries for these abstract ideas. Focus on logical blocks of text that convey a specific thought or topic.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        super().__init__(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, seed=seed)

    def format_prompt(self, text: str) -> str:
        return self.prompt_format.format(text=text)

    def get_target_token(self) -> str:
        return self.target_token

    def analyze_logprobs(self, text: str, logprobs: np.ndarray) -> List[Tuple[int, int]]:
        """
        Analyzes log probabilities to identify potential abstract functions (ideas).

        Returns:
            A list of tuples, where each tuple contains the start and end character
            indices of a potential abstract function.
        """
        norm_logprobs = self.normalize_logprobs(logprobs)
        peaks, _ = find_peaks(norm_logprobs, height=self.logprob_threshold, distance=self.min_distance)

        potential_abstraction_points = []
        tokens = self.llm.tokenize(text.encode("utf-8"), special=True)
        for peak_index in peaks:
            if peak_index < len(tokens):
                try:
                    token_text = self.llm.detokenize([tokens[peak_index]]).decode("utf-8")
                    if token_text == self.target_token:
                        potential_abstraction_points.append(peak_index)
                except IndexError:
                    pass

        abstraction_ranges = []
        start_index_token = 0
        for end_index_token in potential_abstraction_points:
            if end_index_token < len(tokens):
                end_index_char = self._token_index_to_char_index(text, end_index_token)
                start_index_char = self._token_index_to_char_index(text, start_index_token)

                num_tokens = end_index_token - start_index_token
                if num_tokens >= self.min_tokens:
                    abstraction_ranges.append((start_index_char, end_index_char))
                    start_index_token = end_index_token + 1 # Move start for next potential abstraction

        return abstraction_ranges

    def _token_index_to_char_index(self, text: str, token_index: int) -> int:
        """Helper function to convert token index to character index."""
        tokens = self.llm.tokenize(text.encode("utf-8"), special=True)
        if token_index >= len(tokens):
            return len(text)  # Handle edge case

        char_count = 0
        for i in range(token_index):
            try:
                char_count += len(self.llm.detokenize([tokens[i]]).decode("utf-8"))
            except UnicodeDecodeError:
                pass # Handle potential decode errors
        return char_count

    def handle_empty_input(self) -> List[Tuple[int, int]]:
        return []

class TextChunker:
    """
    Orchestrates the process of calculating next byte entropies and chunking text
    based on patch boundaries.
    """

    def __init__(self, language_model=None, global_threshold=None, relative_threshold=None):
        self.global_threshold = global_threshold
        self.relative_threshold = relative_threshold
        self.byte_pair_probs = None
        self.language_model = language_model if language_model else self._default_language_model

    def train(self, corpus):
        """
        Trains the language model on the given corpus to estimate byte pair probabilities.
        """
        byte_counts = defaultdict(int)
        pair_counts = defaultdict(int)

        for text in corpus:
            encoded_text = text.encode('utf-8')
            for byte in encoded_text:
                byte_counts[byte] += 1
            for i in range(len(encoded_text) - 1):
                pair = (encoded_text[i], encoded_text[i + 1])
                pair_counts[pair] += 1

        total_bytes = sum(byte_counts.values())
        self.byte_pair_probs = {}
        for pair, count in pair_counts.items():
            # Calculate the probability of the second byte given the first
            first_byte_count = byte_counts[pair[0]]
            if first_byte_count > 0:
                self.byte_pair_probs[pair] = count / first_byte_count

    def _default_language_model(self, previous_bytes, next_byte):
        """
        Uses the trained byte pair probabilities to estimate the probability of the next byte.
        """
        if not previous_bytes:
            # If no previous bytes, use a uniform distribution or some prior
            return 1 / 256.0

        previous_byte = previous_bytes[-1]
        prob = self.byte_pair_probs.get((previous_byte, next_byte), 0.00001) # Add a small default prob
        return prob

    def calculate_entropies(self, text):
        """
        Calculates the next byte entropies for a given text using the trained language model.
        """
        entropies = []
        encoded_text = text.encode('utf-8')
        for i in range(1, len(encoded_text)):
            previous_bytes = list(encoded_text[:i])
            current_byte = encoded_text[i]
            prob = self.language_model(previous_bytes, current_byte)
            if prob > 0:
                entropy = -prob * math.log2(prob)
                entropies.append(entropy)
        return entropies

    def identify_patch_boundaries(self, entropies):
        """
        Identifies patch boundaries in a sequence of byte entropies using two methods.
        """
        global_boundaries = []
        relative_boundaries = []

        # Method 1: Global Threshold
        if self.global_threshold is not None:
            for i, entropy in enumerate(entropies):
              if entropy > self.global_threshold:
                 global_boundaries.append(i+1) # Add 1 to match the original index in text

        # Method 2: Approximate Monotonic Constraint
        if self.relative_threshold is not None and len(entropies) > 1:
            for i in range(1, len(entropies)):
              if entropies[i] - entropies[i - 1] > self.relative_threshold:
                relative_boundaries.append(i+1) # Add 1 to match the original index in text

        return global_boundaries, relative_boundaries

    def chunk_text(self, text, boundaries):
        """Chunks a text string based on provided boundary indices."""
        chunks = []
        sorted_boundaries = sorted(list(set(boundaries)))
        start_index = 0
        for boundary in sorted_boundaries:
            chunks.append(text[start_index:boundary]) # Changed to boundary without +1 to match results
            start_index = boundary
        if start_index < len(text):
            chunks.append(text[start_index:])
        return chunks

    def text_to_hex(self, text):
        return text.encode('utf-8').hex()

    def chunk(self, text):
        entropies = self.calculate_entropies(text)
        global_boundaries, relative_boundaries = self.identify_patch_boundaries(entropies)
        combined_boundaries = sorted(list(set(global_boundaries + relative_boundaries)))
        chunks = self.chunk_text(text, combined_boundaries)
        hex_chunks = [self.text_to_hex(chunk) for chunk in chunks]

        return {
            "text": text,
            "entropies": entropies,
            "global_boundaries": global_boundaries,
            "relative_boundaries": relative_boundaries,
            "combined_boundaries": combined_boundaries,
            "chunks": chunks,
            "hex_chunks": hex_chunks,
        }

async def main():
    # Initialize the LLM
    llm = Llama(
        model_path="/path/to/your/model.gguf", # Add path to model
        n_ctx=8192,
        n_gpu_layers=17,
        seed=42,
    )

    # Initialize LlamaIntention with the model
    llama_intention = LLamaIntention(llm)

    # Initialize TextChunker
    text_chunker = TextChunker(global_threshold=0.6, relative_threshold=0.2)
    # Train the TextChunker (replace with your corpus)
    text_chunker.train(["This is a sentence. This is another one.", "Short text."])

    # Initialize the LLM calls with LlamaIntention and TextChunker
    llm_call1 = LLMCall("Translate '{text}' to French. Here are the text chunks: {chunks}", llama_intention=llama_intention, text_chunker=text_chunker)
    llm_call2 = LLMCall("Summarize this text: {text}.", llama_intention=llama_intention)
    llm_call3 = LLMCall("What is the sentiment of this text: {text}?", llama_intention=llama_intention)

    # Combine LLM calls using arithmetic operators
    # Example of using the division operator
    combined_call = (llm_call1 + llm_call2) / llm_call3

    # Initial input
    input_text = "This is a test. This is more text to test the chunker. And even more."

    # Create a template modifier
    modifier = TemplateModifier(llm_call1.template)

    # Main loop
    for iteration in range(3):  # Increased number of iterations
        print(f"Iteration {iteration + 1}")

        # Execute the combined call with specific inputs
        result = combined_call(
            text=input_text,
            summary="Provide a concise summary.",
            sentiment="Positive",
        )
        result.execute()

        print(f"Output: {result.output}")

        # Get an error signal from LlamaIntention
        error_signal = combined_call.evaluate_output()
        print(f"Error Signal: {error_signal:.2f}")

        # Perform backpropagation
        combined_call.backward(error_signal)

        # Randomly suggest and apply modifications
        if error_signal > 0.5:
            modifications = modifier.suggest_modifications()
            mod_type, mod_kwargs = random.choice(modifications)

            # Randomly pick a call to modify
            call_to_modify = random.choice([llm_call1, llm_call2, llm_call3])

            call_to_modify.modify_template(mod_type, **mod_kwargs)
            print(
                f"Modified template of {call_to_modify} with: {mod_type} - New template: {call_to_modify.template}"
            )

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
