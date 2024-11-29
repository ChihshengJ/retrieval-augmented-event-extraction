import openai
from openai import OpenAIError, RateLimitError, APIError
from openai.types.chat import ChatCompletionMessageParam
from abc import ABC, abstractmethod

completion_tokens = prompt_tokens = 0


class CompletionAPI(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, system_prompt: str, **kwargs) -> str:
        "Abstract method to get the completion from a prompt"
        pass


class OpenAICompletion(CompletionAPI):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def get_completion(self, prompt: str, system_prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        global completion_tokens, prompt_tokens

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt
            }
        ]
        try:
            response = openai.chat.completions.create(
                model=kwargs.get("model", "gpt-4o"),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # print('testing_response', response.choices)
            completion_tokens += response.usage.completion_tokens
            prompt_tokens += response.usage.prompt_tokens
            return response.choices[0].message.content
        except RateLimitError:
            print("Rate limit exceeded, check usage panel.")
            return None
        except Exception as e:
            print(f"An unexpected error {e} occurred when calling OpenAI's API.")
            return None 


class CompletionAPIFactory:
    @staticmethod
    def get_api(api_name: str, **kwargs) -> CompletionAPI:
        if api_name == "openai":
            return OpenAICompletion(api_key=kwargs["api_key"])
        else:
            raise ValueError(f"Unsupported API: {api_name}")
