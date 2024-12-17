from abc import ABC, abstractmethod
import subprocess
import json

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

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt
            }
        ]

        params = {
            "model": kwargs.get("model", "gpt-4o"),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        with open('./llms/input.json', 'w') as f:
            json.dump(params, f)

        try:
            result = subprocess.run(
                ["conda", "run", "-n", "openai_env", "python", "llms/openai_calling.py"],
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                print("Error calling OpenAI API handler:", result.stderr)
                return None

            # Parse the response from the handler script
            output = json.loads(result.stdout)
            return output['response']
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return None


class CompletionAPIFactory:
    @staticmethod
    def get_api(api_name: str, **kwargs) -> CompletionAPI:
        if api_name == "openai":
            return OpenAICompletion(api_key=kwargs["api_key"])
        else:
            raise ValueError(f"Unsupported API: {api_name}")
