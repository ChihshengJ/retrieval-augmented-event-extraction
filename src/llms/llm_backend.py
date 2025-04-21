from abc import ABC, abstractmethod
import subprocess
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import time

completion_tokens = prompt_tokens = 0

CLAUDEURL= 'https://api.anthropic.com/v1/messages'


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

        with open('/data/cjin/retrieval-augmented-event-extraction/src/llms/input.json', 'w') as f:
            json.dump(params, f)

        try:
            result = subprocess.run(
                ["conda", "run", "-n", "openai_env", "python", "/data/cjin/retrieval-augmented-event-extraction/src/llms/openai_calling.py"],
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


class ClaudeCompletion(CompletionAPI):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def _call_claude(self, url, data, headers, delay=20):
        retry = Retry(
            total=5,
            backoff_factor=20,
            status_forcelist=[429, 500, 529],  # Only retry on rate limits and server errors
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount('https://', adapter)

        try:
            response = session.post(
                url,
                headers=headers,
                json=data
            )
            # print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                return str(response.json()['content'][0]['text'])

            error_msg = response.json()
            print(f"Error Details: {error_msg}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

        finally:
            time.sleep(delay)

    def get_completion(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        global completion_tokens, prompt_tokens

        url = CLAUDEURL
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        data = {
            "model": kwargs.get("model", "claude-3-7-sonnet-2025021"),
            "system": system_prompt,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        return self._call_claude(url, data, headers)


class CompletionAPIFactory:
    @staticmethod
    def get_api(api_name: str, **kwargs) -> CompletionAPI:
        if api_name == "openai":
            return OpenAICompletion(api_key=kwargs["api_key"])
        if api_name == "claude":
            return ClaudeCompletion(api_key=kwargs["api_key"])
        raise ValueError(f"Unsupported API: {api_name}")
