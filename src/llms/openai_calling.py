import sys
import json
from openai import OpenAIError
import openai


def call_openai_api():
    try:
        with open("./llms/input.json", "r") as f:
            params = json.load(f)
        response = openai.chat.completions.create(**params)
        return {"success": True, "response": response.choices[0].message.content}
    except OpenAIError as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = call_openai_api()

    # Send the result back to stdout
    print(json.dumps(result))
