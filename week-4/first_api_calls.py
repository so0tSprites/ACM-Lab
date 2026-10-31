from openrouter import OpenRouter
import os
from dotenv import load_dotenv

load_dotenv()

with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
    response = client.responses.send(
        model="google/gemma-4-26b-a4b-it:free",
        input="how many r's are in the word strawberry?",
    )

    if response.status != "completed":
        message = response.error.message if response.error else f"Response ended as {response.status}"
        raise RuntimeError(message)

    message = next(item for item in response.output if item.type == "message")
    print(next(part.text for part in message.content if part.type == "output_text"))