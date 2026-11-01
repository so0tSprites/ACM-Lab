from openrouter import OpenRouter
import os
from dotenv import load_dotenv

load_dotenv()

with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
    response = client.responses.send(
        model="meta-llama/llama-3.1-8b-instruct",
        input="list the top 5 skills needed to chop an onion properly",
        max_output_tokens = 1000,
    )

    if response.status != "completed":
        message = response.error.message if response.error else f"Response ended as {response.status}"
        raise RuntimeError(message)

    message = next(item for item in response.output if item.type == "message")
    print(next(part.text for part in message.content if part.type == "output_text"))

'''OUTPUT:
A crucial life skill! Here are the top 5 skills needed to chop an onion properly:

1. **Knife control**: The ability to hold a knife with confidence and control is essential for chopping an onion. This involves a firm, yet gentle grip on the knife, with the blade facing the correct direction (usually away from your body).
2. **Chopping technique**: A smooth, rocking motion is key to chopping an onion effectively. This involves using a gentle, sawing motion to separate the onion's cells, rather than applying too much pressure and crushing the onion.
3. **Onion understanding**: To chop an onion properly, you need to understand its structure and how to navigate its layers. This involves identifying the onion's natural "grain" or lines of separation, and chopping along those lines to minimize tears and maximize evenness.
4. **Eye-hand coordination**: Chopping an onion requires a delicate balance between the knife, the onion, and your eyes. You need to be able to track the knife's movement, gauge the onion's resistance, and make adjustments on the fly to avoid accidents.
5. **Tear control**: Let's face it, chopping an onion can be a tear-inducing activity! To minimize tears, you need to develop the skill of chopping quickly and efficiently, without applying too much pressure or causing the onion's cells to rupture. This involves developing a smooth, flowing motion, and using techniques like "pinching" the onion or using a very sharp knife to reduce tear-inducing irritants.

By mastering these skills, you'll be well on your way to becoming an onion-chopping pro!'''