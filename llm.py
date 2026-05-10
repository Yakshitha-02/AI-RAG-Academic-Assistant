from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class LLM:

    def __init__(self):

        self.client = OpenAI(

            base_url="https://openrouter.ai/api/v1",

            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    # -----------------------------------
    # Generate Answer
    # -----------------------------------

    def generate_answer(
        self,
        query,
        context
    ):

        prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY from the given context.

If answer is not available in context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{query}

Answer:
"""

        completion = self.client.chat.completions.create(

            model="openrouter/auto",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        answer = completion.choices[0].message.content

        return answer