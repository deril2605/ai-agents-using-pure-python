import os
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
import json

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def search_kb(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("augmented-llm/kb.json", "r") as f:
        return json.load(f) # dummy gives back entire KB, we can add filters to make it more RAG like behavior
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(
    model=os.getenv("AZURE_DEPLOYMENT_NAME"),
    messages=messages,
    tools=tools,
)

# print(completion.choices[0].message) 
# ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_Gj15KYt9D8sL7PsyZCMkzQif', function=Function(arguments='{"question":"What is the return policy?"}', name='search_kb'), type='function')])

def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

# print(messages)
# [{'role': 'system', 'content': 'You are a helpful assistant that answers questions from the knowledge base about our e-commerce store.'}, {'role': 'user', 'content': 'What is the return policy?'}, ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_c00LKLMDtc7z7DXqlOUpvxeN', function=Function(arguments='{"question":"What is the return policy?"}', name='search_kb'), type='function')]), {'role': 'tool', 'tool_call_id': 'call_c00LKLMDtc7z7DXqlOUpvxeN', 'content': '{"records": [{"id": 1, "question": "What is the return policy?", "answer": "Items can be returned within 30 days of purchase with original receipt. Refunds will be processed to the original payment method within 5-7 business days."}, {"id": 2, "question": "Do you ship internationally?", "answer": "Yes, we ship to over 50 countries worldwide. International shipping typically takes 7-14 business days and costs vary by destination. Please note that customs fees may apply."}, {"id": 3, "question": "What payment methods do you accept?", "answer": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. All payments are processed securely through our encrypted payment system."}]}'}]

class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question and make it more creative and funny.")
    source: int = Field(description="The record id of the answer.")

completion_2 = client.beta.chat.completions.parse(
    model=os.getenv("AZURE_DEPLOYMENT_NAME"),
    messages=messages,
    tools=tools,
    response_format=KBResponse,
)

final_response = completion_2.choices[0].message.parsed
print(final_response.answer)
print(final_response.source)