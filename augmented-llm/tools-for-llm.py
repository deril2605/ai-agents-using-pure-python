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

def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

messages = [
    {"role": "system", "content": "You are a helpful weather assistant."},
    {"role": "user", "content": "What's the weather like in Paris today?"},
]

completion = client.chat.completions.create(
    model=os.getenv("AZURE_DEPLOYMENT_NAME"),
    messages=messages,
    tools=tools,
)

#print(completion)
# up until here - the llm stops with finish_reason = tool_calls it extracts lat and long and knows which too/function to call. But it hasn't called the tool yet
# Why Does the LLM Stop at finish_reason = tool_calls?

# ✅ What’s working:
# The LLM correctly interprets the user's request: "What's the weather like in Paris today?"
# It understands that it needs to call the get_weather function.
# It extracts the necessary parameters (latitude and longitude) for Paris.
# It correctly marks finish_reason = tool_calls, indicating that the model has determined the function it needs to call but has not executed it yet.

# ❌ What's missing?
# The LLM does not automatically execute the tool/function.
# Azure OpenAI (and OpenAI's API in general) only returns the tool/function call information.
# You need to manually execute get_weather() using the extracted parameters and then send a follow-up message with the result.

# print(completion.choices[0].message)
# ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_IA6jNkwCWHVMpehp25Krtj0L', function=Function(arguments='{"latitude":48.8566,"longitude":2.3522}', name='get_weather'), type='function')]

def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    
for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args) # this is just an api call - no AI involved 
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )

completion_2 = client.beta.chat.completions.parse(
    model=os.getenv("AZURE_DEPLOYMENT_NAME"),
    messages=messages,
    tools=tools,
    response_format=WeatherResponse,
)

final_response = completion_2.choices[0].message.parsed
print(final_response.temperature)
print()
print(final_response.response)