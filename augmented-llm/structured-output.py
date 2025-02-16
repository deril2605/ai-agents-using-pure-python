import os
from openai import AzureOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model=os.getenv("AZURE_DEPLOYMENT_NAME"),
    messages=[
        {"role": "system", "content": "Extract the event information where the person works from office."},
        {
            "role": "user",
            "content": "Deril generally goes to office on Tuesday and Thursday and work from home on other days",
        },
    ],
    response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed
event.name
event.date
event.participants

print(event)
