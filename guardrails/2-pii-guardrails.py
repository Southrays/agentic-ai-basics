from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.5")

#Define a simple dummy tool
@tool
def customer_lookup(query: str) -> str:
    """Lookup customer information"""
    return f"Customer record found for query: {query}"

#Create agent with PII Middleware
agent = create_agent(
    model=model,
    tools=[customer_lookup],
    middleware=[
        #Redacts email in user input before sending it to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True
        ),

        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),

        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        )
    ]
)

print("Agent with PII middleware created succefully")

#Test PII Redaction
result = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "My email is john.doe@example.com and my card is 5105-1051-0510-5100. Can you help me?" 
    }]
})

print("=== Agent Response ===")
print(result["messages"][-1].content)


# Test API Key Blocking
try:
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Here is my key: sk-abcdefghijklmnopqrstuvwxyz123456"
        }]
    })
    
except Exception as e:
    print(f"🚫 Blocked as expected: {e}")
    raise