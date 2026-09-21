from typing import Any, ClassVar

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    hook_config,
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command

load_dotenv()
model = init_chat_model("anthropic:claude-sonnet-4.5")

# --- Healthcare-specific content filter ---
class HealthcareSafetyFilter(AgentMiddleware):
    """Block non-medical or harmful requests in a healthcare context."""

    BLOCKED_TOPICS: ClassVar[list[str]] = [
        "drug synthesis", 
        "self-harm", 
        "suicide method", 
        "weapon", 
        "hack"
    ]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_msg = state["messages"][0]
        if first_msg.type != "human":
            return None

        content = first_msg.content
        
        if isinstance(content, str):
            str_first_msg = content
        else:
            str_first_msg = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        content = str_first_msg.lower()
        for topic in self.BLOCKED_TOPICS:
            if topic in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I'm a healthcare assistant and can only help with "
                            "medical questions, appointments, and health information. "
                            "If you're in crisis, please call 112 or your local emergency number."
                        )
                    }],
                    "jump_to": "end"
                }
        return None


# --- Medical output validator ---
class MedicalOutputValidator(AgentMiddleware):
    """Ensure all responses include appropriate medical disclaimers."""

    DISCLAIMER = "\n\n⚕️ *This is general health information, not medical advice. Please consult a qualified healthcare professional.*"

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        content = last_message.content

        if isinstance(content, str):
            str_last_message = content
        else:
            str_last_message = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        # Add disclaimer if not already present
        if "medical advice" not in str_last_message.lower():
            last_message.content += self.DISCLAIMER

        return None


# --- Healthcare tools ---
@tool
def search_symptoms(symptoms: str) -> str:
    """Search for information about medical symptoms."""
    return f"Symptom information for: {symptoms}. Please consult a doctor for diagnosis."

@tool
def book_appointment(patient_name: str, date: str, doctor: str) -> str:
    """Book a medical appointment."""
    return f"Appointment booked for {patient_name} with Dr. {doctor} on {date}"

@tool
def get_medication_info(medication: str) -> str:
    """Get information about a medication."""
    return f"General info about {medication}. Always follow your doctor's prescription."


# --- Build the healthcare chatbot ---
healthcare_bot = create_agent(
    model=model,
    tools=[search_symptoms, book_appointment, get_medication_info],
    middleware=[
        # Guardrail 1: Block harmful/off-topic requests
        HealthcareSafetyFilter(),

        # Guardrail 2: Redact patient PII from inputs
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Guardrail 3: Require approval before booking appointments
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_appointment": True,
                "search_symptoms": False,
                "get_medication_info": False,
            }
        ),

        # Guardrail 4: Add medical disclaimer to all outputs
        MedicalOutputValidator(),
    ],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "You are a helpful healthcare assistant. "
        "You can search for symptoms, medication information, and help book appointments. "
        "Always be empathetic and remind users to consult a doctor for diagnosis."
    )
)

print("🏥 Healthcare chatbot with full guardrail stack created!")

# Test 1: Safe medical query
config_t1: RunnableConfig = {"configurable": {"thread_id": "healthcare_session_t1"}}

result = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": "What are symptoms of Type 2 Diabetes?"}]},
    config=config_t1
)

print(result["messages"][-1].content)


# Test 2: Query with PII (email gets redacted)
result = healthcare_bot.invoke({
    "messages": [{
        "role": "user",
        "content": "My email is patient123@gmail.com. What can I take for a headache?"
    }]},
    config=config_t1
)
print("=== PII Redaction Test ===")
print(result["messages"][-1].content)


# Test 3: Off-topic / harmful request — gets blocked
result = healthcare_bot.invoke({
    "messages": [{"role": "user", "content": "How do I synthesize drugs at home?"}]
},
 config=config_t1)
print("=== Blocked Request ===")
print(result["messages"][-1].content)


# Test 4: Appointment booking — requires human approval
config_t2: RunnableConfig = {"configurable": {"thread_id": "healthcare_session_001"}}

result = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": "Book me an appointment with Dr. Sharma on March 15"}]},
    config=config_t2
)
print("=== Appointment Booking — Awaiting Approval ===")
print(result["messages"][-1].content)

# Approve
approved = healthcare_bot.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config_t2
)
print("\n=== After Approval ===")
print(approved["messages"][-1].content)