from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.5")

class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    Model-based guardrail: Use an LLM to evaluate response safety.
    Runs AFTER the agent produces a response, before it reaches the user.
    """

    def __init__(self):
        super().__init__()
        self.safety_model = model

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a lightweight model as the safety judge
        safety_prompt = f"""Evaluate if this AI response is safe and appropriate for users.
        Respond with only 'SAFE' or 'UNSAFE'.

        Response to evaluate:
        {last_message.content}"""

        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        content = result.content

        if isinstance(content, str):
            str_result = content
        else:
            str_result = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        if "UNSAFE" in str_result.upper():
            print("⚠️  Output flagged as UNSAFE — replacing with safe fallback")
            last_message.content = (
                "I'm unable to provide that response. "
                "Please rephrase your request or contact support."
            )

        return None


@tool
def general_tool(query: str) -> str:
    """A general purpose tool."""
    return f"Tool result: {query}"


safe_agent = create_agent(
    model=model,
    tools=[general_tool],
    middleware=[SafetyGuardrailMiddleware()],
)

print("Output safety agent created!")

# Test output safety check
result = safe_agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like today?"}]
})
print("Response:")
print(result["messages"][-1].content)
