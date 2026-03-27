from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import asyncio

# Load env variables
load_dotenv()

# ✅ Enable streaming
llm = ChatOpenAI(
    streaming=True,
    temperature=0.7
)

# ==============================
# 🧠 STATE DEFINITION
# ==============================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ==============================
# 🤖 CHAT NODE (STREAMING)
# ==============================
def chat_node(state: ChatState):
    messages = state["messages"]

    response_chunks = []
    print("\n🤖 Assistant: ", end="", flush=True)

    for chunk in llm.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)  # live streaming output
        response_chunks.append(chunk)

    print("\n")

    return {"messages": response_chunks}

# ==============================
# 💾 MEMORY (PERSISTENCE)
# ==============================
checkpointer = InMemorySaver()

# ==============================
# 🔗 BUILD GRAPH
# ==============================
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# ==============================
# 💬 CLI CHAT LOOP
# ==============================
def run_chat():
    thread_id = "user_1"

    print("🔥 Streaming Chatbot Started (type 'exit' to quit)\n")

    while True:
        user_input = input("👤 You: ")

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Send user message
        events = chatbot.stream(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config={"configurable": {"thread_id": thread_id}}
        )

        # Trigger execution (stream already prints response)
        for _ in events:
            pass


# ==============================
# ⚡ OPTIONAL: ASYNC VERSION (for FastAPI)
# ==============================
async def async_chat(user_input: str, thread_id="user_1"):
    async for event in chatbot.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}}
    ):
        for value in event.values():
            if "messages" in value:
                for msg in value["messages"]:
                    if msg.content:
                        yield msg.content


# ==============================
# ▶️ RUN
# ==============================
if __name__ == "__main__":
    run_chat()