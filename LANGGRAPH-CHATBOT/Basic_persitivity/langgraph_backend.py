from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv


load_dotenv()

llm= ChatOpenAI()
# DEFINING THE STATE OF THE CHATBOT
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    

# DEFINING THE FUNCTION

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# adding checkpointer for persistivity in the chatbot
checkpointer = InMemorySaver()

# DEFINING THE GRAPH

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)