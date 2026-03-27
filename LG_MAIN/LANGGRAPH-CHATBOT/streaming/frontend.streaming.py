import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Import your chatbot graph
from your_backend_file import chatbot   # 👈 replace with your filename

load_dotenv()

st.set_page_config(page_title="AI Chatbot", layout="wide")

# ==============================
# 🧠 SESSION STATE (CHAT MEMORY)
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_1"

# ==============================
# 🎨 UI HEADER
# ==============================
st.title("🤖 AI Streaming Chatbot")
st.caption("Powered by LangGraph + OpenAI")

# ==============================
# 💬 DISPLAY CHAT HISTORY
# ==============================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# ==============================
# ✍️ USER INPUT
# ==============================
user_input = st.chat_input("Type your message...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # ==============================
    # 🤖 STREAMING RESPONSE
    # ==============================
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Call LangGraph streaming
        events = chatbot.stream(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )

        # Stream tokens
        for event in events:
            for value in event.values():
                if "messages" in value:
                    for msg in value["messages"]:
                        if msg.content:
                            full_response += msg.content
                            response_placeholder.markdown(full_response + "▌")

        # Final response
        response_placeholder.markdown(full_response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )