(deployment-streamlit)=

# Streamlit

[Streamlit](https://github.com/streamlit/streamlit) lets you transform Python scripts into interactive web apps in minutes, instead of weeks. Build dashboards, generate reports, or create chat apps.

It can be quickly integrated with vLLM as a backend API server, enabling powerful LLM inference via API calls.

## Prerequisites

- Setup vLLM environment

## Deploy

- Start the vLLM server with the supported chat completion model, e.g.

```console
vllm serve qwen/Qwen1.5-0.5B-Chat
```

- Install streamlit and openai:

```console
pip install streamlit openai
```

- Create python file(streamlit-ui-app.py) and copy the below code into it:

```console
import streamlit as st
from openai import OpenAI
from datetime import datetime
import os

# Get command line arguments from environment variables
openai_api_key = os.getenv('VLLM_API_KEY', "EMPTY")
openai_api_base = os.getenv('VLLM_API_BASE', "http://localhost:8000/v1")

# Initialize session states for managing chat sessions
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_session" not in st.session_state:
    st.session_state.active_session = None

# Initialize session state for API base URL
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = openai_api_base

def create_new_chat_session():
    """Create a new chat session with timestamp as ID"""
    session_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.sessions[session_id] = []
    st.session_state.current_session = session_id
    st.session_state.active_session = session_id
    st.session_state.messages = []

def switch_to_chat_session(session_id):
    """Switch to a different chat session"""
    st.session_state.current_session = session_id
    st.session_state.active_session = session_id
    st.session_state.messages = st.session_state.sessions[session_id]

def get_llm_response(messages, model):
    """Get streaming response from llm
    
    Args:
        messages: List of message dictionaries
        model: Name of model
    
    Returns:
        Streaming response object or error message string
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        return response
    except Exception as e:
        st.error(f"Error details: {str(e)}")
        return f"Error: {str(e)}"

# Sidebar - API Settings first
st.sidebar.title("API Settings")
new_api_base = st.sidebar.text_input("API Base URL:", value=st.session_state.api_base_url)
if new_api_base != st.session_state.api_base_url:
    st.session_state.api_base_url = new_api_base
    st.rerun()

st.sidebar.divider()

# Sidebar - Session Management
st.sidebar.title("Chat Sessions")
if st.sidebar.button("New Session"):
    create_new_chat_session()

# Display all sessions in reverse chronological order
for session_id in sorted(st.session_state.sessions.keys(), reverse=True):
    # Mark the active session with a pinned button
    if session_id == st.session_state.active_session:
        st.sidebar.button(
            f"📍 {session_id}",
            key=session_id,
            type="primary",
            on_click=switch_to_chat_session,
            args=(session_id,)
        )
    else:
        st.sidebar.button(
            f"Session {session_id}",
            key=session_id,
            on_click=switch_to_chat_session,
            args=(session_id,)
        )

# Main interface
st.title("vLLM Chat Assistant")

# Initialize OpenAI client with API settings
client = OpenAI(
    api_key=openai_api_key,
    base_url=st.session_state.api_base_url
)

# Get and display current model id
models = client.models.list()
model = models.data[0].id
st.markdown(f"**Model**: {model}")

# Initialize first session if none exists
if st.session_state.current_session is None:
    create_new_chat_session()
    st.session_state.active_session = st.session_state.current_session

# Display chat history for current session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input and generate llm response
if prompt := st.chat_input("Type your message here..."):
    # Save user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.sessions[st.session_state.current_session] = st.session_state.messages
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Prepare messages for llm
    messages_for_llm = [
        {"role": m["role"], "content": m["content"]} 
        for m in st.session_state.messages
    ]

     # Generate and display llm response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Get streaming response from llm
        response = get_llm_response(messages_for_llm, model)
        if isinstance(response, str):
            message_placeholder.markdown(response)
            full_response = response
        else:
            for chunk in response:
                if hasattr(chunk.choices[0].delta, "content"):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
      
    # Save llm response to session history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
```

- Start the streamlit web UI and start to chat:

```console
streamlit run streamlit-ui-app.py

# or specify the VLLM_API_BASE or VLLM_API_KEY
VLLM_API_BASE="http://vllm-server-host:vllm-server-port/v1" streamlit run streamlit-ui-app.py

# start with debug mode to view more details
streamlit run streamlit-ui-app.py --logger.level=debug
```

:::{image} /assets/deployment/streamlit-chat.png
:::
