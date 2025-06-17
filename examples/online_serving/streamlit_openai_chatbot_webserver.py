# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Chat Assistant - A Streamlit Web Interface

A streamlined chat interface that quickly integrates
with vLLM API server.

Features:
- Multiple chat sessions management
- Streaming response display
- Configurable API endpoint
- Real-time chat history
- Reasoning Display: Optional thinking process visualization 

Requirements:
    pip install streamlit openai

Usage:
    # Start the app with default settings
    streamlit run streamlit_openai_chatbot_webserver.py

    # Start with custom vLLM API endpoint
    VLLM_API_BASE="http://your-server:8000/v1" \
        streamlit run streamlit_openai_chatbot_webserver.py

    # Enable debug mode
    streamlit run streamlit_openai_chatbot_webserver.py \
        --logger.level=debug
"""

import os
from datetime import datetime

import streamlit as st
from openai import OpenAI

# Get command line arguments from environment variables
openai_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
openai_api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")

# Initialize session states for managing chat sessions
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_session" not in st.session_state:
    st.session_state.active_session = None

# Add new session state for reasoning
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = {}

# Initialize session state for API base URL
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = openai_api_base


def create_new_chat_session():
    """Create a new chat session with timestamp as unique identifier.

    This function initializes a new chat session by:
    1. Generating a timestamp-based session ID
    2. Creating an empty message list for the new session
    3. Setting the new session as both current and active session
    4. Resetting the messages list for the new session

    Returns:
        None

    Session State Updates:
        - sessions: Adds new empty message list with timestamp key
        - current_session: Sets to new session ID
        - active_session: Sets to new session ID
        - messages: Resets to empty list
    """
    session_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.sessions[session_id] = []
    st.session_state.current_session = session_id
    st.session_state.active_session = session_id
    st.session_state.messages = []


def switch_to_chat_session(session_id):
    """Switch the active chat context to a different session.

    Args:
        session_id (str): The timestamp ID of the session to switch to

    This function handles chat session switching by:
    1. Setting the specified session as current
    2. Updating the active session marker
    3. Loading the messages history from the specified session

    Session State Updates:
        - current_session: Updated to specified session_id
        - active_session: Updated to specified session_id
        - messages: Loaded from sessions[session_id]
    """
    st.session_state.current_session = session_id
    st.session_state.active_session = session_id
    st.session_state.messages = st.session_state.sessions[session_id]


def get_llm_response(messages, model, reason, content_ph=None, reasoning_ph=None):
    """Generate and stream LLM response with optional reasoning process.

    Args:
        messages (list): List of conversation message dicts with 'role' and 'content'
        model (str): The model identifier to use for generation
        reason (bool): Whether to enable and display reasoning process
        content_ph (streamlit.empty): Placeholder for streaming response content
        reasoning_ph (streamlit.empty): Placeholder for streaming reasoning process

    Returns:
        tuple: (str, str)
            - First string contains the complete response text
            - Second string contains the complete reasoning text (if enabled)

    Features:
        - Streams both reasoning and response text in real-time
        - Handles model API errors gracefully
        - Supports live updating of thinking process
        - Maintains separate content and reasoning displays

    Raises:
        Exception: Wrapped in error message if API call fails

    Note:
        The function uses streamlit placeholders for live updates.
        When reason=True, the reasoning process appears above the response.
    """
    full_text = ""
    think_text = ""
    live_think = None
    # Build request parameters
    params = {"model": model, "messages": messages, "stream": True}
    if reason:
        params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    try:
        response = client.chat.completions.create(**params)
        if isinstance(response, str):
            if content_ph:
                content_ph.markdown(response)
            return response, ""

        # Prepare reasoning expander above content
        if reason and reasoning_ph:
            exp = reasoning_ph.expander("💭 Thinking Process (live)", expanded=True)
            live_think = exp.empty()

        # Stream chunks
        for chunk in response:
            delta = chunk.choices[0].delta
            # Stream reasoning first
            if reason and hasattr(delta, "reasoning_content") and live_think:
                rc = delta.reasoning_content
                if rc:
                    think_text += rc
                    live_think.markdown(think_text + "▌")
            # Then stream content
            if hasattr(delta, "content") and delta.content and content_ph:
                full_text += delta.content
                content_ph.markdown(full_text + "▌")

        # Finalize displays: reasoning remains above, content below
        if reason and live_think:
            live_think.markdown(think_text)
        if content_ph:
            content_ph.markdown(full_text)

        return full_text, think_text
    except Exception as e:
        st.error(f"Error details: {str(e)}")
        return f"Error: {str(e)}", ""


# Sidebar - API Settings first
st.sidebar.title("API Settings")
new_api_base = st.sidebar.text_input(
    "API Base URL:", value=st.session_state.api_base_url
)
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
            args=(session_id,),
        )
    else:
        st.sidebar.button(
            f"Session {session_id}",
            key=session_id,
            on_click=switch_to_chat_session,
            args=(session_id,),
        )

# Main interface
st.title("vLLM Chat Assistant")

# Initialize OpenAI client with API settings
client = OpenAI(api_key=openai_api_key, base_url=st.session_state.api_base_url)

# Get and display current model id
models = client.models.list()
model = models.data[0].id
st.markdown(f"**Model**: {model}")

# Initialize first session if none exists
if st.session_state.current_session is None:
    create_new_chat_session()
    st.session_state.active_session = st.session_state.current_session

# Update the chat history display section
for idx, msg in enumerate(st.session_state.messages):
    # Render user messages normally
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    # Render assistant messages with reasoning above
    else:
        # If reasoning exists for this assistant message, show it above the content
        if idx in st.session_state.show_reasoning:
            with st.expander("💭 Thinking Process", expanded=False):
                st.markdown(st.session_state.show_reasoning[idx])
        with st.chat_message("assistant"):
            st.write(msg["content"])


# Setup & Cache reasoning support check
@st.cache_data(show_spinner=False)
def server_supports_reasoning():
    """Check if the current model supports reasoning capability.

    Returns:
        bool: True if the model supports reasoning, False otherwise
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
    )
    return hasattr(resp.choices[0].message, "reasoning_content") and bool(
        resp.choices[0].message.reasoning_content
    )


# Check support
supports_reasoning = server_supports_reasoning()

# Add reasoning toggle in sidebar if supported
reason = False  # Default to False
if supports_reasoning:
    reason = st.sidebar.checkbox("Enable Reasoning", value=False)
else:
    st.sidebar.markdown(
        "<span style='color:gray;'>Reasoning unavailable for this model.</span>",
        unsafe_allow_html=True,
    )
    # reason remains False

# Update the input handling section
if prompt := st.chat_input("Type your message here..."):
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.sessions[st.session_state.current_session] = (
        st.session_state.messages
    )
    with st.chat_message("user"):
        st.write(prompt)

    # Prepare LLM messages
    msgs = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    # Stream assistant response
    with st.chat_message("assistant"):
        # Placeholders: reasoning above, content below
        reason_ph = st.empty()
        content_ph = st.empty()
        full, think = get_llm_response(msgs, model, reason, content_ph, reason_ph)
        # Determine index for this new assistant message
        message_index = len(st.session_state.messages)
        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": full})
        # Persist reasoning in session state if any
        if reason and think:
            st.session_state.show_reasoning[message_index] = think
