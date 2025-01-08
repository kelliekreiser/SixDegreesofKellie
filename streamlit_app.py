import streamlit as st
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(page_title="Multi-Modal Chatbot", layout="wide")

# Load banner image
try:
    st.image("SixDegreesBanner.png", caption="Created by Kellie Kreiser (2025)", use_container_width=True)
except Exception as e:
    logger.error(f"Error loading banner image: {e}")
    st.error("Error loading banner image.")

# Title and description
st.title("Six Degrees of Kellie")
st.write("Get to know Kellie by seeing how you are connected!")
st.caption("Lost? Type 'Help' for instructions")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API key not found. Please check your Streamlit secrets.")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores chat messages
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-4o-mini"  # Default model
if "temperature" not in st.session_state:
    st.session_state.temperature = 1.0  # Fixed temperature to match original behavior

# Load system prompt from file
def load_system_prompt(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found at: {file_path}")
        return "You are a helpful assistant."

system_prompt = load_system_prompt("instructions.txt")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("What do you want to do?")

if user_input:
    # Add user message to session state
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)

    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare assistant response
    assistant_response = {"role": "assistant", "content": ""}
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Placeholder for assistant's response

        # Prepare conversation for OpenAI API
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(st.session_state.messages)  # Add chat history

        try:
            # Log the payload for debugging
            logger.info(f"Sending messages to OpenAI: {messages}")

            # OpenAI API call
            response = client.chat.completions.create(
                model=st.session_state.model_name,
                messages=messages,
                temperature=st.session_state.temperature,
                max_tokens=300,
            )

            # Validate the response
            if not response or not response.choices:
                raise ValueError("Invalid response from OpenAI API.")

            # Extract assistant response
            assistant_content = response.choices[0].message.content
            assistant_response["content"] = assistant_content

            # Display assistant response
            message_placeholder.markdown(assistant_content)
            st.session_state.messages.append(assistant_response)

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            st.error(f"An error occurred: {e}")

# Debugging info (optional, enabled via secrets)
if st.secrets.get("DEBUG_MODE", False):
    st.write("Debug Info:")
    st.write({"messages": st.session_state.messages, "model_name": st.session_state.model_name})
