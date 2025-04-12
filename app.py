# resume_chatbot.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from resume_text import resume_text  # Resume content stored here

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="Vanjivaka Sairam's Resume Assistant", layout="centered")

def main():
    st.title("🤖 Vanjivaka Sairam's Resume Assistant")
    st.write("Ask anything about my resume. I'll only answer using that information.")

    # Sidebar settings
    st.sidebar.title("Settings")
    model = st.sidebar.selectbox("Choose LLM Model", ["gemma2-9b-it"])
    memory_length = st.sidebar.slider("Conversational Memory Length", 1, 10, 3)

    # Initialize memory for chat
    memory = ConversationBufferWindowMemory(k=memory_length)

    # Session state for persistent chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        memory.save_context({"input": message["human"]}, {"output": message["AI"]})

    # User input
    user_question = st.text_input("Ask a question about Vanjivaka Sairam's resume:")

    if user_question:
        chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

        # Messages
        system_message = SystemMessage(
            content=f"""
You are an AI assistant that answers questions based on Vanjivaka Sairam's resume below.

RULES:
1. Do NOT make up any information.
2. If the question can't be answered from the resume, say: "I can only answer questions based on Vanjivaka Sairam's resume."
3. Anser the questions based on previous questions also
4. Share only explicitly mentioned content from the resume.
5. Mention only listed projects, skills, and details—no assumptions.

Resume:
{resume_text}
"""
        )
        human_message = HumanMessage(content=user_question)

        try:
            response = chat([system_message, human_message]).content
            

            # Validate response content
            if "I can only answer" not in response and not any(
                keyword.lower() in response.lower()
                for keyword in ["sairam", "iit", "ropar"] + resume_text.split()[:20]
            ):
                response = "I can only answer questions based on Vanjivaka Sairam's resume."

            # Save chat
            st.session_state.chat_history.append({"human": user_question, "AI": response})
            st.markdown(f"**Resume Assistant:** {response}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Chat history
    # st.markdown("---")
    # st.subheader("Chat History")
    # for message in st.session_state.chat_history:
    #     st.markdown(f"**You:** {message['human']}")
    #     st.markdown(f"**Assistant:** {message['AI']}")

if __name__ == "__main__":
    main()
