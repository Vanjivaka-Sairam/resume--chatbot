import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from resume_text import resume_text

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Resume Assistant Chatbot", layout="centered")

def main():
    st.title("🤖 Resume Assistant Chatbot")
    st.write("Ask anything about my resume. The bot will only answer using that information.")

    st.sidebar.title("Settings")
    model = st.sidebar.selectbox("Choose LLM Model", ["mixtral-8x7b-32768", "llama-70b-4096"])
    memory_length = st.sidebar.slider("Conversational Memory Length", 1, 10, 5)

    # Memory for maintaining conversation context
    memory = ConversationBufferWindowMemory(k=memory_length)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        memory.save_context({"input": message["human"]}, {"output": message["AI"]})

    user_question = st.text_area("Ask a question about my resume:")

    if user_question:
        chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

        system_prompt = (
            "You are an AI assistant that only answers questions using the following resume:\n\n"
            + resume_text +
            "\n\nNow answer this question truthfully:\n"
            + user_question
        )

        conversation = ConversationChain(llm=chat, memory=memory)
        response = conversation.predict(input=system_prompt)

        st.session_state.chat_history.append({"human": user_question, "AI": response})
        st.markdown(f"**Chatbot:** {response}")

    st.markdown("---")
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        st.markdown(f"**You:** {message['human']}")
        st.markdown(f"**AI:** {message['AI']}")

if __name__ == "__main__":
    main()
