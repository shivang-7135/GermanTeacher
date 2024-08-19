import proto
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
tes = os.getenv("GOOGLE_API_KEY")
print(tes)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Initialize embeddings (if needed)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def main():

    st.title("German Learning Assistant with Google Gemini")

    # Feature 1: Translation
    st.header("Translation")
    text_to_translate = st.text_area("Enter text to translate:")
    if st.button("Translate to German"):
        prompt = f"Translate the following text to German: {text_to_translate}"
        response = model.predict(prompt)
        print("###$$$$$#####res$$$$###"+response)
        translation = response
        st.write(f"Translation: {translation}")

    # Feature 2: Grammar Correction
    st.header("Grammar Correction")
    german_text = st.text_area("Enter German text for grammar correction:")
    if st.button("Correct Grammar"):
        prompt = f"Correct the grammar of the following German sentence: {german_text}"
        response = model.predict(prompt)
        corrected_text = response
        st.write(f"Corrected Sentence: {corrected_text}")

    # Feature 3: Vocabulary Building
    st.header("Vocabulary Building")
    if st.button("Learn a New Word"):
        prompt = "Provide a random German word with its English translation, an example sentence, and its usage."
        response = model.predict(prompt)
        vocab_info = response
        st.write(vocab_info)

    # Feature 4: Conversation Practice
    st.header("Conversation Practice")
    user_input = st.text_input("Say something in German:")
    if st.button("Respond"):
        prompt = f"Respond to this in German: {user_input}"
        response = model.predict(prompt)
        reply = response
        st.write(f"AI: {reply}")


if __name__ == "__main__":
    main()