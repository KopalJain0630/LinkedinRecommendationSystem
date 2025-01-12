import os
import streamlit as st
import json
import numpy as np
import requests
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import pickle
FAISS_INDEX_PATH = "faiss_index"
PROCESSED_FILES_PATH = "processed_files.pkl"

# Step 1: Load all JSON files from the folder
def load_json_files(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                content = json.dumps(data, indent=2)
                doc = Document(page_content=content, metadata={"id": os.path.splitext(file_name)[0]})
                documents.append(doc)
    return documents

# Folder containing the JSON files
folder_path = "final_user_profiles"
documents = load_json_files(folder_path)

def load_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        # Enable dangerous deserialization
        return FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)
    else:
        return None
    
def load_processed_files():
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, "rb") as f:
                processed_files = pickle.load(f)
                if isinstance(processed_files, set):
                    return processed_files
                else:
                    print("Warning: Processed files data is not a set. Resetting to an empty set.")
                    return set()
        except (EOFError, pickle.UnpicklingError):
            print("Warning: Processed files pickle file is empty or corrupted. Resetting to an empty set.")
            return set()
    else:
        return set()
    
def save_processed_files(processed_files):
    with open(PROCESSED_FILES_PATH, "wb") as f:
        pickle.dump(processed_files, f)

def update_vector_store(folder_path):
    documents = load_json_files(folder_path)
    processed_files = load_processed_files()
    vector_store = load_vector_store()
    
    if vector_store is None:
        # Initialize a new vector store if it doesn't exist
        vector_store = FAISS.from_documents(documents, OpenAIEmbeddings(model="text-embedding-3-large"))
    
    # Identify new files
    new_documents = [doc for doc in documents if doc.metadata["id"] not in processed_files]
    
    if new_documents:
        print(f"Embedding {len(new_documents)} new files...")
        vector_store.add_documents(new_documents)
        
        # Update processed files list
        processed_files.update(doc.metadata["id"] for doc in new_documents)
        save_processed_files(processed_files)
        
        # Save the updated vector store
        vector_store.save_local("faiss_index")
    else:
        print("No new files to embed.")

    return vector_store

def find_similar_profiles(profile_id, k=5, folder_path="final_user_profiles"):
    vector_store = update_vector_store(folder_path)
    processed_files = load_processed_files()
    
    if profile_id not in processed_files:
        raise ValueError(f"Profile with ID '{profile_id}' not found in the embedded data.")
    
    # Retrieve the input document
    documents = load_json_files(folder_path)
    query_doc = next((doc for doc in documents if doc.metadata["id"] == profile_id), None)
    if not query_doc:
        raise ValueError(f"Profile with ID '{profile_id}' not found in JSON files.")
    
    # Retrieve similar profiles
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k + 1})
    similar_docs = retriever.get_relevant_documents(query_doc.page_content)
    
    # Filter out the input profile itself
    similar_ids = [doc.metadata["id"] for doc in similar_docs if doc.metadata["id"] != profile_id][:k]
    return similar_ids

def main():
    st.title("Profile Recommendation System")
    st.sidebar.title("Settings")
    
    # Inputs
    folder_path = "final_user_profiles"
    input_profile_id = st.text_input("Input Profile ID", "ankurc07")
    top_k = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

    # Check for valid folder path
    if not os.path.exists(folder_path):
        st.error("The specified folder path does not exist. Please provide a valid path.")
        return

    # Perform recommendation
    if st.button("Find Similar Profiles"):
        try:
            similar_profiles = find_similar_profiles(input_profile_id, k=top_k, folder_path=folder_path)

            # Display recommendations
            st.subheader(f"Top {top_k} Similar Profiles for ID '{input_profile_id}':")
            for idx, profile_id in enumerate(similar_profiles, 1):
                st.markdown(f"**{idx}. Profile ID**: `{profile_id}`")
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()