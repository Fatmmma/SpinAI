import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import HuggingFacePipeline, PromptTemplate
from fuzzywuzzy import process
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA

# Function to process CSV
def process_csv(file):
    df = pd.read_csv(file, encoding="latin-1")
    df['product_title'] = df['Product Name'].fillna('').astype(str).str.lower().str.strip()
    texts = list(map(lambda x: x.replace("\n", " "), df['product_title'].tolist()))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(texts, embeddings)
    return df, knowledge_base

# Function to get TinyLlama LLM
def get_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    generation_config = GenerationConfig(max_new_tokens=256, temperature=0.1, top_p=0.9, do_sample=True, repetition_penalty=1.1)
    text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=True, generation_config=generation_config)
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm

# Function to process queries
def process_query(df, knowledge_base, query, llm):
    docs = knowledge_base.similarity_search(query, k=10)
    results = []
    seen_titles = set()
    doc_titles = [doc.page_content.lower() for doc in docs]
    best_matches = process.extract(query.lower(), doc_titles, limit=10)
    for match in best_matches:
        product_title = match[0].strip()
        if product_title in seen_titles:
            continue
        seen_titles.add(product_title)
        sku_matches = df.loc[df['product_title'] == product_title, 'SKU'].values
        if len(sku_matches) > 0:
            results.append(f"{product_title.title()}\nSKU: {sku_matches[0]}")
    return "We have found:\n\n" + "\n\n".join(results) if results else "No matching products found."

# Function to generate a recipe
def generate_recipe(prompt, llm):
    template = """
    You are a professional chef and recipe generator. Your task is to create a detailed, clear, and concise recipe based on the user's request. Please ensure the response is well-structured and includes the following sections without any additional commentary:

    - **Ingredients**: List all necessary ingredients with exact quantities.
    - **Equipment**: List all required equipment.
    - **Instructions**: Provide clear, step-by-step cooking instructions.
    - **Tips**: Offer any useful tips, variations, or serving suggestions.

    Please deliver the recipe in a professional tone, free of any unnecessary content or extraneous formatting instructions. Focus solely on the recipe details.

    Request: {question}
    """
    formatted_prompt = template.format(question=prompt)
    response = llm(formatted_prompt)
    return response[0]['generated_text'] if isinstance(response, list) else response

# Function to handle user prompts
def handle_prompt(prompt, df, knowledge_base, llm):
    if prompt.lower().startswith("how to"):
        return generate_recipe(prompt, llm)
    else:
        return process_query(df, knowledge_base, prompt, llm)

# Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title="Chat with Spinneys AI", page_icon=":shark:", layout="wide")
    st.title("Chat with Spinneys AI")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Upload CSV")
        if 'df' not in st.session_state or 'knowledge_base' not in st.session_state:
            csv_file = st.file_uploader("Upload your CSV File", type="csv")
            if csv_file is not None:
                with st.spinner("Processing..."):
                    df, knowledge_base = process_csv(csv_file)
                    st.session_state.df = df
                    st.session_state.knowledge_base = knowledge_base
                    st.session_state.llm = get_llm()
                st.success("CSV successfully uploaded and processed!")
        else:
            st.write("CSV file already uploaded and processed!")

    with col2:
        st.header("Chat")
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What product are you looking for?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if 'df' in st.session_state:
                response = handle_prompt(prompt, st.session_state.df, st.session_state.knowledge_base, st.session_state.llm)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.chat_message("assistant"):
                    st.markdown("Please upload a CSV first.")
