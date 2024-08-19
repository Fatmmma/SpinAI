import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.llms import HuggingFaceHub
import fuzzywuzzy
from fuzzywuzzy import process
import re

def process_csv(file):
    df = pd.read_csv(file, encoding="latin-1")
    # Ensure all product titles are strings and handle NaNs
    df['product_title'] = df['Product Name'].fillna('').astype(str).str.lower().str.strip()

    # Create embeddings and knowledge base
    texts = list(map(lambda x: x.replace("\n", " "), df['product_title'].tolist()))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(texts, embeddings)
    
    return df, knowledge_base

def get_llm():
    # Provide your API key directly here
    api_key = "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"
    return HuggingFaceHub(
        repo_id="NousResearch/Llama-3-7b-chat-hf",
        model_kwargs={"temperature": 0.5, "max_length": 1024},
        huggingfacehub_api_token=api_key
    )

def process_query(df, knowledge_base, query, llm):
    # Retrieve multiple similar matches
    docs = knowledge_base.similarity_search(query, k=10)
    results = []
    seen_titles = set()
    
    # Convert document contents to lower case for matching
    doc_titles = [doc.page_content.lower() for doc in docs]
    
    # Fuzzy matching to find best matches
    best_matches = process.extract(query.lower(), doc_titles, limit=10)
    
    for match in best_matches:
        product_title = match[0].strip()
        if product_title in seen_titles:
            continue
        seen_titles.add(product_title)
        
        sku_matches = df.loc[df['product_title'] == product_title, 'SKU'].values
        if len(sku_matches) > 0:
            results.append(f"{product_title.title()}\nSKU: {sku_matches[0]}")
    
    if results:
        return "We have found:\n\n" + "\n\n".join(results)
    else:
        return "No matching products found."
        



import re

def generate_recipe(prompt, llm):
    # Template to guide the LLM in providing a focused, professional recipe
    template = """
    [Template Start]

    ### Instruction:
    You are an expert chef. Your sole purpose is to generate detailed recipes based on the customer's request. 
    Please do not respond to any non-relevant questions. If a question is not related to recipes, respond with: 
    "Sorry, I specialize in generating recipes only. How can I assist you with a recipe today?"

    ### Input:
    - The input provided will be a customer's request for a specific recipe.
    - If the input is related to a recipe, generate a detailed and unique recipe without repeating previous answers.
    - If the input is not related to recipes, respond with the standard non-relevant response.

    ### Output Requirements:
    - Only output the recipe content without any additional text, explanation, or repetition.
    - Do not print the input prompt in the output.
    - Ensure the recipe is unique and not a replication of previous responses.

    Request: {}
    """.format(prompt)

    # Call the LLM with the formatted template
    response = llm(template)

    # Extract the recipe content
    # Assuming the model outputs a properly formatted response, otherwise, handle accordingly
    final_response = response.strip()

    # Return the final recipe output
    return final_response
    
def handle_prompt(prompt, df, knowledge_base, llm):
    if prompt.lower().startswith("how to"):
        return generate_recipe(prompt, llm)
    else:
        return process_query(df, knowledge_base, prompt,llm)


if __name__ == '__main__':
    st.set_page_config(page_title="Chat with Spinneys AI", page_icon=":shark:", layout="wide")
    st.title("Chat with Spinneys AI")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Upload CSV")

        # Check if the CSV data is already in session_state
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
