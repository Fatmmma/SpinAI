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
        repo_id="Meta-Llama-3-8B-Instruct",
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
        



def generate_recipe(prompt, llm):
    # Refined prompt to guide the LLM in providing a clean, professional recipe
    formatted_prompt = (
        "You are a professional chef and recipe generator. Your task is to create a detailed, clear, and concise recipe based on the user's request. Please ensure the response is well-structured and includes the following sections without any additional commentary:\n\n"
        "- **Ingredients**: List all necessary ingredients with exact quantities.\n"
        "- **Equipment**: List all required equipment.\n"
        "- **Instructions**: Provide clear, step-by-step cooking instructions.\n"
        "- **Tips**: Offer any useful tips, variations, or serving suggestions.\n\n"
        "Please deliver the recipe in a professional tone, free of any unnecessary content or extraneous formatting instructions. Focus solely on the recipe details.\n\n"
        "Request: {}"
    ).format(prompt)

    # Call the LLM with the formatted prompt
    response = llm(formatted_prompt)

    # Clean and structure the response
    sections = ["Ingredients", "Equipment", "Instructions", "Tips"]
    final_response = []

    for section in sections:
        start_index = response.find(f"**{section}**:")
        if start_index != -1:
            end_index = len(response)
            for next_section in sections[sections.index(section) + 1:]:
                next_index = response.find(f"**{next_section}**:", start_index)
                if next_index != -1:
                    end_index = min(end_index, next_index)
            section_content = response[start_index:end_index].strip()
            final_response.append(section_content)

    # Join all sections to form the final structured recipe
    formatted_response = "\n\n".join(final_response)

    # Return the cleaned and formatted recipe
    return f"Here is your recipe:\n\n{formatted_response}\n\nEnjoy your meal!"




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
