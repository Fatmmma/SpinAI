import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
import fuzzywuzzy
from fuzzywuzzy import process
import re
import replicate
import os

# App title
st.set_page_config(page_title="Chat with Spinneys AI", page_icon=":shark:", layout="wide")
st.title("Chat with Spinneys AI")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(
        llm,
        input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
               "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1}
    )
    return output

# CSV processing and knowledge base creation
def process_csv(file):
    df = pd.read_csv(file, encoding="latin-1")
    df['product_title'] = df['Product Name'].fillna('').astype(str).str.lower().str.strip()
    texts = list(map(lambda x: x.replace("\n", " "), df['product_title'].tolist()))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(texts, embeddings)
    return df, knowledge_base

# Query processing
def process_query(df, knowledge_base, query):
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
    if results:
        return "We have found:\n\n" + "\n\n".join(results)
    else:
        return "No matching products found."

# Recipe generation
def generate_recipe(prompt):
    formatted_prompt = (
        "You are a professional chef and recipe generator. Your task is to create a detailed, clear, and concise recipe based on the user's request. Please ensure the response is well-structured and includes the following sections without any additional commentary:\n\n"
        "- **Ingredients**: List all necessary ingredients with exact quantities.\n"
        "- **Equipment**: List all required equipment.\n"
        "- **Instructions**: Provide clear, step-by-step cooking instructions.\n"
        "- **Tips**: Offer any useful tips, variations, or serving suggestions.\n\n"
        "Please deliver the recipe in a professional tone, free of any unnecessary content or extraneous formatting instructions. Focus solely on the recipe details.\n\n"
        "Request: {}"
    ).format(prompt)

    response = generate_llama2_response(formatted_prompt)
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
    formatted_response = "\n\n".join(final_response)
    return f"Here is your recipe:\n\n{formatted_response}\n\nEnjoy your meal!"

# Handle prompt
def handle_prompt(prompt, df, knowledge_base):
    if prompt.lower().startswith("how to"):
        return generate_recipe(prompt)
    else:
        return process_query(df, knowledge_base, prompt)

if __name__ == '__main__':
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
                response = handle_prompt(prompt, st.session_state.df, st.session_state.knowledge_base)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.chat_message("assistant"):
                    st.markdown("Please upload a CSV first.")
