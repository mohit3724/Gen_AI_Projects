import streamlit as st
from PyPDF2 import PdfReader
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Title of the Streamlit app
st.title("Policy Recommendation")

# File uploader to accept multiple PDF files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Dictionary to store the content of each PDF
pdf_contents = {}

# Function to chunk text
def chunk_text(text, chunk_size=512, overlap=50):
    if not text:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

if uploaded_files:
    with ThreadPoolExecutor() as executor:
        futures = []
        for uploaded_file in uploaded_files:
            futures.append(executor.submit(PdfReader, io.BytesIO(uploaded_file.read())))
        
        for future, uploaded_file in zip(futures, uploaded_files):
            pdf_reader = future.result()
            pdf_text = [page.extract_text() for page in pdf_reader.pages]
            pdf_contents[uploaded_file.name] = pdf_text

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dictionary to store the embeddings and corresponding chunks
pdf_embeddings = {}
pdf_chunks = {}

# Generate embeddings for each PDF content
def generate_embeddings(file_name, texts):
    all_chunks = []
    all_text_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_text_chunks.extend(chunks)
    embeddings = model.encode(all_chunks)
    return file_name, embeddings, all_text_chunks

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(generate_embeddings, file_name, texts) for file_name, texts in pdf_contents.items()]
    for future in futures:
        file_name, embeddings, chunks = future.result()
        pdf_embeddings[file_name] = embeddings
        pdf_chunks[file_name] = chunks

# Create a FAISS index
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
for embeddings in pdf_embeddings.values():
    index.add(np.array(embeddings, dtype=np.float32))

# Display the number of vectors in the FAISS index (for debugging purposes)
st.write(f"Number of vectors in the FAISS index: {index.ntotal}")

# User prompt input
user_prompt = st.text_input("Enter your requirements for the insurance policy:")

# Submit button to send the user query to the LLM
if st.button("Submit"):
    if user_prompt:
        # Generate embedding for the user prompt
        user_prompt_embedding = model.encode([user_prompt])

        # Search the FAISS index for the most similar chunks
        D, I = index.search(np.array(user_prompt_embedding, dtype=np.float32), k=5)  # Get top 5 results

        # Collect the most relevant chunks (actual text)
        relevant_chunks = []
        for idx in I[0]:
            for file_name, embeddings in pdf_embeddings.items():
                if idx < len(embeddings):
                    relevant_chunks.append(f"From {file_name}: {pdf_chunks[file_name][idx]}")
                    break
                else:
                    idx -= len(embeddings)

        llm_prompt =  f"You are an insurance advisor tasked with recommending the best term insurance policy based on the following user requirements:{user_prompt}"

        for chunk in relevant_chunks:
            llm_prompt += f"{chunk}\n"

        llm_prompt += """
        Considering the user’s profile and requirements, analyze the provided policy details and recommend the best policy. Explain why this policy is the best choice and highlight any potential drawbacks or limitations.
        """

        response = get_gemini_response(llm_prompt)


        # Display the decision
        st.write("Decision on the best policy:")
        st.write(response)


# User Profile to be give in this fashion for better output: - 
# Gender: Male - 
# Age: 25 years - 
# Salary: 55k - D
# esired Policy Term: 75 years - 
# Desired Coverage: 50 to 75 lakhs - 
# Additional Benefits: Accidental and critical illness rider benefits