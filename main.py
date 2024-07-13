import json
import streamlit as st
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


with open('data.json') as f:
    data = json.load(f)


def prepare_job_data(data):
    documents = []
    for field in data['jobs']:
        for company in field['companies']:
            for job in company['jobs']:
                document = f"{job['title']} at {company['name']} in {field['field']} field"
                documents.append(document)
    return documents

documents = prepare_job_data(data)


d = 512  
vectors = np.random.random((len(documents), d)).astype('float32')

index = faiss.IndexFlatL2(d)
index.add(vectors)

def get_recommendations(query):
    query_vector = np.random.random((1, d)).astype('float32') 
    D, I = index.search(query_vector, 5)
    recommendations = [documents[i] for i in I[0]]
    return recommendations


llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key="AI******************", temperature=0.5)


prompt_template = PromptTemplate(input_variables=["context"], template="You are a job recommendation assistant. Use the following context to provide job recommendations: {context}")

chain = LLMChain(llm=llm, prompt=prompt_template)

st.title('Job Recommendation Chatbot')
user_input = st.text_input('Ask me about job recommendations:')
if user_input:
    recommendations = get_recommendations(user_input)
    context = "\n".join(recommendations)
    response = chain.run(context=context)
    st.write(response)
