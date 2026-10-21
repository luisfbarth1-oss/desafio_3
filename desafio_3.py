# TERCEIRO PROJETO - AGENTE DE RH
###############################################################
########### IMPORTAÇÃO DE PACOTES E BIBLIOTECAS ###############
###############################################################
import os
import streamlit as st

# Loaders e chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings e LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Vector Store
from langchain_community.vectorstores import Chroma

# Prompt
from langchain_core.prompts import PromptTemplate
###############################################################
########### TESTE DE INICIALIZAÇÃO DO STREAMLIT ###############
###############################################################
st.set_page_config(page_title="Agente de RH")

st.title("Agente de RH")
st.write("Etapa 1: aplicação iniciada")

st.write("PDFs disponíveis:")
for arquivo in [
    "politica_ferias.pdf",
    "politica_home_office.pdf",
    "codigo_conduta.pdf"
]:
    st.write(f"{arquivo}: {os.path.exists(arquivo)}")

st.write("Etapa 2: verificação de arquivos concluída")
