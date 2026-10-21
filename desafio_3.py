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
from langchain_chroma import Chroma

# Prompt
from langchain_core.prompts import PromptTemplate

###############################################################
################### CONFIGURAÇÕES GERAIS ######################
###############################################################

# Diretório do banco vetorial
PERSIST_DIRECTORY = "./chroma_rh"

# Modelo de embeddings
EMBEDDING_MODEL = "BAAI/bge-m3"

# Modelo de linguagem
LLM_MODEL = "google/gemma-4-e2b"
BASE_URL = "http://localhost:1235/v1"

###############################################################
########### IMPORTAÇÃO E PROCESSAMENTO DE DOCUMENTOS ##########
###############################################################
@st.cache_data
def carregar_documentos():
    """
    Carrega os PDFs de políticas internas de RH
    """
    caminhos = [
        "politica_ferias.pdf",
        "politica_home_office.pdf",
        "codigo_conduta.pdf"
    ]

    documentos = []
    for caminho in caminhos:
        loader = PyPDFLoader(caminho)
        docs = loader.load()
        for doc in docs:
            doc.metadata["documento"] = caminho
        documentos.extend(docs)
    return documentos

###############################################################
################ DIVIDINDO DOCUMENTOS EM CHUNKS ###############
###############################################################

def gerar_chunks(documentos):
    """
    Divide os documentos em chunks semânticos
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documentos)

###############################################################
######## CRIANDO VECTORSTORE E REORDENANDO DOCUMENTOS #########
###############################################################
def enriquecer_chunks(chunks):
    """
    Classifica os chunks por categoria semântica
    """
    for chunk in chunks:
        texto = chunk.page_content.lower()

        if "ferias" in texto:
            chunk.metadata["categoria"] = "ferias"
        elif "home office" in texto or "remoto" in texto:
            chunk.metadata["categoria"] = "home_office"
        elif "conduta" in texto or "etica" in texto:
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"
    return chunks

@st.cache_resource
def criar_vectorstore(_chunks):
    """
    Cria ou carrega o banco vetorial.
    O parâmetro _chunks não entra no hash do cache.
    """
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

def rerank_documentos(pergunta, documentos, llm):
    """
    Reordena os documentos recuperados com base na relevância
    usando o próprio LLM (reranking semântico)
    """
    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""..."""
    )
    
    documentos_com_score = []
    for doc in documentos:
        score = llm.invoke(
            prompt_rerank.format(
                pergunta=pergunta,
                texto=doc.page_content
            )
        ).content

        try:
            score = float(score)
        except:
            score = 0
        
        documentos_com_score.append((score, doc))

    # Ordena do mais relevante para o menos relevante
    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )
    # Retorna apenas os documentos
    return [doc for _, doc in documentos_ordenados]

def responder_pergunta(pergunta, vectorstore):
    """
    Pipeline completo:
    - Recuperação
    - Reranking
    - Geração de resposta
    """
    # LLM
    llm = ChatOpenAI(
        base_url= BASE_URL,
        api_key="lm_studio",
        model=LLM_MODEL,
        temperature=0
    )

    # Recuperação inicial (top-k mais alto)
    documentos_recuperados = vectorstore.similarity_search(
        pergunta,
        k=8
    )

    # Reranking
    documentos_rerankeados = rerank_documentos(
        pergunta,
        documentos_recuperados,
        llm
    )
    
    # Seleciona os melhores
    contexto_final = documentos_rerankeados[:4]

    # Prompt final
    contexto_texto = "\n\n".join(
        [doc.page_content for doc in contexto_final]
    )

    prompt_final = f"""..."""

    resposta = llm.invoke(prompt_final)

    return resposta.content, contexto_final
    
###############################################################
########### TESTE DE INICIALIZAÇÃO DO STREAMLIT ###############
###############################################################

st.title("Agente de RH — Diagnóstico")

try:
    st.write("Etapa 1: carregando documentos...")
    documentos = carregar_documentos()
    st.success(f"Documentos carregados: {len(documentos)} páginas")

    st.write("Etapa 2: gerando chunks...")
    chunks = gerar_chunks(documentos)
    chunks = enriquecer_chunks(chunks)
    st.success(f"Chunks gerados: {len(chunks)}")

    st.write("Etapa 3: criando banco vetorial...")
    vectorstore = criar_vectorstore(chunks)
    st.success("Banco vetorial criado!")

    st.write("Etapa 4: testando recuperação...")
    resultados = vectorstore.similarity_search(
        "Como funciona a política de férias?",
        k=2
    )
    st.success(f"Documentos recuperados: {len(resultados)}")

except Exception as e:
    st.error("Falha durante o diagnóstico")
    st.exception(e)
