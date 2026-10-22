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
LLM_MODEL = "gemini-3.5-flash-lite"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
#API_KEY = "GEMINI_AULA_ALURA"

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
        api_key = st.secrets["GEMINI_AULA_ALURA"],
        model=LLM_MODEL,
        temperature=0,
        timeout=60,
        max_retries=1
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
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = criar_vectorstore(chunks)
    
st.title("Agente de RH — Diagnóstico")
st.caption("Esclareça suas dúvidas sobre a empresa")

# Inicializa o histórico
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Exibe as mensagens anteriores
for mensagem in st.session_state.messages:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])

        fontes = mensagem.get("fontes", [])
        if fontes:
            with st.expander("Documentos utilizados"):
                for i, fonte in enumerate(fontes, start=1):
                    st.markdown(f"**Documento {i}**")
                    st.write(f"Origem: {fonte['origem']}")

                    if fonte["pagina"] is not None:
                        st.write(f"Página: {fonte['pagina'] + 1}")

                    st.write(fonte["conteudo"])
                    st.divider()

# Campo de entrada do chat
pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    # Exibe a mensagem do usuário
    st.session_state.messages.append({
        "role": "user",
        "content": pergunta
    })

    with st.chat_message("user"):
        st.markdown(pergunta)

    # Verifica se o banco vetorial está disponível
    if "vectorstore" not in st.session_state:
        st.error("Carregue e processe os documentos primeiro.")
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Consultando documentos e gerando resposta..."
                ):
                   resposta, contexto_final = responder_pergunta(
                       pergunta,
                       st.session_state.vectorstore
                   )

 # Converte os documentos para dados simples
                fontes = []

                for doc in contexto_final:
                    fontes.append({
                        "origem": doc.metadata.get(
                            "source", "Não identificada"
                        ),
                        "pagina": doc.metadata.get("page"),
                        "conteudo": doc.page_content[:500]
                    })

                st.markdown(resposta)

                with st.expander("Documentos utilizados"):
                    for i, fonte in enumerate(fontes, start=1):
                        st.markdown(f"**Documento {i}**")
                        st.write(f"Origem: {fonte['origem']}")

                        if fonte["pagina"] is not None:
                            st.write(f"Página: {fonte['pagina'] + 1}")

                        st.write(fonte["conteudo"])
                        st.divider()
                        
 # Armazena resposta e fontes
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resposta,
                    "fontes": fontes
                })

            except Exception as erro:
                st.error(f"Erro ao consultar o RAG: {erro}")
