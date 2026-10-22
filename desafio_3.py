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
    Reordena os documentos usando o LLM.
    Se o reranking falhar, mantém a ordem original
    fornecida pelo Chroma.
    """
    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""...
        """
    )

    documentos_com_score = []

    for indice, doc in enumerate(documentos):
        try:
            resultado = llm.invoke(
                prompt_rerank.format(
                    pergunta=pergunta,
                    texto=doc.page_content
                )
            ).content

            score = float(resultado.strip())

        except (ValueError, TypeError):
            # Resposta de classificação inválida
            score = 0

        except Exception as erro:
            # Falha de API: abandona o reranking e
            # preserva a ordem original do Chroma.
            st.warning(
                "O reranking ficou indisponível. "
                "Utilizando a ordem original da busca."
            )
            st.caption(
                f"Detalhe técnico: {type(erro).__name__}"
            )
            return documentos

        documentos_com_score.append((score, indice, doc))

    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda item: (item[0], -item[1]),
        reverse=True
    )

    return [doc for _, _, doc in documentos_ordenados]


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
        max_retries=2
    )

    try:
        resultado = llm.invoke("Responda apenas: OK")
        st.success(f"Modelo disponível: {resultado.content}")
    except Exception as erro:
        st.error(f"Falha na API: {type(erro).__name__}: {erro}")

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

@st.cache_data
def preparar_chunks():
    documentos = carregar_documentos()
    chunks = gerar_chunks(documentos)
    return enriquecer_chunks(chunks)
    
@st.cache_resource
def criar_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL
    )

    if (
        os.path.isdir(PERSIST_DIRECTORY)
        and os.listdir(PERSIST_DIRECTORY)
    ):
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

    return Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    
###############################################################
################ Inicialização da aplicação  ##################
###############################################################
chunks = preparar_chunks()
vectorstore = criar_vectorstore(chunks)
st.session_state.vectorstore = vectorstore
    

###############################################################
#################### INTERFACE DO CHAT ########################
###############################################################

st.title("Agente de RH — Diagnóstico")
st.caption("Esclareça suas dúvidas sobre as políticas da empresa.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico
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

# Entrada de mensagem
pergunta = st.chat_input("Digite sua pergunta sobre RH...")

if pergunta:
    st.session_state.messages.append({
        "role": "user",
        "content": pergunta
    })

    with st.chat_message("user"):
        st.markdown(pergunta)

    try:
        with st.chat_message("assistant"):
            with st.spinner(
                "Consultando documentos e gerando resposta..."
            ):
                resposta, contexto_final = responder_pergunta(
                    pergunta,
                    st.session_state.vectorstore
                )

            st.markdown(resposta)

            # Prepara fontes simples para o histórico
            fontes = []

            for doc in contexto_final:
                fontes.append({
                    "origem": doc.metadata.get(
                        "documento",
                        doc.metadata.get("source", "Não identificada")
                    ),
                    "pagina": doc.metadata.get("page"),
                    "conteudo": doc.page_content[:500]
                })

            with st.expander("Documentos utilizados"):
                for i, fonte in enumerate(fontes, start=1):
                    st.markdown(f"**Documento {i}**")
                    st.write(f"Origem: {fonte['origem']}")

                    if fonte["pagina"] is not None:
                        st.write(f"Página: {fonte['pagina'] + 1}")

                    st.write(fonte["conteudo"])
                    st.divider()

        # Salva a resposta somente após o processamento bem-sucedido
        st.session_state.messages.append({
            "role": "assistant",
            "content": resposta,
            "fontes": fontes
        })

    except Exception as erro:
        st.error(
            "Não foi possível responder à pergunta. "
            "Verifique os logs para mais detalhes."
        )
        st.exception(erro)

