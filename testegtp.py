import streamlit as st
from ollama.chatbot import Chatbot
from llama3 import Llama
from llama_index import LlamaIndex
from llama_index_embeddings_ollama import LlamaIndexEmbeddingsOllama

# Configuração inicial
st.title("Meu Chatbot")

# Inicializando os componentes necessários
ollama_chatbot = Chatbot()
llama_model = Llama()
llama_index = LlamaIndex()
llama_index_embeddings = LlamaIndexEmbeddingsOllama()

# Função para obter a resposta do chatbot
def get_bot_response(user_input):
    # Processamento da entrada do usuário
    processed_input = llama_model.process(user_input)
    indexed_input = llama_index.index_text(processed_input)
    embeddings_input = llama_index_embeddings.embed_text(indexed_input)
    bot_response = ollama_chatbot.get_response(embeddings_input)
    return bot_response

# Interface de entrada de texto para o usuário
user_input = st.text_input("Você: ", "")

# Botão para enviar a mensagem
if st.button("Enviar"):
    # Obtendo a resposta do bot
    bot_response = get_bot_response(user_input)
    # Exibindo a resposta
    st.text_area("Bot: ", value=bot_response, height=200, max_chars=None, key=None)
