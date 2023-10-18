import os
import requests
from haystack.schema import Document
from haystack.nodes import (
    FARMReader,
    PreProcessor,
)
from flask import Flask, request, jsonify
from trafilatura import fetch_url, extract
from dotenv import load_dotenv

from doc_classifier.transformers import TransformersDocumentClassifier

# Carregando variáveis de ambiente de um arquivo .env.
load_dotenv()

# Obtendo as credenciais para a API Moz e configurando o endpoint da API.
moz_username = os.getenv("MOZ_API_ID")
moz_password = os.getenv("MOZ_API_KEY")
moz_endpoint = "https://lsapi.seomoz.com/v2/url_metrics"


# Extrai o conteúdo de uma página web dada sua URL.
# Usa a função fetch_url() para obter o conteúdo HTML e depois extract() para extrair o texto do HTML.
def extract_web_page_content(url):
    html_content = fetch_url(url)
    return extract(html_content)


# Envia uma requisição POST para obter métricas da página web usando a API Moz.
def extract_web_page_metrics(url):
    data = {"targets": [url]}
    return requests.post(moz_endpoint, auth=(moz_username, moz_password), json=data)


# Prepara os documentos para classificação de notícias reais e falsas
def prepare_docs_for_classification(title, docs):
    for doc in docs:
        doc.content = "<title>" + title + "<content>" + doc.content + "<end>"
    return docs


# Calcula a pontuação de confiança a partir dos documentos classificados
def calculate_trust_score_from_docs(docs):
    trust_score = 0
    for doc in docs:
        trust_score += doc.meta["classification"]["details"]["TRUE"]
    return trust_score / len(docs)


# Criando uma instância do Haystack para pré-processar os documentos.
doc_preprocessor = PreProcessor()

# Carregando um modelo do Haystack para responder a perguntas
reader_model_name = "deepset/roberta-base-squad2"
reader = FARMReader(
    reader_model_name,
    context_window_size=386,  # Seta o tamanho da janela de contexto para 386 caracteres
    max_seq_len=386,  # Seta o tamanho máximo da sequência para 386 caracteres
    batch_size=96,  # Seta o tamanho do batch para 96
    use_gpu=True,  # Seta para utilizar a GPU
    top_k=1,  # Seta o número de respostas finais para 1 (a resposta mais confiante)
    top_k_per_candidate=1,  # Seta o número de respostas por documento para 1
)

# Carregando um modelo do Haystack para classificar notícias reais e falsas
doc_classifier_model_name = "hamzab/roberta-fake-news-classification"
doc_classifier = TransformersDocumentClassifier(
    model_name_or_path=doc_classifier_model_name, use_gpu=True, top_k=None
)

# Criando uma instância do aplicativo Flask.
app = Flask(__name__)


# Definindo uma rota no Flask que aceita requisições POST no endpoint /.
@app.route("/", methods=["POST"])
def index():
    # Recebe os dados JSON da requisição POST
    data = request.json

    # Verifica se os dados contêm as chaves "query" e "url"
    if data is None or "query" not in data or "url" not in data:
        # Se não, retorna uma mensagem de erro e um código de status 400 (Bad Request)
        return jsonify({"error": "Invalid request data"}), 400

    # Chama a função para extrair métricas da página web a partir da URL
    authority_response = extract_web_page_metrics(data["url"])

    # Verifica se a resposta da requisição para as métricas tem um status code diferente de 200 (OK)
    if authority_response.status_code != 200:
        # Se não, retorna uma mensagem de erro e um código de status 400 (Bad Request)
        return jsonify({"error": "Error with page authority lookup"}), 400

    # Extrai a pontuação de autoridade da página e o título da resposta JSON
    page_authority_score = authority_response.json()["results"][0]["page_authority"]
    page_title = authority_response.json()["results"][0]["title"]

    # Verifica se o título da página é nulo
    if page_title is None:
        # Se for, retorna uma mensagem de erro e um código de status 400 (Bad Request)
        return (
            jsonify({"error": "URL provided did not return any title"}),
            400,
        )

    # Extrai o conteúdo da página web a partir da URL
    page_content = extract_web_page_content(data["url"])

    # Verifica se o conteúdo da página é nulo
    if page_content is None:
        # Se for, retorna uma mensagem de erro e um código de status 400 (Bad Request)
        return (
            jsonify({"error": "URL provided did not return any content"}),
            400,
        )

    # Pré-processa o conteúdo da página web dividindo em documentos do Haystack
    content_docs = doc_preprocessor.process(
        Document.from_dict({"content": page_content})
    )

    # Faz uma predição de resposta com o modelo de leitura do Haystack
    prediction = reader.predict(
        query=data["query"],
        documents=content_docs,
    )

    # Obtém a pontuação de confiança da resposta
    answer_confidence_score = prediction["answers"][0].score

    # Prepara os documentos para classificação de notícias reais e falsas
    classifier_docs = prepare_docs_for_classification(page_title, content_docs)

    # Faz uma predição de classificação de notícias reais e falsas
    processed_classifier_documents = doc_classifier.predict(documents=classifier_docs)

    # Obtém a pontuação de confiança na notícia real
    trust_score = calculate_trust_score_from_docs(processed_classifier_documents)

    # Verifica se há uma resposta do modelo
    if prediction["answers"]:
        # Verifica se a primeira resposta é válida
        if prediction["answers"][0]:
            # Se for, retorna um JSON com várias informações
            return jsonify(
                {
                    "query": data["query"],
                    "content": page_content,
                    "url": data["url"],
                    "answer_confidence_score": answer_confidence_score,
                    "trust_score": trust_score,
                    "page_authority_score": page_authority_score,
                    "answer": prediction["answers"][0].answer,
                }
            )


# Inicializando o servidor Flask para escutar em um determinado endereço IP e porta, com o modo de depuração ativado.
if __name__ == "__main__":
    app.run(host="172.18.92.237", port=5000, debug=True)
