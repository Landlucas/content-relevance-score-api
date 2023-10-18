import os
import requests
import torch
from haystack.schema import Document
from haystack.nodes import FARMReader, TransformersSummarizer
from flask import Flask, request, jsonify
from trafilatura import fetch_url, extract
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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


def summarize_text(text):
    docs = [Document(text)]

    summarizer = TransformersSummarizer(model_name_or_path="google/pegasus-xsum", use_gpu=True)
    summary = summarizer.predict(documents=docs)

    return summary[0].meta["summary"]

def summarize_large_text(input_text):
    max_chunk_size = 512

    chunks = [input_text[i:i+max_chunk_size] for i in range(0, len(input_text), max_chunk_size)]

    print(f"Input size: {len(input_text)}")
    print(f"Number of chunks: {len(chunks)}")

    summaries = []
    for chunk in chunks:
        chunk_summary = summarize_text(chunk)
        summaries.append(chunk_summary)

    combined_summary = " ".join(summaries)
    return combined_summary


# Realiza a predição de notícias falsas usando um modelo de classificação de texto pré-treinado.
def predict_fake_news(title, text):
    # Carrega o tokenizer pré-treinado para o modelo de classificação de notícias falsas
    tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")

    # Carrega o modelo de classificação de notícias falsas
    model = AutoModelForSequenceClassification.from_pretrained(
        "hamzab/roberta-fake-news-classification"
    )

    # Combina o título e o texto da notícia em uma única string formatada de acordo com os requisitos do modelo
    input_str = "<title>" + title + "<content>" + text + "<end>"

    # Converte a string em tokens e os codifica
    input_ids = tokenizer.encode_plus(
        input_str,
        max_length=512,  # Define o comprimento máximo do input
        padding="max_length",  # Preenche com zeros até o comprimento máximo
        truncation=True,  # Trunca o input se exceder o comprimento máximo
        return_tensors="pt",  # Retorna tensores PyTorch
    )

    # Determina o dispositivo a ser usado para inferência (GPU ou CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move o modelo para o dispositivo especificado
    model.to(device)

    # Desabilita o cálculo de gradientes para economizar memória durante a inferência
    with torch.no_grad():
        # Realiza a inferência com o modelo
        output = model(
            input_ids["input_ids"].to(device),
            attention_mask=input_ids["attention_mask"].to(device),
        )

    # Converte as probabilidades de saída em um formato legível: Fake e Real, ambos com valores entre 0 e 1.
    return dict(
        zip(
            ["Fake", "Real"],
            [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])],
        )
    )


# Carregando um modelo do Haystack para responder a perguntas baseado no modelo "deepset/roberta-base-squad2".
reader_model_name = "deepset/roberta-base-squad2"
reader = FARMReader(
    reader_model_name,
    context_window_size=386,
    max_seq_len=386,
    batch_size=96,
    use_gpu=True,
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

    # Faz uma predição com o modelo de leitura do Haystack
    prediction = reader.predict(
        query=data["query"],
        documents=[Document.from_dict({"content": page_content})],
    )

    # Obtém a primeira predição de resposta do modelo
    answer_prediction = prediction["answers"][0]

    # Obtém a pontuação de confiança da resposta
    answer_confidence_score = answer_prediction.score

    summary = summarize_large_text(page_content)

    # Faz uma predição de notícias falsas
    fake_news_prediction = predict_fake_news(page_title, summary)

    # Obtém a pontuação de confiança na notícia real
    trust_score = fake_news_prediction["Real"]

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
                    "summary": summary,
                }
            )


# Inicializando o servidor Flask para escutar em um determinado endereço IP e porta, com o modo de depuração ativado.
if __name__ == "__main__":
    app.run(host="172.18.92.237", port=5000, debug=True)
