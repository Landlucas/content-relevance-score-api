import os
import requests
import torch
from haystack.schema import Document
from haystack.nodes import FARMReader
from flask import Flask, request, jsonify
from trafilatura import fetch_url, extract, extract_metadata
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()
moz_username = os.getenv("MOZ_API_ID")
moz_password = os.getenv("MOZ_API_KEY")
moz_endpoint = "https://lsapi.seomoz.com/v2/url_metrics"


def extract_web_page_content(url):
    html_content = fetch_url(url)
    return extract(html_content)


def extract_web_page_metrics(url):
    data = {"targets": [url]}
    return requests.post(moz_endpoint, auth=(moz_username, moz_password), json=data)


def predict_fake_news(title, text):
    tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
    model = AutoModelForSequenceClassification.from_pretrained(
        "hamzab/roberta-fake-news-classification"
    )
    input_str = "<title>" + title + "<content>" + text + "<end>"
    input_ids = tokenizer.encode_plus(
        input_str,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    with torch.no_grad():
        output = model(
            input_ids["input_ids"].to(device),
            attention_mask=input_ids["attention_mask"].to(device),
        )
    return dict(
        zip(
            ["Fake", "Real"],
            [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])],
        )
    )


reader_model_name = "deepset/roberta-base-squad2"
reader = FARMReader(reader_model_name, use_gpu=True)

app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    data = request.json

    if data is None or "query" not in data or "url" not in data:
        return jsonify({"error": "Invalid request data"}), 400

    authority_response = extract_web_page_metrics(data["url"])

    if authority_response.status_code != 200:
        return jsonify({"error": "Error with page authority lookup"}), 400

    page_authority_score = authority_response.json()["results"][0]["page_authority"]
    page_title = authority_response.json()["results"][0]["title"]

    if page_title is None:
        return (
            jsonify({"error": "URL provided did not return any title"}),
            400,
        )

    page_content = extract_web_page_content(data["url"])

    if page_content is None:
        return (
            jsonify({"error": "URL provided did not return any content"}),
            400,
        )

    prediction = reader.predict(
        query=data["query"],
        documents=[Document.from_dict({"content": page_content})],
    )
    answer_prediction = prediction["answers"][0]
    answer_confidence_score = answer_prediction.score

    fake_news_prediction = predict_fake_news(page_title, page_content)
    trust_score = fake_news_prediction["Real"]

    if prediction["answers"]:
        if prediction["answers"][0]:
            return jsonify(
                {
                    "query": data["query"],
                    "content": page_content,
                    "url": data["url"],
                    "answer_confidence_score": answer_confidence_score,
                    "trust_score": trust_score,
                    "page_authority_score": page_authority_score,
                }
            )


if __name__ == "__main__":
    app.run(host="172.18.92.237", port=5000, debug=True)
