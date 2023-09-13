# import json
# import os
# import requests
from haystack.schema import Document
from haystack.nodes import FARMReader, TransformersQueryClassifier
from flask import Flask, request, jsonify
from trafilatura import fetch_url, extract
from dotenv import load_dotenv

load_dotenv()


def extract_web_page_content(url):
    downloaded = fetch_url(url)
    return extract(downloaded)


reader_model_name = "deepset/roberta-base-squad2"
reader = FARMReader(reader_model_name, use_gpu=True)

statement_classifier_model_name = "jy46604790/Fake-News-Bert-Detect"
# statement_classifier_model_name = "typeform/distilbert-base-uncased-mnli"
statement_classifier = TransformersQueryClassifier(model_name_or_path=statement_classifier_model_name, use_gpu=True)

app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    data = request.json

    if data is None or "query" not in data or "url" not in data:
        return jsonify({"error": "Invalid request data"}), 400

    content = extract_web_page_content(data["url"])

    if content is None:
        return jsonify({"error": "URL provided did not return any content"}), 400

    prediction = reader.predict(
        query=data["query"],
        documents=[Document.from_dict({"content": content})],
    )
    answer_prediction = prediction["answers"][0]

    fake_news_prediction = statement_classifier.run(query=content)

    if prediction["answers"]:
        if prediction["answers"][0]:
            return jsonify(
                {
                    "query": data["query"],
                    "url": data["url"],
                    "answer_prediction": answer_prediction,
                    "fake_news_prediction": fake_news_prediction,
                }
            )


if __name__ == "__main__":
    app.run(host="172.18.92.237", port=5000, debug=True)
