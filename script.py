import json
import os
import requests
from haystack.schema import Document
from haystack.nodes import FARMReader
from flask import Flask, request, jsonify
from trafilatura import fetch_url, extract
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.environ.get('HF_API_TOKEN')
HF_FAKE_NEWS_API_URL = "https://api-inference.huggingface.co/models/jy46604790/Fake-News-Bert-Detect"
MOZ_API_ID = os.environ.get('MOZ_API_ID')
MOZ_API_KEY = os.environ.get('MOZ_API_KEY')

fake_news_api_headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query_fake_news(payload):
    request_body = json.dumps(payload)
    response = requests.request(
        "POST", HF_FAKE_NEWS_API_URL, headers=fake_news_api_headers, data=request_body
    )
    response_body = response.content.decode("utf-8")
    return json.loads(response_body)[0]


model_name = "deepset/roberta-base-squad2"
reader = FARMReader(model_name, use_gpu=True)

app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    data = request.json

    if data is None or "question" not in data or "documents" not in data:
        return jsonify({"error": "Invalid request data"}), 400

    documents = [
        Document.from_dict(document_dict) for document_dict in data["documents"]
    ]
    prediction = reader.predict(
        query=data["question"],
        documents=documents,
    )
    answer_prediction = prediction["answers"][0]

    fake_news_prediction = query_fake_news(data["documents"][0]["content"])

    if prediction["answers"]:
        if prediction["answers"][0]:
            return jsonify(
                {
                    "query": data["question"],
                    "content": data["documents"][0]["content"],
                    "answer_prediction": answer_prediction,
                    "fake_news_prediction": fake_news_prediction,
                }
            )


if __name__ == "__main__":
    app.run(host="172.18.92.237", port=5000, debug=True)
