# Importa os módulos necessários
from haystack.schema import Document
from haystack.nodes import (
    PreProcessor,
    TransformersDocumentClassifier,
)

# Criando uma instância do Haystack para pré-processar os documentos.
doc_preprocessor = PreProcessor()

# Carrega um modelo do Haystack para classificar notícias reais e falsas
doc_classifier_model_name = "hamzab/roberta-fake-news-classification"
doc_classifier = TransformersDocumentClassifier(
    model_name_or_path=doc_classifier_model_name, use_gpu=True, top_k=None
)

# Exemplo de título de uma página web.
title = "About fake news"

# Exemplo de conteúdo da mesma página.
content = """
"Fake news" is a term that refers to false or misleading information.
[...]
"""

# Pré-processa o conteúdo da página web dividindo em documentos do Haystack
docs = doc_preprocessor.process(Document.from_dict({"content": content}))

# Prepara os documentos para classificação de notícias reais e falsas
for doc in docs:
    doc.content = "<title>" + title + "<content>" + doc.content + "<end>"

# Faz uma predição de classificação de notícias reais e falsas
processed_classifier_documents = doc_classifier.predict(documents=docs)

# Calcula a pontuação de confiança a partir dos documentos classificados
trust_score = 0
for doc in docs:
    trust_score += doc.meta["classification"]["details"]["TRUE"]
trust_score = trust_score / len(docs)

print(trust_score)
# Exemplo de saída: 0.986523...
