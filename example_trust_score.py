# Carrega a ferramenta pyTorch para inferência do modelo
import torch

# Carrega o AutoTokenizer para o processo de tokenização de texto
# Carrega o AutoModel para inicializar automaticamente o modelo dado seu nome
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Exemplo de título de uma página web.
title = "About fake news"

# Exemplo de conteúdo da mesma página.
text = """
"Fake news" is a term that refers to false or misleading information.
[...]
"""

# Inicializa o tokenizer com o vocabulário do modelo de classificação de notícias falsas
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")

# Inicializa o modelo de classificação de notícias falsas
model = AutoModelForSequenceClassification.from_pretrained(
    "hamzab/roberta-fake-news-classification"
)

# Combina o título e o texto da notícia em uma única string formatada
# de acordo com os requisitos do modelo
input_str = "<title>" + title + "<content>" + text + "<end>"

# Converte a string em tokens e os codifica
input_ids = tokenizer.encode_plus(
    input_str,
    max_length=512,  # Define um comprimento máximo para o input
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

# Converte as probabilidades de saída em um formato legível:
# Fake e Real, ambos com valores entre 0 e 1.
fake_news_prediction = (
    dict(
        zip(
            ["Fake", "Real"],
            [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])],
        )
    )
)
# Exemplo de estrutura: {'Fake': 0.02, 'Real': 0.98}

# Obtém a pontuação de confiança na notícia real
trust_score = fake_news_prediction["Real"]
print(trust_score)
# Exemplo de saída: 0.98