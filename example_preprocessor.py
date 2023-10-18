# Importa os módulos necessários.
from haystack.schema import Document
from haystack.nodes import (
    PreProcessor,
)

# Cria uma instância do Haystack para pré-processar os documentos.
doc_preprocessor = PreProcessor(
    clean_whitespace=True, # Remove espaços em branco duplicados
    split_length=60, # Seta o tamanho máximo de cada parte para 56 caracteres
    split_respect_sentence_boundary=True, # Respeita os limites das sentenças
)

# Exemplo de conteúdo.
content = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Vix probo dicam neglegentur an, viris latine ex mel.
"""

# Pré-processa o conteúdo da página web dividindo em documentos do Haystack.
processed_docs = doc_preprocessor.process(Document.from_dict({"content": content}))

print(processed_docs)
# Exemplo de saída de documentos:
# [
#     {
#         "content": "\nLorem ipsum dolor sit amet, consectetur adipiscing elit.",
#         ...
#     },
#     ...
#     {
#         "content": "\nVix probo dicam neglegentur an, viris latine ex mel.",
#         ...
#     }
# ]


