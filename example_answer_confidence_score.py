# Carrega componentes necessários da biblioteca Haystack.
from haystack.nodes import FARMReader, PreProcessor
from haystack.schema import Document
from haystack.utils import print_answers

# Define o nome do modelo a ser utilizado: "deepset/roberta-base-squad2".
reader_model_name = "deepset/roberta-base-squad2"

# Cria uma instância do FARMReader com o modelo definido solicitando o 
# uso da GPU para as predições.
reader = FARMReader(reader_model_name, use_gpu=True)

# Exemplo de conteúdo de uma página web.
page_content = '''
FARMReader is a component developed by Deepset.
[...]
'''

# Exemplo de uma pergunta.
query = "Who developed FARMReader?"

# Pré-processa o conteúdo da página web dividindo em documentos do Haystack.
doc_preprocessor = PreProcessor()
docs = doc_preprocessor.process(Document.from_dict({"content": page_content}))

# Faz uma predição com o modelo de leitura do Haystack
prediction = reader.predict(
    query=query,
    documents=[docs],
)

# Obtém a primeira predição de resposta do modelo
answer_prediction = prediction["answers"][0]
print_answers(prediction, details="all")

# Obtém e exibe a pontuação de confiança da resposta
answer_confidence_score = answer_prediction.score
print(answer_confidence_score)
# Exemplo de saída: 0.998

