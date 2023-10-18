import requests
import pandas as pd

# Define a URL do serviço web
web_service_url = "http://172.18.92.237:5000"

# Carrega o dataset em formato CSV
df = pd.read_csv('serp_ai_gen_processed_dataset_sample.csv')

# Espera o pressionamento de uma tecla para iniciar as iteracoes
input("Press any key to start...")

# Itera sobre cada linha no DataFrame
for index, row in df.iterrows():
    query = row['query']
    rank = row['rank']
    url = row['url']
    
    # Cria o corpo da requisição
    request_body = {
        'query': query,
        'url': url
    }

    # Envia uma requisição POST para o serviço web
    response = requests.post(web_service_url, json=request_body)

    # Verifica se a requisição foi bem sucedida (código de status 200)
    if response.status_code == 200:
        # Extrai os dados da resposta
        response_data = response.json()
        answer_confidence_score = response_data['answer_confidence_score']
        page_authority_score = response_data['page_authority_score']
        trust_score = response_data['trust_score']
        content = response_data['content']

        # Imprime as informações
        print(f"Query: {query}, Rank: {rank}, URL: {url}")
        print(f"Answer Confidence Score: {answer_confidence_score}, Trust Score: {trust_score}, Page Authority Score: {page_authority_score}")
        # print("-" * 30)
        # print(content)
        print("-" * 30)
    else:
        print(f"Error sending request for query '{query}' and URL '{url}'")