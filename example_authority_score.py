# Carrega pacote para receber variáveis de ambiente do sistema.
import os

# Carrega pacote para realizar requisições em APIs externas.
import requests

# Obtendo as credenciais para a API Moz e configurando o endpoint da API.
moz_username = os.getenv("MOZ_API_ID")  # Moz Access ID
moz_password = os.getenv("MOZ_API_KEY")  # Moz Secret Key
moz_endpoint = (
    "https://lsapi.seomoz.com/v2/url_metrics"  # Moz Links API URL Metrics Endpoint
)

# Define uma URL para a qual deseja extrair métricas
url = "https://feevale.br/"

# Monta o payload da requisição de acordo com a documentação da API.
data = {"targets": [url]}

# Realiza a requisição POST para o endpoint de URL Metrics com autenticação.
moz_response = requests.post(
    moz_endpoint, auth=(moz_username, moz_password), json=data
)

# Extrai a pontuação de autoridade da página e o título da resposta JSON
page_authority_score = moz_response.json()["results"][0]["page_authority"]
page_title = moz_response.json()["results"][0]["title"]

print(page_authority_score)
print(page_title)
# Exemplo de saída:
# 47
# Home | Universidade Feevale

