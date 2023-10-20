import requests
import pandas as pd

web_service_url = "http://172.18.92.237:5000"

df = pd.read_csv("serp_client_test_sample.csv")

input("Press any key to start...")

result_data = []

for index, row in df.iterrows():
    query = row["query"]
    rank = row["rank"]
    url = row["url"]

    request_body = {"query": query, "url": url}

    response = requests.post(web_service_url, json=request_body)

    if response.status_code == 200:
        response_data = response.json()
        answer_confidence_score = response_data["answer_confidence_score"]
        page_authority_score = response_data["page_authority_score"]
        trust_score = response_data["trust_score"]
        content = response_data["content"]

        result_data.append({
            "query": query,
            "rank": rank,
            "url": url,
            "answer_confidence_score": answer_confidence_score,
            "trust_score": trust_score,
            "page_authority_score": page_authority_score
        })

        print("-" * 30)
        print(f"Query: {query}, Rank: {rank}, URL: {url}")
        print(
            f"Answer Confidence Score: {answer_confidence_score}, Trust Score: {trust_score}, Page Authority Score: {page_authority_score}"
        )
    else:
        print(f"Error sending request for query '{query}' and URL '{url}'")

print("-" * 30)

result_df = pd.DataFrame(result_data)

result_df.to_csv("serp_client_results.csv", index=False)