from trafilatura import fetch_url, extract

def extract_web_page_content(url):
    html_content = fetch_url(url)
    return extract(html_content)

extracted_text = extract_web_page_content("https://www.oracle.com/artificial-intelligence/machine-learning/what-is-machine-learning/")

file_path = "example_trafilatura_extracted_text.txt"

with open(file_path, "w", encoding="utf-8") as file:
    file.write(extracted_text)

print(extracted_text)
print(f"Extracted size: {len(extracted_text)}")