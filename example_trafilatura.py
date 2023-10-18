from trafilatura import fetch_url, extract

def extract_web_page_content(url):
    html_content = fetch_url(url)
    return extract(html_content)

extracted_text = extract_web_page_content("https://web.dev/blog/webdev-migration")

print(extracted_text)
print(f"Extracted size: {len(extracted_text)}")