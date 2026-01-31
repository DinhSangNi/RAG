"""
Script to crawl HTML content from Wikipedia
Downloads and saves Wikipedia pages as HTML files
"""

import wikipedia

# Set language to Vietnamese
wikipedia.set_lang("vi")

# Search for pages related to "Nhà Hậu Lê"
page_names = wikipedia.search("Mahatma Gandhi")
print(f"Tìm thấy {len(page_names)} trang liên quan đến 'Mahatma Gandhi':") 

for page_name in page_names:
    print(f"- {page_name}")

# Get the first page and save as HTML
page = wikipedia.page(page_names[0])
stored_dir = "src/raw_html_files"

with open(f"{stored_dir}/Mahatma_Gandhi.html", "w", encoding="utf-8") as f:
    f.write(page.html())


