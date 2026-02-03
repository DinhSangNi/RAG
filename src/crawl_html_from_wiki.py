import wikipedia

wikipedia.set_lang("vi")  # Thiết lập ngôn ngữ tiếng Việt
page_names = wikipedia.search("Nhà Nguyễn")
print(f"Tìm thấy {len(page_names)} trang liên quan đến 'Nhà Nguyễn':")
for page_name in page_names:
    print(f"- {page_name}")

page = wikipedia.page(page_names[0])
stored_dir = "src/raw_html_files"
with open(f"{stored_dir}/Nguyễn_dynasty.html", "w", encoding="utf-8") as f:
    f.write(page.html())


