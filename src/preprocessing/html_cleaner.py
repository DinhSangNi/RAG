from bs4 import BeautifulSoup
from pathlib import Path
import os

def clean_wikipedia_html(html_file_path):
    with open(html_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    content = soup.find("div", class_="mw-parser-output")

    if not content:
        print("Không tìm thấy vùng nội dung chính!")
        return
    
    # Remove unwanted tags and elements
    tags_without_class = ['audio', 'style', 'img', 'sup', 'link', 'input']
    for tag in tags_without_class:
        for element in content.find_all(tag):
            element.decompose()
    
    # Remove specific tags with certain classes
    tags_with_class = [
        ('ol', 'references'),
        ('span', 'mw-editsection'),
        ('div', 'hatnote'),
        ('div', 'navbox'),
        ('div', 'navbox-styles'),
        ('div', 'metadata'),
        ('div', 'toc'), ## remove table of contents (left side) 
        ('table', 'navbox-inner'),
        ('table', 'navbox'),
        ('table', 'sidebar'),
        ('table', 'infobox'), 
        ('table', 'metadata'),
        ('span', 'languageicon'),
        ('span', 'tocnumber'),
        ('span', 'toctext'),
        ('span', 'reference-accessdate'),
        ('span', 'Z3988'),
        # ('span', 'plainlinks'),
        ('cite', None),
    ]
    
    for tag, class_name in tags_with_class:
        if class_name:
            for element in content.find_all(tag, class_=class_name):
                element.decompose()
        else:
            for element in content.find_all(tag):
                element.decompose()

    for table in content.find_all('table', class_="cquote"):
        quote_text = table.get_text(separator=" ", strip=True)
        p = soup.new_tag('p')
        p.string = quote_text
        table.replace_with(p)
    
    for p in content.find_all('p'):
        if not p.get_text(strip=True):
            p.decompose()
    
    for span in content.find_all('span'):
        if span.get('id') and not span.get_text(strip=True):
            span.decompose()
    
    for figure in content.find_all('figure'):
        figcaption = figure.find('figcaption')
        if figcaption:
            new_p = soup.new_tag('p')
            new_p.string = f"[Hình ảnh: {figcaption.get_text(strip=True)}]"
            figure.replace_with(new_p)
        else:
            figure.decompose()

    for a_tag in content.find_all('a'):
        a_tag.unwrap()
    
    for span in content.find_all('span'):
        span.unwrap()
    
    for tag in content.find_all(['b']):
        tag.unwrap()

    sections_to_kill = [
    "Tham_khảo", 
    "Tài liệu tham khảo",
    "Chú giải",
    "Liên_kết_ngoài", 
    "Danh_mục", 
    "Ghi_chú", 
    "Thư_mục_hậu_cần",
    "Đọc_thêm",
    "Chú_thích",
    "Thư_mục",
    "Nguồn_thứ_cấp",
    "Nguồn_sơ_cấp",
    "Nguồn_trích_dẫn",
    "Diễn_văn_của_Hồ_Chí_Minh",
    "Tác_phẩm_của_Hồ_Chí_Minh",
    "Viết_về_Hồ_Chí_Minh",
    "Những_người_từng_gặp_Hồ_Chí_Minh_kể_về_ông"
    ]

    for section_id in sections_to_kill:
        header = content.find(['h2', 'h3'], id=section_id)
        if header:
            for sibling in header.find_next_siblings():
                if sibling.name in ['h2', 'h3']:
                    break
                sibling.decompose() 
            header.decompose()
    
    # Remove empty <li> elements
    for li in content.find_all('li'):
        if not li.get_text(strip=True):
            li.decompose()
    
    # Clean attributes
    for tag in content.find_all(True):
        if tag.has_attr('class'):
            del tag['class']
        if tag.has_attr('style'):
            del tag['style']
        if tag.has_attr('id') and tag.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            del tag['id']
        if tag.has_attr('dir'):
            del tag['dir']
        if tag.has_attr('lang'):
            del tag['lang']

    input_file_name = Path(html_file_path).stem
    temp_html_file_path = f"data/raw_data/wikipedia/temp_clean_html/{input_file_name}.html"
    os.makedirs(os.path.dirname(temp_html_file_path), exist_ok=True)
    with open(temp_html_file_path, "w", encoding="utf-8") as f:
        f.write(str(content))

    print(f"Đã lưu HTML đã làm sạch vào: {temp_html_file_path}")

    return temp_html_file_path
