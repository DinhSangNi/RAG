"""
HTML to Markdown Converter Functions
Core functions for HTML cleaning, conversion, and Markdown normalization
Used in the RAG pipeline
"""

import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import markitdown
except ImportError:
    print("❌ Error: markitdown not installed. Install with: pip install markitdown")
    sys.exit(1)


def clean_wikipedia_html(html_content):
    """
    Clean Wikipedia HTML content by removing unwanted elements
    Same logic as used in the RAG pipeline

    Args:
        html_content: Raw HTML content

    Returns:
        Cleaned HTML content
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Remove navigation elements
    for nav in soup.find_all(['nav', 'navigation', 'navbar']):
        nav.decompose()

    # Remove footer
    for footer in soup.find_all('footer'):
        footer.decompose()

    # Remove sidebar
    for sidebar in soup.find_all(class_=re.compile(r'sidebar|side-bar|navbox')):
        sidebar.decompose()

    # Remove table of contents if it exists
    toc = soup.find(id=re.compile(r'toc|table-of-contents'))
    if toc:
        toc.decompose()

    # Remove edit links and other Wikipedia-specific elements
    for edit_link in soup.find_all('span', class_='mw-editsection'):
        edit_link.decompose()

    # Remove references section
    for ref_section in soup.find_all(id=re.compile(r'references|footnotes')):
        ref_section.decompose()

    # Remove categories at bottom
    for catlinks in soup.find_all(id='catlinks'):
        catlinks.decompose()

    # Remove external links section
    for ext_links in soup.find_all(class_=re.compile(r'external|ext-links')):
        ext_links.decompose()

    return str(soup)


def normalize_markdown(md_text):
    """
    Normalize markdown text using the same logic as the RAG pipeline

    Args:
        md_text: Raw markdown text

    Returns:
        Normalized markdown text
    """
    lines = md_text.split('\n')
    normalized_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Nếu là header, giữ nguyên
        if line.strip().startswith('#'):
            normalized_lines.append(line)
            i += 1
            continue

        # Nếu là bullet point, table, hoặc dòng trống, giữ nguyên
        if (line.strip().startswith(('* ', '- ', '+ ', '|', '>')) or
            re.match(r'^\s*\d+\.', line) or
            line.strip() == '' or
            line.strip().startswith(':')):
            normalized_lines.append(line)
            i += 1
            continue

        # Gộp các dòng liên tiếp không phải là đoạn đặc biệt
        paragraph = line
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # Dừng nếu gặp dòng trống, header, bullet, table
            if (next_line.strip() == '' or
                next_line.strip().startswith(('#', '* ', '- ', '+ ', '|', '>', ':')) or
                re.match(r'^\s*\d+\.', next_line)):
                break
            # Gộp dòng
            paragraph += ' ' + next_line.strip()
            i += 1

        normalized_lines.append(paragraph)

    # Join và làm sạch khoảng trắng thừa
    result = '\n'.join(normalized_lines)

    # Chuẩn hóa bullet points: chuyển tất cả thành *
    result = re.sub(r'^\s*[-+]\s+', '* ', result, flags=re.MULTILINE)

    # Loại bỏ khoảng trắng thừa ở cuối dòng
    result = re.sub(r' +\n', '\n', result)

    # Chuẩn hóa block quotes: chuyển : thành >
    result = re.sub(r'^:   \*', '>   *', result, flags=re.MULTILINE)
    result = re.sub(r'^:\s+', '> ', result, flags=re.MULTILINE)

    # Đảm bảo có dòng trống trước header
    result = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', result)
    # Loại bỏ nhiều dòng trống liên tiếp (giữ tối đa 2)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def convert_html_to_markdown(html_content, clean_html=True):
    """
    Convert HTML content to normalized Markdown

    Args:
        html_content: Raw HTML content (string or file path)
        clean_html: Whether to clean HTML before conversion

    Returns:
        Normalized markdown content
    """
def convert_html_to_markdown(html_content, clean_html=True):
    """
    Convert HTML content to normalized Markdown

    Args:
        html_content: Raw HTML content (string or file path)
        clean_html: Whether to clean HTML before conversion

    Returns:
        Normalized markdown content
    """
    import tempfile
    import os

    # Check if html_content is a file path
    is_file_path = False
    if isinstance(html_content, str):
        # Simple heuristic: if it looks like HTML content (contains <html or <body), treat as content
        # Otherwise, try to treat as file path
        if not (html_content.strip().startswith('<') and ('<html' in html_content.lower() or '<body' in html_content.lower())):
            # Try to read as file path
            try:
                with open(html_content, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                html_content = file_content
                is_file_path = True
            except (FileNotFoundError, OSError):
                # Not a file path, treat as HTML content
                pass

    # Clean HTML if requested and not a file path
    if clean_html and not is_file_path:
        html_content = clean_wikipedia_html(html_content)

    # Convert to markdown using temporary file for HTML content
    md_converter = markitdown.MarkItDown()

    if is_file_path:
        # If it's a file path, convert directly
        result = md_converter.convert(html_content)
    else:
        # For HTML content string, create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(html_content)
            temp_file_path = temp_file.name

        try:
            result = md_converter.convert(temp_file_path)
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    raw_md = result.text_content

    # Normalize markdown
    normalized_md = normalize_markdown(raw_md)

    return normalized_md


def convert_html_file_to_markdown(html_file_path, output_md_file_path=None, clean_html=True):
    """
    Convert HTML file to Markdown file

    Args:
        html_file_path: Path to HTML file
        output_md_file_path: Optional output path for markdown file
        clean_html: Whether to clean HTML before conversion

    Returns:
        Path to the created markdown file
    """
    # Read HTML file
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        raise Exception(f"Error reading HTML file {html_file_path}: {e}")

    # Convert to markdown
    markdown_content = convert_html_to_markdown(html_content, clean_html)

    # Determine output path
    if output_md_file_path is None:
        input_path = Path(html_file_path)
        output_md_file_path = str(input_path.parent / f"{input_path.stem}.md")

    # Create output directory if needed
    output_dir = Path(output_md_file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write markdown file
    try:
        with open(output_md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
    except Exception as e:
        raise Exception(f"Error writing markdown file {output_md_file_path}: {e}")

    return output_md_file_path


# Configuration - Change these variables to convert specific files
INPUT_HTML_FILE = "Nguyễn_Hữu_An.html"  # File name in raw_html_files/
OUTPUT_MD_FILE = "Nguyễn_Hữu_An.md"     # Output file name in processed_markdown/

# Example usage
if __name__ == "__main__":
    """
    Convert a specific HTML file from raw_html_files/ to processed_markdown/
    Change INPUT_HTML_FILE and OUTPUT_MD_FILE above to convert different files
    """

    # Define directories
    raw_html_dir = Path(__file__).parent / "raw_html_files"
    processed_md_dir = Path(__file__).parent / "processed_markdown"

    # Input and output paths
    input_html_path = raw_html_dir / INPUT_HTML_FILE
    output_md_path = processed_md_dir / OUTPUT_MD_FILE

    # Check if input file exists
    if not input_html_path.exists():
        print(f"❌ Input file not found: {input_html_path}")
        print(f"Available files in {raw_html_dir}:")
        if raw_html_dir.exists():
            html_files = list(raw_html_dir.glob("*.html"))
            if html_files:
                for f in html_files:
                    print(f"  - {f.name}")
            else:
                print("  (no HTML files found)")
        sys.exit(1)

    # Create output directory if it doesn't exist
    processed_md_dir.mkdir(exist_ok=True)

    try:
        print(f"📄 Converting: {INPUT_HTML_FILE}")
        print(f"📂 From: {raw_html_dir}")
        print(f"🎯 To: {processed_md_dir}")
        print("-" * 50)

        # Convert file
        result_path = convert_html_file_to_markdown(
            html_file_path=str(input_html_path),
            output_md_file_path=str(output_md_path),
            clean_html=True
        )

        print("-" * 50)
        print(f"✅ Successfully converted!")
        print(f"📄 Input:  {input_html_path}")
        print(f"📝 Output: {result_path}")

        # Show file size info
        input_size = input_html_path.stat().st_size
        output_size = Path(result_path).stat().st_size
        print(f"📊 Size:   {input_size:,} bytes → {output_size:,} bytes")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)