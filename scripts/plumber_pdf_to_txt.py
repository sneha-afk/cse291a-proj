'''
pdf_line_extractor.py - Extract text from PDF using pdfplumber line extraction

Usage:
    python pdf_line_extractor.py input.pdf output.txt
    python pdf_line_extractor.py input.pdf output.txt --pages 1-5,10,15-20
'''

import pdfplumber
import sys
import argparse

# Parse out the ranges and make it a set of pages to read through
def parse_page_ranges(range_string):
    pages = set()
    for part in range_string.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return pages

def extract_lines_from_pdf(pdf_path, output_path, page_ranges=None):
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_ranges and page_num not in page_ranges:
                continue

            all_text.append(f"\n--- Page {page_num} ---\n")

            text = page.extract_text(x_tolerance=1, layout=False)

            if text:
                all_text.append(text)

            tables = page.extract_tables()
            if tables:
                all_text.append("\n[TABLE START]\n")
                for table_idx, table in enumerate(tables, start=1):
                    all_text.append(format_table_as_plain_text(table))
                all_text.append("\n[TABLE END]\n")

    text_content = "\n".join(all_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"Extracted text from {len(pdf.pages)} pages")
    print(f"Output written to: {output_path}")

def format_table_as_plain_text(table):
    if not table or len(table) == 0:
        return ""

    lines = []
    for row in table:
        row_text = "\t".join(str(cell or "").strip() for cell in row)
        if row_text.strip():  # Only add non-empty rows
            lines.append(row_text)

    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from PDF')
    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('output_txt', help='Output text file')
    parser.add_argument('--pages', help='Page ranges to extract (e.g., 1-5,10,15-20)', default=None)

    args = parser.parse_args()
    page_ranges = parse_page_ranges(args.pages) if args.pages else None

    extract_lines_from_pdf(args.input_pdf, args.output_txt, page_ranges)
