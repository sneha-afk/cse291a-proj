import re
import os

def parse_filename(file_path):
    """
    Extract company ticker and year from filename.
    Expected format: NASDAQ_AAPL_2024.txt
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Tuple of (company_ticker, year) or (None, None) if parsing fails
    """
    filename = os.path.basename(file_path)
    # Remove .txt extension
    name_without_ext = filename.replace('.txt', '')
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    if len(parts) >= 3:
        # Format: EXCHANGE_TICKER_YEAR
        company = parts[1]  # AAPL
        year = parts[2]     # 2024
        return company, year
    
    return None, None


def chunk_by_pages(file_path):
    """
    Read a text file and chunk it by page tokens.
    Returns list of tuples: (company, year, page_number, content)
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of tuples (company, year, page_number, content)
    """
    # Parse filename for company and year
    company, year = parse_filename(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by page tokens and capture page numbers
    # This regex matches "--- Page" followed by any number, followed by "---"
    parts = re.split(r'--- Page (\d+) ---', content)
    
    # parts will be: ['', '4', 'content4', '5', 'content5', '6', 'content6', ...]
    # Every odd index is a page number, every even index (except 0) is content
    
    chunks = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            page_num = parts[i]
            page_content = parts[i + 1].strip()
            
            if page_content:  # Only add non-empty pages
                chunks.append((company, year, page_num, page_content))
    
    return chunks


def main():
    # Example usage
    file_path = '../dataset/apple/NASDAQ_AAPL_2024.txt'  # Replace with your file path
    
    try:
        pages = chunk_by_pages(file_path)
        
        print(f"Found {len(pages)} pages\n")
        
        # Print information about each page
        for company, year, page_num, content in pages:
            print(f"Company: {company}, Year: {year}, Page: {page_num}")
            print(f"Content length: {len(content)} chars")
            print(f"Preview: {content[:150]}...")
            print("-" * 80)
            print()
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
