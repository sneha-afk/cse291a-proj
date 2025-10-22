'''
plumbertxtextract.py takes the csv output from pdfplumber, extracts the 'char' column and combines characters into a txt file.

replace 'input.csv' with location of pdfplumber csv
replace 'output.txt' with location of txt conversion
'''

import csv
import sys

input_csv = "input.csv"
output_txt = "output.txt"

# Increase field size limit to handle huge CSV cells
csv.field_size_limit(sys.maxsize)

chars = []

with open(input_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row.get("object_type") == "char" and row.get("text"):
            chars.append(row["text"])

# Join all the characters
text_content = "".join(chars)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(text_content)

print(f"Extracted {len(chars)} characters to {output_txt}")
