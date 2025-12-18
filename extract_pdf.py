import PyPDF2
import sys

pdf_path = r"c:\Users\moham\Desktop\ML - Project [Fall 2025].pdf"

with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    num_pages = len(reader.pages)
    
    # Write to file to avoid encoding issues
    output_file = "pdf_content.txt"
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"PDF has {num_pages} pages\n")
        out.write("="*80 + "\n")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            out.write(f"\n--- PAGE {i+1} ---\n\n")
            out.write(text)
            out.write("\n" + "="*80 + "\n")
    
    print(f"PDF content extracted to {output_file}")
