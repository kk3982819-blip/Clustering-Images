from pypdf import PdfReader
reader = PdfReader('PixelDwell AI Photo Editing Platform-technical approach document (1) (1).pdf')
text = '\n'.join([page.extract_text() for page in reader.pages])
with open('pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write(text)
