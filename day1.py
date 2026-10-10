from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)
    return pages

pages = load_pdf("data/handbook_30pages.pdf")
print(len(pages))

splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks = splitter.split_text(pages[9])
print(len(chunks))
print(chunks[0])
