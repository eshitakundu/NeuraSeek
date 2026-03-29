from typing import List
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def extract_text_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_text_from_excel(self, file_path: str) -> str:
        df = pd.read_excel(file_path)
        return df.to_string()
    
    def extract_text_from_csv(self, file_path: str) -> str:
        df = pd.read_csv(file_path)
        return df.to_string()
    
    def process_document(self, file_path: str, file_type: str) -> List[str]:
        # Extract text based on file type
        extractors = {
            'pdf': self.extract_text_from_pdf,
            'docx': self.extract_text_from_docx,
            'txt': self.extract_text_from_txt,
            'xlsx': self.extract_text_from_excel,
            'csv': self.extract_text_from_csv,
        }
        
        if file_type not in extractors:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        text = extractors[file_type](file_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        return chunks