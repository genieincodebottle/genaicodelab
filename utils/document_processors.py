import PyPDF2
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List
from io import BytesIO
import trafilatura
import validators

class DocumentProcessor:
    """Process different types of documents and extract text content."""
    
    @staticmethod
    def process_pdf(file_content: bytes) -> List[str]:
        """Process PDF file and extract text content."""
        documents = []
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                documents.extend(paragraphs)
        
        return documents

    @staticmethod
    def process_csv(file_content: bytes) -> List[str]:
        """Process CSV file and extract text content."""
        df = pd.read_csv(BytesIO(file_content))
        documents = []
        
        # Convert each row to a string representation
        for _, row in df.iterrows():
            # Join all non-null values in the row
            row_text = ' | '.join(str(val) for val in row if pd.notna(val))
            if row_text.strip():
                documents.append(row_text)
        
        return documents

    @staticmethod
    def process_text(file_content: bytes) -> List[str]:
        """Process text file and extract content."""
        text = file_content.decode('utf-8', errors='ignore')
        # Split by double newline to separate paragraphs
        documents = [doc.strip() for doc in text.split('\n\n') if doc.strip()]
        return documents

    @staticmethod
    def process_webpage(url: str) -> List[str]:
        """Process webpage and extract main content."""
        if not validators.url(url):
            raise ValueError("Invalid URL provided")
            
        try:
            # Use trafilatura for main content extraction
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                # Extract main content and remove boilerplate
                text = trafilatura.extract(downloaded, include_links=False, 
                                        include_images=False, 
                                        include_tables=False)
                if text:
                    # Split into paragraphs
                    documents = [p.strip() for p in text.split('\n\n') if p.strip()]
                    return documents
            
            # Fallback to basic BeautifulSoup extraction if trafilatura fails
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text and split into paragraphs
            text = soup.get_text()
            documents = [p.strip() for p in text.split('\n\n') if p.strip()]
            return documents
            
        except Exception as e:
            raise Exception(f"Error processing webpage: {str(e)}")

    @staticmethod
    def get_documents_from_file(uploaded_file) -> List[str]:
        """Process uploaded file based on its type."""
        file_content = uploaded_file.getvalue()
        file_type = uploaded_file.type
        
        if 'pdf' in file_type:
            return DocumentProcessor.process_pdf(file_content)
        elif 'csv' in file_type:
            return DocumentProcessor.process_csv(file_content)
        elif 'text' in file_type:
            return DocumentProcessor.process_text(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")