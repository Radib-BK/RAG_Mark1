"""
PDF Text Extraction Module for Multilingual RAG System

This module handles extraction of text from complex PDFs containing:
- Bangla and English text
- MCQs (Multiple Choice Questions)
- Paragraphs with various formatting
- Tables and structured content
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pdfplumber
import PyPDF2
from loguru import logger

class PDFExtractor:
    """
    Enhanced PDF text extractor for educational content in multiple languages
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        self.text_content = []
        self.metadata = {}
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def extract_text_pdfplumber(self) -> List[Dict[str, Any]]:
        """
        Extract text using pdfplumber (better for complex layouts)
        
        Returns:
            List of dictionaries containing page text and metadata
        """
        pages_content = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                logger.info(f"Extracting text from {len(pdf.pages)} pages using pdfplumber")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout preservation
                    text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=7.25,
                        y_density=13
                    )
                    
                    if text:
                        # Clean and normalize text
                        cleaned_text = self._clean_extracted_text(text)
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        table_text = self._extract_table_text(tables) if tables else ""
                        
                        page_content = {
                            'page_number': page_num,
                            'text': cleaned_text,
                            'tables': table_text,
                            'bbox': page.bbox,
                            'extraction_method': 'pdfplumber'
                        }
                        
                        pages_content.append(page_content)
                        logger.debug(f"Extracted {len(cleaned_text)} characters from page {page_num}")
                    
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {str(e)}")
            # Fallback to PyPDF2
            return self.extract_text_pypdf2()
        
        return pages_content
    
    def extract_text_pypdf2(self) -> List[Dict[str, Any]]:
        """
        Fallback extraction using PyPDF2
        
        Returns:
            List of dictionaries containing page text and metadata
        """
        pages_content = []
        
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"Fallback: Extracting text from {len(pdf_reader.pages)} pages using PyPDF2")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    
                    if text:
                        cleaned_text = self._clean_extracted_text(text)
                        
                        page_content = {
                            'page_number': page_num,
                            'text': cleaned_text,
                            'tables': "",
                            'bbox': None,
                            'extraction_method': 'pypdf2'
                        }
                        
                        pages_content.append(page_content)
                        logger.debug(f"Extracted {len(cleaned_text)} characters from page {page_num}")
                        
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {str(e)}")
            raise
        
        return pages_content
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\t', ' ', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[^\S\n]{2,}', ' ', text)  # Multiple spaces
        text = re.sub(r'\n +', '\n', text)  # Leading spaces on lines
        text = re.sub(r' +\n', '\n', text)  # Trailing spaces on lines
        
        # Normalize Bangla text (basic cleanup)
        # Remove zero-width characters and normalize Unicode
        text = text.replace('\u200c', '').replace('\u200d', '')  # Zero-width non-joiner/joiner
        text = text.replace('\ufeff', '')  # Byte order mark
        
        # Fix common OCR errors in Bangla text
        text = re.sub(r'([০-৯])([a-zA-Z])', r'\1 \2', text)  # Space between Bangla digits and English
        text = re.sub(r'([a-zA-Z])([০-৯])', r'\1 \2', text)  # Space between English and Bangla digits
        
        return text.strip()
    
    def _extract_table_text(self, tables: List[List[List[str]]]) -> str:
        """
        Convert extracted tables to readable text
        
        Args:
            tables: List of tables from pdfplumber
            
        Returns:
            Formatted table text
        """
        table_text = ""
        
        for i, table in enumerate(tables):
            if table:
                table_text += f"\nTable {i+1}:\n"
                for row in table:
                    if row:
                        # Filter out None values and join cells
                        row_text = " | ".join([str(cell) if cell else "" for cell in row])
                        table_text += row_text + "\n"
                table_text += "\n"
        
        return table_text
    
    def identify_content_types(self, text: str) -> Dict[str, bool]:
        """
        Identify types of content in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary indicating presence of different content types
        """
        content_types = {
            'has_mcq': False,
            'has_bangla': False,
            'has_english': False,
            'has_numbers': False,
            'has_equations': False
        }
        
        # Check for MCQ patterns
        mcq_patterns = [
            r'[ক-হ]\)',  # Bangla MCQ options (ক) খ) গ) ঘ)
            r'[a-d]\)',   # English MCQ options a) b) c) d)
            r'\([ক-হ]\)', # (ক) (খ) (গ) (ঘ)
            r'\([a-d]\)', # (a) (b) (c) (d)
            r'[A-D]\.',   # A. B. C. D.
            r'[১-৪]\.',   # Bangla numerals
        ]
        
        for pattern in mcq_patterns:
            if re.search(pattern, text):
                content_types['has_mcq'] = True
                break
        
        # Check for Bangla text (Unicode range for Bengali)
        if re.search(r'[\u0980-\u09FF]', text):
            content_types['has_bangla'] = True
        
        # Check for English text
        if re.search(r'[a-zA-Z]', text):
            content_types['has_english'] = True
        
        # Check for mathematical content
        if re.search(r'[০-৯0-9]+|[+\-×÷=<>≤≥∑∏∫]', text):
            content_types['has_numbers'] = True
        
        # Check for equations
        equation_patterns = [r'\$.*?\$', r'\\[a-zA-Z]+', r'[=≠≈<>≤≥]']
        for pattern in equation_patterns:
            if re.search(pattern, text):
                content_types['has_equations'] = True
                break
        
        return content_types
    
    def extract_full_document(self) -> Dict[str, Any]:
        """
        Extract complete document with metadata
        
        Returns:
            Dictionary containing all extracted content and metadata
        """
        logger.info(f"Starting full document extraction from {self.pdf_path}")
        
        # Try pdfplumber first, fallback to PyPDF2
        pages_content = self.extract_text_pdfplumber()
        
        if not pages_content:
            logger.warning("No content extracted, trying alternative method")
            pages_content = self.extract_text_pypdf2()
        
        # Combine all text for analysis
        full_text = "\n\n".join([page['text'] for page in pages_content if page['text']])
        
        # Analyze content
        content_analysis = self.identify_content_types(full_text)
        
        # Calculate basic statistics
        stats = {
            'total_pages': len(pages_content),
            'total_characters': len(full_text),
            'total_words': len(full_text.split()),
            'avg_chars_per_page': len(full_text) // max(len(pages_content), 1)
        }
        
        document_data = {
            'pages': pages_content,
            'full_text': full_text,
            'content_analysis': content_analysis,
            'statistics': stats,
            'source_file': str(self.pdf_path)
        }
        
        logger.info(f"Extraction completed: {stats['total_pages']} pages, "
                   f"{stats['total_characters']} characters, "
                   f"Bangla: {content_analysis['has_bangla']}, "
                   f"MCQ: {content_analysis['has_mcq']}")
        
        return document_data

def extract_pdf_content(pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract PDF content
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Optional directory to save extracted text
        
    Returns:
        Extracted document data
    """
    extractor = PDFExtractor(pdf_path)
    document_data = extractor.extract_full_document()
    
    # Optionally save to file
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save full text
        text_file = output_path / f"{Path(pdf_path).stem}_extracted.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(document_data['full_text'])
        
        logger.info(f"Extracted text saved to {text_file}")
    
    return document_data

if __name__ == "__main__":
    # Example usage
    pdf_path = "../data/HSC26-Bangla1st-Paper.pdf"
    
    if os.path.exists(pdf_path):
        try:
            document_data = extract_pdf_content(pdf_path, "../data/extracted/")
            print(f"Successfully extracted {document_data['statistics']['total_pages']} pages")
            print(f"Content analysis: {document_data['content_analysis']}")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please place your HSC textbook PDF in the data/ directory") 