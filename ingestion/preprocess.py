"""
Text Preprocessing and Chunking Module for Multilingual RAG System

This module handles:
- Text cleaning and normalization
- Intelligent chunking strategies (paragraph, sentence, semantic)
- Overlap management for better context preservation
- Multilingual text processing (Bangla + English)
- MCQ and structured content handling
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import nltk
from loguru import logger
from langdetect import detect, LangDetectException

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class TextChunk:
    """
    Data class for text chunks with metadata
    """
    content: str
    chunk_id: str
    source_page: int
    chunk_index: int
    language: str
    content_type: str
    word_count: int
    char_count: int
    overlap_start: int = 0
    overlap_end: int = 0

class TextPreprocessor:
    """
    Advanced text preprocessing and chunking for multilingual educational content
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Initialize text preprocessor
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Compile regex patterns for efficiency
        self.bangla_sentence_pattern = re.compile(r'[।!?]+\s*')
        self.english_sentence_pattern = re.compile(r'[.!?]+\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n+')
        # Fix: Use explicit Bangla MCQ options instead of invalid range
        self.mcq_pattern = re.compile(r'([কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়a-dA-D১-৪1-4][\)\. ]\s*[^\n]+)', re.MULTILINE)
        
        logger.info(f"TextPreprocessor initialized with chunk_size={chunk_size}, "
                   f"overlap={chunk_overlap}, min_size={min_chunk_size}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code ('bn' for Bangla, 'en' for English, 'mixed' for multilingual)
        """
        if not text or len(text.strip()) < 10:
            return 'unknown'
        
        try:
            # Check for Bangla characters
            bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
            total_chars = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text))
            
            if total_chars == 0:
                return 'unknown'
            
            bangla_ratio = bangla_chars / total_chars
            
            if bangla_ratio > 0.7:
                return 'bn'  # Primarily Bangla
            elif bangla_ratio > 0.3:
                return 'mixed'  # Mixed language
            else:
                # Try langdetect for English/other languages
                try:
                    detected = detect(text)
                    return detected if detected in ['en', 'bn'] else 'en'
                except LangDetectException:
                    return 'en'  # Default to English
                    
        except Exception as e:
            logger.debug(f"Language detection error: {e}")
            return 'unknown'
    
    def identify_content_type(self, text: str) -> str:
        """
        Identify the type of content - optimized for HSC note structure
        
        Args:
            text: Text to analyze
            
        Returns:
            Content type string
        """
        text_lower = text.lower().strip()
        text_lines = text.split('\n')
        
        # Check for answer table (common patterns in HSC answer sections)
        answer_indicators = ['উত্তর', 'answer', 'ans:', 'উঃ', 'সঠিক উত্তর']
        if any(indicator in text_lower for indicator in answer_indicators) and len(text_lines) > 3:
            return 'answer_table'
        
        # Check for MCQ with enhanced patterns for HSC format
        mcq_patterns = [
            r'[ক-হ]\)',          # Bengali MCQ options
            r'[a-d]\)',           # English MCQ options  
            r'\([ক-হ]\)',        # Bengali with parentheses
            r'\([a-d]\)',         # English with parentheses
            r'[১-৪]\.',           # Bengali numerals
            r'প্রশ্ন\s*[:\-]',    # Question indicator
            r'Question\s*[:\-]'   # English question
        ]
        
        mcq_count = sum(1 for pattern in mcq_patterns if re.search(pattern, text))
        if mcq_count > 0 or self.mcq_pattern.search(text):
            return 'mcq'
        
        # Check for table content (enhanced for HSC tables)
        if ('|' in text and text.count('|') > 2) or \
           (len(text_lines) > 2 and all(len(line.split()) > 2 for line in text_lines[:3])):
            return 'table'
        
        # Check for section headings (common in HSC notes)
        heading_patterns = [
            r'^\d+\.\d+',          # 1.1, 2.3 etc
            r'^[অআইঈউঊএঐওঔ]',    # Bengali section markers
            r'^অধ্যায়',             # Chapter
            r'^পরিচ্ছেদ',           # Section
            r'^বিষয়',              # Topic
        ]
        
        if len(text.strip()) < 150 and any(re.search(pattern, text) for pattern in heading_patterns):
            return 'heading'
        
        # Check for list items
        if re.search(r'^[\s]*[•\-\*]\s', text, re.MULTILINE):
            return 'list'
        
        # Check for mathematical content
        math_indicators = ['=', '≠', '≈', '<', '>', '≤', '≥', '+', '-', '×', '÷', '∑', '∏', '∫']
        if any(indicator in text for indicator in math_indicators) and len(text) < 200:
            return 'equation'
        
        # Check for definitions (common in educational content)
        if any(word in text_lower for word in ['সংজ্ঞা', 'definition', 'অর্থ', 'meaning']) and len(text) < 300:
            return 'definition'
        
        return 'paragraph'
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common OCR errors
        text = text.replace('।।', '।')  # Double dari
        text = text.replace('??', '?')   # Double question marks
        text = text.replace('!!', '!')   # Double exclamation marks
        
        # Normalize punctuation spacing
        text = re.sub(r'\s+([।!?,.;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([।!?])\s*', r'\1 ', text)     # Add space after sentence endings
        
        # Fix spacing around brackets
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str, language: str) -> List[str]:
        """
        Split text into sentences based on language
        
        Args:
            text: Text to split
            language: Language code
            
        Returns:
            List of sentences
        """
        sentences = []
        
        if language == 'bn' or language == 'mixed':
            # Use Bangla sentence pattern
            parts = self.bangla_sentence_pattern.split(text)
            sentences.extend([s.strip() for s in parts if s.strip()])
        
        if language == 'en' or language == 'mixed':
            # Use NLTK for English
            try:
                nltk_sentences = nltk.sent_tokenize(text)
                sentences.extend(nltk_sentences)
            except:
                # Fallback to simple regex
                parts = self.english_sentence_pattern.split(text)
                sentences.extend([s.strip() for s in parts if s.strip()])
        
        if not sentences:
            # Fallback: split by any sentence-ending punctuation
            sentences = re.split(r'[।.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_by_paragraphs(self, text: str, page_num: int = 0) -> List[TextChunk]:
        """
        Chunk text by paragraphs
        
        Args:
            text: Text to chunk
            page_num: Source page number
            
        Returns:
            List of TextChunk objects
        """
        paragraphs = self.paragraph_pattern.split(text)
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < self.min_chunk_size:
                continue
            
            language = self.detect_language(paragraph)
            content_type = self.identify_content_type(paragraph)
            
            # If paragraph is too long, split it further
            if len(paragraph) > self.chunk_size:
                sub_chunks = self._split_long_text(paragraph, page_num, i, language, content_type)
                chunks.extend(sub_chunks)
            else:
                chunk = TextChunk(
                    content=paragraph,
                    chunk_id=f"page_{page_num}_para_{i}",
                    source_page=page_num,
                    chunk_index=i,
                    language=language,
                    content_type=content_type,
                    word_count=len(paragraph.split()),
                    char_count=len(paragraph)
                )
                chunks.append(chunk)
        
        return chunks
    
    def chunk_by_sentences(self, text: str, page_num: int = 0) -> List[TextChunk]:
        """
        Chunk text by sentences with overlap
        
        Args:
            text: Text to chunk
            page_num: Source page number
            
        Returns:
            List of TextChunk objects
        """
        language = self.detect_language(text)
        sentences = self.split_into_sentences(text, language)
        chunks = []
        
        current_chunk = ""
        current_sentences = []
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_content = current_chunk.strip()
                if len(chunk_content) >= self.min_chunk_size:
                    content_type = self.identify_content_type(chunk_content)
                    
                    chunk = TextChunk(
                        content=chunk_content,
                        chunk_id=f"page_{page_num}_sent_{chunk_index}",
                        source_page=page_num,
                        chunk_index=chunk_index,
                        language=language,
                        content_type=content_type,
                        word_count=len(chunk_content.split()),
                        char_count=len(chunk_content)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) > 1 else current_sentences
                current_chunk = " ".join(overlap_sentences + [sentence])
                current_sentences = overlap_sentences + [sentence]
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_content = current_chunk.strip()
            content_type = self.identify_content_type(chunk_content)
            
            chunk = TextChunk(
                content=chunk_content,
                chunk_id=f"page_{page_num}_sent_{chunk_index}",
                source_page=page_num,
                chunk_index=chunk_index,
                language=language,
                content_type=content_type,
                word_count=len(chunk_content.split()),
                char_count=len(chunk_content)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_long_text(self, text: str, page_num: int, base_index: int, 
                        language: str, content_type: str) -> List[TextChunk]:
        """
        Split long text into smaller chunks
        
        Args:
            text: Long text to split
            page_num: Source page number
            base_index: Base index for chunk IDs
            language: Detected language
            content_type: Content type
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        sentences = self.split_into_sentences(text, language)
        
        current_chunk = ""
        sub_index = 0
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"page_{page_num}_para_{base_index}_sub_{sub_index}",
                    source_page=page_num,
                    chunk_index=base_index * 100 + sub_index,  # Ensure unique indexing
                    language=language,
                    content_type=content_type,
                    word_count=len(current_chunk.split()),
                    char_count=len(current_chunk)
                )
                chunks.append(chunk)
                sub_index += 1
                
                # Start new chunk with some overlap
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_id=f"page_{page_num}_para_{base_index}_sub_{sub_index}",
                source_page=page_num,
                chunk_index=base_index * 100 + sub_index,
                language=language,
                content_type=content_type,
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_content_structure(self, text: str, page_num: int = 0) -> List[TextChunk]:
        """
        Chunk text based on HSC note structure (paragraph → table → MCQ → answers)
        
        Args:
            text: Text to chunk
            page_num: Source page number
            
        Returns:
            List of TextChunk objects
        """
        # Split into sections based on content structure
        sections = self.paragraph_pattern.split(text)
        chunks = []
        current_section = ""
        section_type = "paragraph"
        chunk_index = 0
        
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) < self.min_chunk_size:
                continue
            
            content_type = self.identify_content_type(section)
            language = self.detect_language(section)
            
            # Group related content types together
            if content_type in ['paragraph', 'definition']:
                # Start new section for paragraphs and definitions
                if current_section and section_type != content_type:
                    # Create chunk from previous section
                    chunk = self._create_chunk(current_section, page_num, chunk_index, section_type, language)
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_section = section
                    section_type = content_type
                else:
                    current_section = current_section + "\n\n" + section if current_section else section
                    section_type = content_type
            
            elif content_type in ['mcq', 'answer_table']:
                # Always create separate chunks for MCQs and answer tables
                if current_section:
                    chunk = self._create_chunk(current_section, page_num, chunk_index, section_type, language)
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_section = ""
                
                # Create chunk for MCQ/answer section
                chunk = self._create_chunk(section, page_num, chunk_index, content_type, language)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
            
            elif content_type == 'table':
                # Attach tables to the previous content (usually paragraph)
                if current_section and section_type in ['paragraph', 'definition']:
                    current_section = current_section + "\n\nTable:\n" + section
                else:
                    # Create standalone table chunk
                    chunk = self._create_chunk(section, page_num, chunk_index, content_type, language)
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
            
            else:
                # Handle other content types (headings, lists, etc.)
                chunk = self._create_chunk(section, page_num, chunk_index, content_type, language)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
        
        # Create final chunk if there's remaining content
        if current_section:
            chunk = self._create_chunk(current_section, page_num, chunk_index, section_type, 
                                     self.detect_language(current_section))
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, page_num: int, chunk_index: int, 
                     content_type: str, language: str) -> Optional[TextChunk]:
        """Helper method to create TextChunk objects"""
        if len(content.strip()) < self.min_chunk_size:
            return None
        
        # Split long content
        if len(content) > self.chunk_size:
            sub_chunks = self._split_long_text(content, page_num, chunk_index, language, content_type)
            return sub_chunks[0] if sub_chunks else None
        
        return TextChunk(
            content=content.strip(),
            chunk_id=f"page_{page_num}_{content_type}_{chunk_index}",
            source_page=page_num,
            chunk_index=chunk_index,
            language=language,
            content_type=content_type,
            word_count=len(content.split()),
            char_count=len(content)
        )

    def process_document(self, document_data: Dict[str, Any], 
                        chunking_strategy: str = "hsc_structure") -> List[TextChunk]:
        """
        Process complete document and create chunks - optimized for HSC notes
        
        Args:
            document_data: Document data from PDF extraction
            chunking_strategy: "paragraphs", "sentences", "mixed", or "hsc_structure"
            
        Returns:
            List of TextChunk objects
        """
        logger.info(f"Processing document with {chunking_strategy} chunking strategy")
        
        all_chunks = []
        
        if chunking_strategy == "hsc_structure":
            # Use HSC-optimized structure-based chunking
            for page_data in document_data['pages']:
                if page_data['text']:
                    cleaned_text = self.clean_text(page_data['text'])
                    page_chunks = self.chunk_by_content_structure(cleaned_text, page_data['page_number'])
                    all_chunks.extend(page_chunks)
        
        elif chunking_strategy == "paragraphs":
            # Process each page with paragraph chunking
            for page_data in document_data['pages']:
                if page_data['text']:
                    cleaned_text = self.clean_text(page_data['text'])
                    page_chunks = self.chunk_by_paragraphs(cleaned_text, page_data['page_number'])
                    all_chunks.extend(page_chunks)
        
        elif chunking_strategy == "sentences":
            # Process each page with sentence chunking
            for page_data in document_data['pages']:
                if page_data['text']:
                    cleaned_text = self.clean_text(page_data['text'])
                    page_chunks = self.chunk_by_sentences(cleaned_text, page_data['page_number'])
                    all_chunks.extend(page_chunks)
        
        else:  # mixed strategy (fallback)
            # Use different strategies based on content type
            for page_data in document_data['pages']:
                if page_data['text']:
                    cleaned_text = self.clean_text(page_data['text'])
                    
                    # Detect content characteristics
                    content_analysis = document_data.get('content_analysis', {})
                    
                    if content_analysis.get('has_mcq', False) or 'mcq' in cleaned_text.lower():
                        # Use paragraph chunking for MCQ content
                        page_chunks = self.chunk_by_paragraphs(cleaned_text, page_data['page_number'])
                    else:
                        # Use sentence chunking for regular content
                        page_chunks = self.chunk_by_sentences(cleaned_text, page_data['page_number'])
                    
                    all_chunks.extend(page_chunks)
        
        # Filter and deduplicate chunks
        final_chunks = self._filter_chunks(all_chunks)
        
        logger.info(f"Document processing completed: {len(final_chunks)} chunks created")
        if final_chunks:
            logger.info(f"Average chunk size: {sum(c.char_count for c in final_chunks) / len(final_chunks):.0f} characters")
            
            # Log content type distribution
            content_types = {}
            for chunk in final_chunks:
                content_types[chunk.content_type] = content_types.get(chunk.content_type, 0) + 1
            logger.info(f"Content type distribution: {content_types}")
        
        return final_chunks
    
    def _filter_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Filter and deduplicate chunks
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            Filtered list of chunks
        """
        filtered_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            # Skip very short chunks
            if chunk.char_count < self.min_chunk_size:
                continue
            
            # Basic deduplication
            content_hash = hash(chunk.content.strip().lower())
            if content_hash in seen_content:
                continue
            
            seen_content.add(content_hash)
            filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def export_chunks(self, chunks: List[TextChunk], output_path: str):
        """
        Export chunks to file for analysis
        
        Args:
            chunks: List of chunks to export
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Text Chunks Export\n\n")
            
            for chunk in chunks:
                f.write(f"## Chunk ID: {chunk.chunk_id}\n")
                f.write(f"- Page: {chunk.source_page}\n")
                f.write(f"- Language: {chunk.language}\n")
                f.write(f"- Content Type: {chunk.content_type}\n")
                f.write(f"- Words: {chunk.word_count}, Characters: {chunk.char_count}\n\n")
                f.write(f"{chunk.content}\n\n")
                f.write("---\n\n")
        
        logger.info(f"Chunks exported to {output_file}")

def preprocess_document(document_data: Dict[str, Any], 
                       chunk_size: int = 512,
                       chunk_overlap: int = 50,
                       chunking_strategy: str = "hsc_structure") -> List[TextChunk]:
    """
    Convenience function to preprocess document
    
    Args:
        document_data: Document data from PDF extraction
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        chunking_strategy: Chunking strategy to use
        
    Returns:
        List of processed text chunks
    """
    preprocessor = TextPreprocessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = preprocessor.process_document(document_data, chunking_strategy)
    
    return chunks

if __name__ == "__main__":
    # Example usage
    from extract_pdf import extract_pdf_content
    
    pdf_path = "../data/hsc26.pdf"
    
    if Path(pdf_path).exists():
        try:
            # Extract PDF content
            document_data = extract_pdf_content(pdf_path)
            
            # Preprocess and chunk
            chunks = preprocess_document(document_data, 
                                       chunk_size=512, 
                                       chunking_strategy="mixed")
            
            print(f"Created {len(chunks)} chunks")
            
            # Export for analysis
            preprocessor = TextPreprocessor()
            preprocessor.export_chunks(chunks, "../data/processed/chunks_analysis.md")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please place your HSC textbook PDF in the data/ directory") 