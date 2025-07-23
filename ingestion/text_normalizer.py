"""
Bangla Text Normalization Utilities

This module provides utilities for fixing Unicode issues in Bangla text
extracted from PDFs and other sources.
"""

import re
import unicodedata
from typing import Optional

try:
    from ftfy import fix_text
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

from loguru import logger


def normalize_bangla_unicode(text: str) -> str:
    """
    Normalize Bangla text by fixing Unicode ordering and combining characters
    
    Args:
        text: Input text with potential Unicode issues
        
    Returns:
        Normalized text with proper Unicode composition
    """
    if not text:
        return ""
    
    # Step 1: Unicode NFC normalization (combines decomposed characters)
    text = unicodedata.normalize("NFC", text)
    
    # Step 2: Use ftfy if available to fix encoding issues
    if FTFY_AVAILABLE:
        try:
            text = fix_text(text)
        except Exception as e:
            logger.debug(f"ftfy normalization failed: {e}")
    
    return text


def clean_unicode_artifacts(text: str) -> str:
    """
    Remove unwanted Unicode characters and artifacts from text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove zero-width characters
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u200d', '')  # Zero-width joiner  
    text = text.replace('\ufeff', '')  # Byte order mark
    text = text.replace('\u061c', '')  # Arabic letter mark
    
    # Replace various Unicode spaces with regular space
    unicode_spaces = [
        '\u00a0',  # Non-breaking space
        '\u2000',  # En quad
        '\u2001',  # Em quad
        '\u2002',  # En space
        '\u2003',  # Em space
        '\u2004',  # Three-per-em space
        '\u2005',  # Four-per-em space
        '\u2006',  # Six-per-em space
        '\u2007',  # Figure space
        '\u2008',  # Punctuation space
        '\u2009',  # Thin space
        '\u200a',  # Hair space
        '\u202f',  # Narrow no-break space
        '\u205f',  # Medium mathematical space
        '\u3000',  # Ideographic space
    ]
    
    for space_char in unicode_spaces:
        text = text.replace(space_char, ' ')
    
    return text


def fix_bangla_character_separation(text: str) -> str:
    """
    Fix common Bangla character separation issues from PDF extraction
    
    Args:
        text: Input text with potential character separation issues
        
    Returns:
        Text with fixed character combinations
    """
    if not text:
        return ""
    
    # Fix separated hasanta (্) - should be attached to preceding consonant
    text = re.sub(r'্(\s+)([ক-হড়ঢ়য়ৎ])', r'্\2', text)
    
    # Fix separated vowel marks that should be attached to consonants
    vowel_marks = '[িীুূৃেৈোৌ]'
    text = re.sub(r'([ক-হড়ঢ়য়ৎ])(\s+)(' + vowel_marks + ')', r'\1\3', text)
    
    # Fix separated anusvara (ং) and visarga (ঃ)
    text = re.sub(r'([ক-হড়ঢ়য়ৎ' + vowel_marks + '])(\s+)([ংঃ])', r'\1\3', text)
    
    # Fix separated chandrabindu (ঁ)
    text = re.sub(r'([ক-হড়ঢ়য়ৎ' + vowel_marks + '])(\s+)(ঁ)', r'\1\3', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Multiple newlines to double newline
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    
    # Tabs to spaces
    text = re.sub(r'\t', ' ', text)
    
    # Remove leading and trailing spaces on lines
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r' +\n', '\n', text)
    
    # Remove multiple non-newline whitespace
    text = re.sub(r'[^\S\n]{2,}', ' ', text)
    
    return text.strip()


def fix_digit_language_separation(text: str) -> str:
    """
    Fix spacing issues between Bangla digits and English text
    
    Args:
        text: Input text
        
    Returns:
        Text with proper spacing between different scripts
    """
    if not text:
        return ""
    
    # Add space between Bangla digits and English letters
    text = re.sub(r'([০-৯])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([০-৯])', r'\1 \2', text)
    
    # Add space between English digits and Bangla letters  
    text = re.sub(r'([0-9])([ক-হড়ঢ়য়ৎ])', r'\1 \2', text)
    text = re.sub(r'([ক-হড়ঢ়য়ৎ])([0-9])', r'\1 \2', text)
    
    return text


def comprehensive_bangla_normalize(text: str) -> str:
    """
    Apply comprehensive normalization to Bangla text
    
    Args:
        text: Raw text with potential issues
        
    Returns:
        Fully normalized text
    """
    if not text:
        return ""
    
    # Apply all normalization steps in order
    text = normalize_bangla_unicode(text)
    text = clean_unicode_artifacts(text)
    text = fix_bangla_character_separation(text)
    text = fix_digit_language_separation(text)
    text = normalize_whitespace(text)
    
    return text


def analyze_text_issues(text: str) -> dict:
    """
    Analyze text for common Unicode and formatting issues
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    issues = {
        'has_zero_width_chars': bool(re.search(r'[\u200c\u200d\ufeff]', text)),
        'has_unicode_spaces': bool(re.search(r'[\u00a0\u2000-\u200a\u202f\u205f\u3000]', text)),
        'has_separated_diacritics': bool(re.search(r'্\s+[ক-হড়ঢ়য়ৎ]', text)),
        'has_separated_vowels': bool(re.search(r'[ক-হড়ঢ়য়ৎ]\s+[িীুূৃেৈোৌ]', text)),
        'has_mixed_digits': bool(re.search(r'[০-৯][a-zA-Z]|[a-zA-Z][০-৯]', text)),
        'excessive_whitespace': bool(re.search(r'\s{3,}', text)),
        'character_count': len(text),
        'normalized_count': len(normalize_bangla_unicode(text))
    }
    
    return issues


if __name__ == "__main__":
    # Test the normalization functions
    test_texts = [
        "র্নম্নর্ব্িব্যজক্তি",  # Garbled text example
        "আি আমাি ব্ স সাতাি মাত্র",  # Age text with issues
        "প্রশ্ন ১   :   উত্তর   (ক)   ",  # Whitespace issues
        "২০১৯-২০ সা ল",  # Mixed script issues
    ]
    
    print("🔧 Testing Bangla text normalization:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Original: '{text}'")
        
        # Analyze issues
        issues = analyze_text_issues(text)
        print(f"   Issues: {[k for k, v in issues.items() if v and k != 'character_count' and k != 'normalized_count']}")
        
        # Apply normalization
        normalized = comprehensive_bangla_normalize(text)
        print(f"   Normalized: '{normalized}'")
        
        if text != normalized:
            print(f"   ✅ Fixed!")
        else:
            print(f"   ℹ️  No changes needed")
