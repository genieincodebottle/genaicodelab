
import re
import time
import json
from functools import wraps
from typing import Optional, Any, Dict
import streamlit as st

class XMLExtractionError(Exception):
    """Custom exception for XML extraction errors"""
    pass

def extract_xml_content(text: str, tag: str) -> Optional[str]:
    """Enhanced XML content extraction with better error handling."""
    if not text:
        raise XMLExtractionError("Empty response received")
        
    # Clean up whitespace and newlines
    text = text.strip()
    
    # Define start and end tags
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    try:
        # Find the content between tags
        start_pos = text.find(start_tag)
        if start_pos == -1:
            raise XMLExtractionError(f"Start tag <{tag}> not found in response")
            
        start_pos += len(start_tag)
        end_pos = text.find(end_tag)
        
        if end_pos == -1:
            raise XMLExtractionError(f"End tag </{tag}> not found in response")
            
        content = text[start_pos:end_pos].strip()
        
        if not content:
            raise XMLExtractionError(f"Empty content found between {tag} tags")
            
        return content
        
    except Exception as e:
        if isinstance(e, XMLExtractionError):
            raise
        raise XMLExtractionError(f"Error extracting XML content: {str(e)}")

def fix_json_string(json_str: str) -> str:
    """Enhanced JSON string fixing with more robust handling."""
    if not json_str:
        return "{}"
        
    # Remove any leading/trailing whitespace and newlines
    json_str = json_str.strip()
    
    # Remove any XML tags that might have slipped through
    json_str = re.sub(r'<[^>]+>', '', json_str)
    
    # Fix common JSON formatting issues
    json_str = re.sub(r'(?<![\{\[,])\s*"(?![}\]])', ',"', json_str)  # Add missing commas
    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)  # Quote property names
    json_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)', r': "\1"', json_str)  # Quote string values
    
    # Ensure proper structure
    if not json_str.startswith('{') and not json_str.startswith('['):
        json_str = '{' + json_str
    if not json_str.endswith('}') and not json_str.endswith(']'):
        json_str = json_str + '}'
        
    return json_str

def safe_json_loads(json_str: str, default_value: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced safe JSON parsing with detailed error handling."""
    if default_value is None:
        default_value = {}
        
    if not json_str:
        return default_value
        
    try:
        # First try direct parsing
        return json.loads(json_str)
    except json.JSONDecodeError as e1:
        try:
            # Try fixing common issues
            fixed_json = fix_json_string(json_str)
            return json.loads(fixed_json)
        except json.JSONDecodeError as e2:
            st.error(f"Failed to parse JSON even after fixes.")
            st.error(f"Original error: {str(e1)}")
            st.error(f"After fix error: {str(e2)}")
            st.code(f"Original JSON:\n{json_str}\n\nFixed JSON:\n{fixed_json}", language="json")
            return default_value

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1,
    exponential_base: float = 2,
    show_progress: bool = True
):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if show_progress:
                        st.warning(f"Attempt {retry + 1}/{max_retries} failed: {str(e)}")
                        st.warning(f"Retrying in {delay:.1f} seconds...")
                    
                    time.sleep(delay)
                    delay *= exponential_base
            
            if show_progress:
                st.error(f"All {max_retries} attempts failed.")
            raise last_exception
            
        return wrapper
    return decorator

@retry_with_exponential_backoff(max_retries=3, initial_delay=1, exponential_base=2)
def safe_extract_and_parse_json(
    response: str,
    xml_tag: str,
    default_value: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Combined safe XML extraction and JSON parsing with retries."""
    try:
        # Extract XML content
        content = extract_xml_content(response, xml_tag)
        
        # Parse JSON with safety measures
        result = safe_json_loads(content, default_value)
        
        return result
    except XMLExtractionError as e:
        st.warning(f"XML extraction failed: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Unexpected error in parsing: {str(e)}")
        raise