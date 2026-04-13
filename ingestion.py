"""
Document ingestion: PDF extraction, table conversion, OCR cleaning, chunking.
"""

import re
import logging
import pdfplumber
from pathlib import Path
from operator import itemgetter
from typing import List, Tuple, Dict

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_PAGE_WORDS

logger = logging.getLogger(__name__)


def check_bboxes(word: dict, table_bbox: tuple) -> bool:
    """Check whether a word is inside a table bounding box."""
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]


def clean_duplicate_chars(text: str) -> str:
    """
    Remove repeated characters caused by overlapping text in PDFs.
    'CCuussttoommeerr' → 'Customer'
    """
    text = re.sub(r'(.)\1{2}', r'\1', text)
    text = re.sub(r'(.)\1', r'\1', text)
    return text


def table_to_natural_language(table_data: list) -> str:
    """Convert table list into readable sentences."""
    if not table_data or len(table_data) < 2:
        return ""
    headers = table_data[0]
    sentences = []
    for row in table_data[1:]:
        parts = []
        for header, value in zip(headers, row):
            if header and value:
                parts.append(f"{header} is {value}")
        if parts:
            sentences.append(". ".join(parts) + ".")
    return " ".join(sentences)


def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, str]]:
    """
    Extract text from a PDF file, handling both regular text and tables.
    
    Returns:
        List of [page_number, page_text] pairs
    """
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p, page in enumerate(pdf.pages):
                page_no = f"Page{p+1}"
                tables = page.find_tables()
                table_bboxes = [i.bbox for i in tables]
                tables = [{"table": i.extract(), 'top': i.bbox[1]} for i in tables]
                non_table_words = [
                    word for word in page.extract_words()
                    if not any(check_bboxes(word, tb) for tb in table_bboxes)
                ]
                lines = []
                for cluster in pdfplumber.utils.cluster_objects(
                    non_table_words + tables, itemgetter("top"), tolerance=5
                ):
                    if "text" in cluster[0]:
                        try:
                            lines.append(" ".join([i['text'] for i in cluster]))
                        except KeyError:
                            pass
                    elif 'table' in cluster[0]:
                        table_text = table_to_natural_language(cluster[0]['table'])
                        if table_text:
                            lines.append(table_text)

                cleaned_lines = [clean_duplicate_chars(line) for line in lines]
                full_text.append([page_no, " ".join(cleaned_lines)])
        logger.info(f"Extracted {len(full_text)} pages from {pdf_path}")
    except Exception as e:
        logger.error(f"Error extracting PDF {pdf_path}: {e}")
        raise
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def process_pdfs(pdf_directory: str) -> Tuple[List[str], List[Dict]]:
    """
    Process all PDFs in a directory: extract, chunk, and prepare for embedding.
    
    Returns:
        documents_list: List of text chunks
        metadata_list: List of metadata dicts for each chunk
    """
    pdf_dir = Path(pdf_directory)
    documents_list = []
    metadata_list = []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return documents_list, metadata_list
    
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}")
        extracted = extract_text_from_pdf(str(pdf_path))
        
        for page_no, page_text in extracted:
            if len(page_text.split()) < MIN_PAGE_WORDS:
                continue
            chunks = chunk_text(page_text)
            for j, chunk in enumerate(chunks):
                documents_list.append(chunk)
                metadata_list.append({
                    "Policy_Name": pdf_path.stem,
                    "Page_No": page_no,
                    "Chunk": j
                })
        logger.info(f"Finished processing {pdf_path.name}")
    
    logger.info(f"Total: {len(pdf_files)} PDFs → {len(documents_list)} chunks")
    return documents_list, metadata_list
