"""
Legal Document Preprocessing System for Occupational Safety Laws (Taiwan)
Handles hierarchical structure extraction, table/formula detection and conversion
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# PDF and OCR
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2

# OpenAI API
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LegalNode:
    """Represents a node in the legal document hierarchy"""
    law_name: str
    chapter: str  # 章
    section: str  # 節
    article: str  # 條
    paragraph: int
    content: str
    original_content: str
    chunk_id: str
    parent_id: str
    confidence_rate: Optional[int]
    tokens: int
    content_type: str = "text"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": {
                "law_name": self.law_name,
                "chapter": self.chapter,
                "section": self.section,
                "article": self.article,
                "paragraph": self.paragraph,
                "chunk_id": self.chunk_id,
                "parent_id": self.parent_id,
                "confidence_rate": self.confidence_rate
            },
            "tokens": self.tokens,
            "original_content": self.original_content
        }


class ChineseNumberConverter:
    """Converts Chinese numbers to Arabic numerals"""
    
    CHINESE_TO_ARABIC = {
        '○': '0', '零': '0',
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
        '壹': '1', '貳': '2', '參': '3', '肆': '4', '伍': '5',
        '陸': '6', '柒': '7', '捌': '8', '玖': '9', '拾': '10'
    }
    
    @staticmethod
    def convert_chinese_number(text: str) -> str:
        """Convert Chinese numbers to Arabic numerals"""
        if not text or text == "unknown":
            return text
            
        # Handle special decimal notation like "一二．五" -> "12.5"
        if '．' in text or '点' in text:
            text = text.replace('点', '.')
            parts = text.split('．')
            result = []
            for part in parts:
                converted = ChineseNumberConverter._convert_part(part)
                result.append(converted)
            return '.'.join(result)
        
        return ChineseNumberConverter._convert_part(text)
    
    @staticmethod
    def _convert_part(text: str) -> str:
        """Convert a part of Chinese number"""
        # Simple character-by-character conversion for concatenated numbers
        result = ""
        for char in text:
            if char in ChineseNumberConverter.CHINESE_TO_ARABIC:
                result += ChineseNumberConverter.CHINESE_TO_ARABIC[char]
            elif char.isdigit():
                result += char
            else:
                result += char
        
        # Handle traditional format like "十五" -> "15"
        if '十' in text or '拾' in text:
            try:
                # Complex Chinese number parsing logic
                result = ChineseNumberConverter._parse_traditional_chinese(text)
            except:
                pass
                
        return result
    
    @staticmethod
    def _parse_traditional_chinese(text: str) -> str:
        """Parse traditional Chinese numbers like 十五, 二十三, etc."""
        text = text.replace('拾', '十')
        
        if text == '十':
            return '10'
        elif text.startswith('十'):
            return '1' + ChineseNumberConverter.CHINESE_TO_ARABIC.get(text[1], text[1])
        elif '十' in text:
            parts = text.split('十')
            tens = ChineseNumberConverter.CHINESE_TO_ARABIC.get(parts[0], parts[0])
            ones = ChineseNumberConverter.CHINESE_TO_ARABIC.get(parts[1], '0') if len(parts) > 1 and parts[1] else '0'
            return str(int(tens) * 10 + int(ones))
        
        return text


class TableDetector:
    """Detects and extracts tables from text using box-drawing characters"""
    
    TABLE_CHARS = {'┌', '─', '┬', '│', '├', '┼', '┤', '└', '┘', '┴', '═', '║', '╔', '╗', '╚', '╝'}
    
    @staticmethod
    def detect_table(text: str) -> bool:
        """Check if text contains table structure"""
        table_char_count = sum(1 for char in text if char in TableDetector.TABLE_CHARS)
        lines = text.split('\n')
        
        # Consider it a table if:
        # 1. Has many table characters
        # 2. Multiple lines with consistent structure
        has_many_separators = table_char_count > 10
        has_structure = sum(1 for line in lines if '─' in line or '═' in line) >= 2
        
        return has_many_separators and has_structure
    
    @staticmethod
    def extract_table_content(text: str) -> str:
        """Extract table content, removing box-drawing characters"""
        lines = text.split('\n')
        content_lines = []
        
        for line in lines:
            # Remove lines that are purely structural
            if all(char in TableDetector.TABLE_CHARS or char.isspace() for char in line):
                continue
            
            # Clean the line
            cleaned = line
            for char in TableDetector.TABLE_CHARS:
                cleaned = cleaned.replace(char, ' ')
            
            cleaned = ' '.join(cleaned.split())
            if cleaned:
                content_lines.append(cleaned)
        
        return '\n'.join(content_lines)


class FormulaDetector:
    """Detects mathematical formulas in text"""
    
    @staticmethod
    def detect_formula(text: str) -> bool:
        """Check if text contains mathematical formulas"""
        # Look for mathematical operators and structure
        has_equals = '=' in text or '＝' in text
        has_operators = any(op in text for op in ['+', '−', '×', '÷', '/', '*', '・'])
        has_variables = bool(re.search(r'[A-Za-z][₀-₉⁰-⁹]*', text))
        has_fraction_structure = '──' in text or any(c in text for c in ['分之', '／'])
        
        # Check for subscripts/superscripts
        has_subscript = bool(re.search(r'[₀-₉]', text))
        has_superscript = bool(re.search(r'[⁰-⁹]', text))
        
        formula_indicators = sum([
            has_equals,
            has_operators,
            has_variables,
            has_fraction_structure,
            has_subscript,
            has_superscript
        ])
        
        return formula_indicators >= 2
    
    @staticmethod
    def extract_formula_components(text: str) -> Dict[str, Any]:
        """Extract components of a formula for better understanding"""
        components = {
            'raw_formula': text,
            'variables': re.findall(r'[A-Za-z]+[₀-₉⁰-⁹]*', text),
            'operators': [op for op in ['+', '−', '×', '÷', '=', '＝', '・'] if op in text],
            'has_fraction': '──' in text or '分之' in text
        }
        return components


class OpenAIConverter:
    """Uses OpenAI API to convert tables and formulas to natural language"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
    
    def convert_table_to_text(self, table_content: str, context: str = "") -> Tuple[str, int]:
        """Convert table to descriptive statements"""
        prompt = f"""你是一個專業的法律文件處理專家。請將以下表格內容轉換為清晰的文字敘述。

表格內容：
{table_content}

上下文資訊：{context if context else '無'}

要求：
1. 將表格的行列關係轉換為完整的句子
2. 保留所有重要資訊和數據
3. 使用正式的法律用語
4. 確保邏輯清晰，易於理解
5. 每個表格單元格的關係都要明確說明

請僅返回轉換後的文字敘述，不要包含其他說明。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一個專業的台灣職業安全法律文件處理專家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            converted_text = response.choices[0].message.content
            
            # Get confidence score
            confidence = self._estimate_confidence(response)
            
            return converted_text, confidence
            
        except Exception as e:
            logger.error(f"OpenAI API error in table conversion: {e}")
            return table_content, None
    
    def convert_formula_to_text(self, formula: str, context: str = "") -> Tuple[str, int]:
        """Convert formula to descriptive statements"""
        prompt = f"""你是一個專業的法律文件處理專家。請將以下數學公式轉換為清晰的文字敘述。

公式內容：
{formula}

上下文資訊：{context if context else '無'}

要求：
1. 解釋公式中每個變數的含義（如果能從上下文推斷）
2. 說明數學運算關係（乘法、除法、加法等）
3. 注意：某些情況下變數相鄰表示相乘（如 qCA 表示 q × C × A）
4. 處理分數、上標、下標等特殊符號
5. 處理中文數字表示法，例如：
   - "一二．五" 表示 12.5
   - "○．七五" 表示 0.75
   - "二二" 表示 22
6. 使用正式的法律/技術用語
7. 確保所有計算關係都清楚表達

請僅返回轉換後的文字敘述，不要包含其他說明。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一個專業的台灣職業安全法律文件處理專家，擅長數學公式解釋。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            converted_text = response.choices[0].message.content
            confidence = self._estimate_confidence(response)
            
            return converted_text, confidence
            
        except Exception as e:
            logger.error(f"OpenAI API error in formula conversion: {e}")
            return formula, None
    
    def _estimate_confidence(self, response) -> int:
        """Estimate confidence based on response characteristics"""
        # This is a heuristic - you might want to use a more sophisticated method
        # or add explicit confidence scoring in the prompt
        
        try:
            # Check if response has finish_reason of 'stop' (completed normally)
            if response.choices[0].finish_reason == 'stop':
                base_confidence = 85
            else:
                base_confidence = 60
            
            # Adjust based on response length (longer, more detailed = higher confidence)
            content_length = len(response.choices[0].message.content)
            if content_length > 200:
                base_confidence += 10
            elif content_length < 50:
                base_confidence -= 15
            
            return min(95, max(50, base_confidence))
            
        except:
            return 75  # Default confidence


class PDFProcessor:
    """Processes PDF files with OCR and structure extraction"""
    
    def __init__(self, openai_api_key: str):
        self.openai_converter = OpenAIConverter(openai_api_key)
        self.chinese_converter = ChineseNumberConverter()
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using both PyMuPDF and Tesseract OCR"""
        pages_data = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Try PyMuPDF text extraction first
                text = page.get_text()
                
                # If text extraction fails or yields little content, use OCR
                if len(text.strip()) < 50:
                    text = self._ocr_page(page)
                
                pages_data.append({
                    'page_num': page_num + 1,
                    'text': text,
                    'raw_text': text
                })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            
        return pages_data
    
    def _ocr_page(self, page) -> str:
        """Perform OCR on a PDF page"""
        try:
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to OpenCV format for preprocessing
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Perform OCR with Chinese support
            custom_config = r'--oem 3 --psm 6 -l chi_tra+eng'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            return text
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""


class HierarchicalParser:
    """Parses legal document hierarchy (章、節、條)"""
    
    # Patterns for legal structure
    CHAPTER_PATTERN = r'第\s*([一二三四五六七八九十百千萬壹貳參肆伍陸柒捌玖拾○零]+)\s*章'
    SECTION_PATTERN = r'第\s*([一二三四五六七八九十百千萬壹貳參肆伍陸柒捌玖拾○零]+)\s*節'
    ARTICLE_PATTERN = r'第\s*(\d+)\s*條'
    
    def __init__(self):
        self.converter = ChineseNumberConverter()
        
    def parse_structure(self, text: str) -> Dict[str, Any]:
        """Parse hierarchical structure from text"""
        structure = {
            'chapter': 'unknown',
            'section': 'unknown',
            'article': 'unknown'
        }
        
        # Extract chapter (章)
        chapter_match = re.search(self.CHAPTER_PATTERN, text)
        if chapter_match:
            chinese_num = chapter_match.group(1)
            structure['chapter'] = self.converter.convert_chinese_number(chinese_num)
        
        # Extract section (節)
        section_match = re.search(self.SECTION_PATTERN, text)
        if section_match:
            chinese_num = section_match.group(1)
            structure['section'] = self.converter.convert_chinese_number(chinese_num)
        
        # Extract article (條)
        article_match = re.search(self.ARTICLE_PATTERN, text)
        if article_match:
            structure['article'] = article_match.group(1)
        
        return structure
    
    def extract_articles(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual articles from text"""
        articles = []
        
        # Split by article markers
        article_splits = re.split(r'(第\s*\d+\s*條)', text)
        
        current_article = None
        for i, segment in enumerate(article_splits):
            article_match = re.match(r'第\s*(\d+)\s*條', segment)
            
            if article_match:
                current_article = article_match.group(1)
            elif current_article and segment.strip():
                articles.append({
                    'article': current_article,
                    'content': segment.strip()
                })
                current_article = None
        
        return articles


class ContentProcessor:
    """Processes content chunks, handling tables and formulas"""
    
    def __init__(self, openai_converter: OpenAIConverter):
        self.openai_converter = openai_converter
        self.table_detector = TableDetector()
        self.formula_detector = FormulaDetector()
        
    def process_content(self, content: str, context: str = "") -> Tuple[str, Optional[int]]:
        """Process content, converting tables and formulas as needed"""
        
        # Check for tables
        if self.table_detector.detect_table(content):
            logger.info("Table detected, converting to text...")
            table_content = self.table_detector.extract_table_content(content)
            converted, confidence = self.openai_converter.convert_table_to_text(
                table_content, context
            )
            return converted, confidence
        
        # Check for formulas
        if self.formula_detector.detect_formula(content):
            logger.info("Formula detected, converting to text...")
            converted, confidence = self.openai_converter.convert_formula_to_text(
                content, context
            )
            return converted, confidence
        
        # No conversion needed
        return content, None


class ChunkManager:
    """Manages content chunking with sentence-aware splitting"""
    
    # Chinese sentence endings
    SENTENCE_ENDINGS = {'。', '！', '？', '；', '：', '\n'}
    MAX_CHUNK_SIZE = 1500  # characters
    MIN_CHUNK_SIZE = 100
    
    @staticmethod
    def chunk_content(content: str) -> List[str]:
        """Split content into chunks, preserving sentence boundaries"""
        if len(content) <= ChunkManager.MAX_CHUNK_SIZE:
            return [content]
        
        chunks = []
        current_chunk = ""
        sentences = ChunkManager._split_sentences(content)
        
        for sentence in sentences:
            # If adding this sentence exceeds max size and we have content
            if len(current_chunk) + len(sentence) > ChunkManager.MAX_CHUNK_SIZE and current_chunk:
                if len(current_chunk) >= ChunkManager.MIN_CHUNK_SIZE:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Current chunk too small, add sentence anyway
                    current_chunk += sentence
            else:
                current_chunk += sentence
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in ChunkManager.SENTENCE_ENDINGS:
                sentences.append(current)
                current = ""
        
        if current:
            sentences.append(current)
        
        return sentences
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (rough approximation for Chinese)"""
        # For Chinese, roughly 1.5 characters per token
        # For mixed Chinese/English, use a blended estimate
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        return estimated_tokens


class LegalDocumentProcessor:
    """Main processor for legal documents"""
    
    def __init__(self, openai_api_key: str, output_dir: str = "./output"):
        self.pdf_processor = PDFProcessor(openai_api_key)
        self.hierarchical_parser = HierarchicalParser()
        self.content_processor = ContentProcessor(
            self.pdf_processor.openai_converter
        )
        self.chunk_manager = ChunkManager()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all PDF files in a folder"""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_nodes = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            nodes = self.process_pdf(str(pdf_file))
            all_nodes.extend(nodes)
            
            # Save individual file results
            self._save_json(nodes, self.output_dir / f"{pdf_file.stem}_processed.json")
        
        # Save combined results
        self._save_json(all_nodes, self.output_dir / "all_documents.json")
        
        logger.info(f"Processing complete. Total nodes: {len(all_nodes)}")
        return all_nodes
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a single PDF file"""
        law_name = Path(pdf_path).stem
        
        # Extract text from PDF
        pages_data = self.pdf_processor.extract_text_from_pdf(pdf_path)
        
        # Combine all pages
        full_text = "\n".join([page['text'] for page in pages_data])
        
        # Parse hierarchical structure
        global_structure = self.hierarchical_parser.parse_structure(full_text)
        
        # Extract articles
        articles = self.hierarchical_parser.extract_articles(full_text)
        
        # Process each article
        nodes = []
        current_chapter = global_structure['chapter']
        current_section = global_structure['section']
        
        for article_data in articles:
            article_num = article_data['article']
            content = article_data['content']
            
            # Update chapter/section if found in content
            local_structure = self.hierarchical_parser.parse_structure(content)
            if local_structure['chapter'] != 'unknown':
                current_chapter = local_structure['chapter']
            if local_structure['section'] != 'unknown':
                current_section = local_structure['section']
            
            # Process content (handle tables/formulas)
            processed_content, confidence = self.content_processor.process_content(
                content, context=f"法律：{law_name}，第{article_num}條"
            )
            
            # Chunk content
            chunks = self.chunk_manager.chunk_content(processed_content)
            
            # Create nodes for each chunk
            for idx, chunk in enumerate(chunks):
                node = LegalNode(
                    law_name=law_name,
                    chapter=current_chapter,
                    section=current_section,
                    article=article_num,
                    paragraph=idx + 1,
                    content=chunk,
                    original_content=content if idx == 0 else "",
                    chunk_id=f"{law_name}.pdf_{article_num}_{idx + 1}",
                    parent_id=f"{current_chapter}_{current_section}_{article_num}",
                    confidence_rate=confidence,
                    tokens=self.chunk_manager.estimate_tokens(chunk)
                )
                nodes.append(node.to_dict())
        
        return nodes
    
    def _save_json(self, data: List[Dict], output_path: Path):
        """Save data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to {output_path}")


# Main execution function
def main():
    """Main execution function"""
    
    # Configuration
    OPENAI_API_KEY = "YOUR=OPENAI_API_KEY"  # Replace with your key
    INPUT_FOLDER = "/mnt/d/__projects_main/dspproject_load/legalnorm_all"  # Folder containing PDF files
    OUTPUT_FOLDER = "./processed_output"
    
    # Initialize processor
    processor = LegalDocumentProcessor(
        openai_api_key=OPENAI_API_KEY,
        output_dir=OUTPUT_FOLDER
    )
    
    # Process all PDFs in folder
    results = processor.process_folder(INPUT_FOLDER)
    
    print(f"\nProcessing complete!")
    print(f"Total nodes generated: {len(results)}")
    print(f"Output saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()