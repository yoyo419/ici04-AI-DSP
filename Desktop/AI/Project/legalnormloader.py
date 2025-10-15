import os
import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import PyPDF2
from openai import OpenAI
import tiktoken
from pathlib import Path
# pip install PyPDF2 openai tiktoken

# note: 
    # 包含全部pdf檔的處理一次約要700000tokens，不知道老師是用哪個模型不過算高一點可能要400台幣
    # 如果要測試可以把不要一次處理全部的檔案，或是減少使用token，ctrl+f "user-changable" 應該可以找到可以調整的地方

@dataclass
class ChunkMetadata:
    """Metadata for each content chunk"""
    chapter: str
    section: str
    article: str
    paragraph: int
    chunk_id: str
    parent_id: str
    level: int
    content_type: str  # 'text', 'table', 'formula'
    
@dataclass
class ProcessedChunk:
    """Processed content chunk with metadata"""
    content: str
    metadata: ChunkMetadata
    original_content: str  # original text before processing
    tokens: int

class LegalDocumentPreprocessor:
    """
    Preprocessing system for occupational safety legal norms
    Handles PDF loading, hierarchical chunking, table/formula detection and transformation
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Hierarchy patterns for legal documents
        self.hierarchy_patterns = {
            'chapter': r'第\s*([一二三四五六七八九十百千萬]+)\s*章\s+(.+)',
            'section': r'第\s*([一二三四五六七八九十百千萬]+)\s*節\s+(.+)',
            'article': r'第\s*(\d+)\s*條',
            'paragraph': r'^(\d+)\s+',
        }
        
        # Table detection pattern
        self.table_pattern = r'┌[\s\S]*?└'
        
        # Formula patterns
        self.formula_patterns = [
            r'[A-Za-zα-ωΑ-Ω]+\s*[＝=]\s*[^。\n]+',
            r'式中.*?[：:].+',
        ]
        
        # Mandarin number conversion
        self.mandarin_numbers = {
            '○': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10', '百': '100', '千': '1000', '萬': '10000'
        }
        
    def load_pdfs_from_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        # Load all PDFs from a folder and extract text
        documents = []
        folder = Path(folder_path)
        
        for pdf_file in folder.glob('*.pdf'):
            try:
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    
                    documents.append({
                        'filename': pdf_file.name,
                        'text': text,
                        'num_pages': len(pdf_reader.pages)
                    })
                    print(f"✓ Loaded: {pdf_file.name} ({len(pdf_reader.pages)} pages)")
            except Exception as e:
                print(f"✗ Error loading {pdf_file.name}: {e}")
        
        return documents
    
    def convert_mandarin_numbers(self, text: str) -> str:
        # Convert Mandarin number representations to Arabic numerals
        pattern = r'([○一二三四五六七八九十]+)[．\.]([○一二三四五六七八九十]+)'
        
        def replace_number(match):
            integer_part = match.group(1)
            decimal_part = match.group(2)
            
            # Convert integer part
            integer_str = ''.join(self.mandarin_numbers.get(c, c) for c in integer_part)
            # Convert decimal part
            decimal_str = ''.join(self.mandarin_numbers.get(c, c) for c in decimal_part)
            
            return f"{integer_str}.{decimal_str}"
        
        text = re.sub(pattern, replace_number, text)
        
        # Convert standalone Mandarin numbers
        for mandarin, arabic in self.mandarin_numbers.items():
            text = text.replace(mandarin, arabic)
        
        return text
    
    def detect_hierarchy(self, text: str) -> Dict[str, Any]:
        # Detect and return hierarchical structure in legal document
        # This can be improved with more detailed patterns, for example, adding "節"
        hierarchy = {
            'chapter': None,
            'chapter_title': None,
            'section': None,
            'section_title': None,
            'article': None,
            'paragraph': None
        }
        
        # Check for chapter
        chapter_match = re.search(self.hierarchy_patterns['chapter'], text)
        if chapter_match:
            hierarchy['chapter'] = chapter_match.group(1)
            hierarchy['chapter_title'] = chapter_match.group(2).strip()
        
        # Check for section
        section_match = re.search(self.hierarchy_patterns['section'], text)
        if section_match:
            hierarchy['section'] = section_match.group(1)
            hierarchy['section_title'] = section_match.group(2).strip()
        
        # Check for article
        article_match = re.search(self.hierarchy_patterns['article'], text)
        if article_match:
            hierarchy['article'] = article_match.group(1)
        
        # Check for paragraph
        paragraph_match = re.search(self.hierarchy_patterns['paragraph'], text)
        if paragraph_match:
            hierarchy['paragraph'] = paragraph_match.group(1)
        
        return hierarchy
    
    def detect_tables(self, text: str) -> List[Tuple[str, int, int]]:
        # Detect tables constructed with box-drawing characters. Returns list of (table_text, start_pos, end_pos).
        # Recall: self.table_pattern = r'┌[\s\S]*?└'
        # This can be improved by considering more complex table structures
        tables = []
        for match in re.finditer(self.table_pattern, text, re.DOTALL):
            tables.append((match.group(0), match.start(), match.end()))
        return tables
    
    def detect_formulas(self, text: str) -> List[Tuple[str, int, int]]:
        # Detect mathematical formulas in text. Returns list of (formula_text, start_pos, end_pos).
        # Recall:　self.formula_patterns = [r'[A-Za-zα-ωΑ-Ω]+\s*[＝=]\s*[^。\n]+',　r'式中.*?[：:].+',]
        # This can be improved by considering more complex formula structures
        formulas = []
        for pattern in self.formula_patterns:
            for match in re.finditer(pattern, text):
                formulas.append((match.group(0), match.start(), match.end()))
        return formulas
    
    def table_to_statements(self, table_text: str) -> str:
        # Use OpenAI API to convert table to natural language statements
        prompt = f"""你是一位專精於職業安全法規的專家。請將以下表格轉換為清晰、詳細的陳述句，保留所有重要信息和數值關係。

表格內容：
{table_text}

要求：
1. 將表格中的每一行轉換為完整的陳述句
2. 清楚描述各欄位之間的關係
3. 保留所有數值和單位
4. 使用自然、流暢的中文表達
5. 確保專業術語的準確性

請提供轉換後的陳述句："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是職業安全衛生法規專家，擅長將複雜表格轉換為清晰的文字描述。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000 ######## user-changable (especially for testing) ########
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in table conversion: {e}")
            return f"[TABLE_CONVERSION_ERROR] {table_text[:100]}..."
    
    def formula_to_statements(self, formula_text: str, context: str = "") -> str:
        """
        Use OpenAI API to convert formula to natural language statements
        """
        prompt = f"""你是一位專精於職業安全法規的專家。請將以下數學公式轉換為清晰、詳細的文字描述。

公式內容：
{formula_text}

上下文：
{context[:500]}

要求：
1. 詳細解釋公式中每個變數的意義
2. 說明各變數之間的數學關係（注意：相鄰字母可能表示相乘）
3. 解釋公式的物理或實務意義
4. 包含變數的單位信息
5. 使用自然、流暢的中文表達
6. 如果公式中有分數、上標、下標，請清楚說明

範例：
輸入：W ＝qCA
輸出：風荷重W等於速度壓q、風力係數C與受風面積A的乘積。其中W代表風荷重，單位為牛頓；q代表速度壓，單位為牛頓每平方公尺；C代表風力係數；A代表受風面積，單位為平方公尺。

請提供轉換後的描述："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是職業安全衛生法規專家，擅長解釋數學公式和技術規範。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500 ######## user-changable (especially for testing) ########
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in formula conversion: {e}")
            return f"[FORMULA_CONVERSION_ERROR] {formula_text[:100]}..."
    
    def intelligent_chunk(self, text: str, max_tokens: int = 400) -> List[str]:
        # Intelligently chunk text while preserving sentence boundaries

        # Split by sentences (Chinese period, exclamation, question marks)
        sentences = re.split(r'([。！？\n])', text)
        
        # Reconstruct sentences with their delimiters
        sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
        if len(sentences) % 2 == 1:
            sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = len(self.encoding.encode(sentence))
            
            # If adding this sentence exceeds limit and we have content, save chunk
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_document(self, document: Dict[str, Any], max_chunk_tokens: int = 400) -> List[ProcessedChunk]:
        # Main processing pipeline for a single document
        text = document['text']
        filename = document['filename']
        
        # Step 1: Convert Mandarin numbers
        text = self.convert_mandarin_numbers(text)
        
        # Step 2: Detect and process tables
        tables = self.detect_tables(text)
        table_replacements = {}
        for table_text, start, end in tables:
            print(f"Processing table at position {start}...")
            statements = self.table_to_statements(table_text)
            table_replacements[(start, end)] = statements
        
        # Step 3: Detect and process formulas
        formulas = self.detect_formulas(text)
        formula_replacements = {}
        for formula_text, start, end in formulas:
            # Get context around formula (±200 chars)
            context_start = max(0, start - 200)
            context_end = min(len(text), end + 200)
            context = text[context_start:context_end]
            
            print(f"Processing formula at position {start}...")
            statements = self.formula_to_statements(formula_text, context)
            formula_replacements[(start, end)] = statements
        
        # Step 4: Replace tables and formulas with statements
        all_replacements = sorted(
            list(table_replacements.items()) + list(formula_replacements.items()),
            key=lambda x: x[0][0],
            reverse=True
        )
        
        processed_text = text
        for (start, end), replacement in all_replacements:
            processed_text = processed_text[:start] + replacement + processed_text[end:]
        
        # Step 5: Split into sections based on hierarchy
        lines = processed_text.split('\n')
        
        current_hierarchy = {
            'chapter': None,
            'chapter_title': None,
            'section': None,
            'section_title': None,
            'article': None
        }
        
        chunks = []
        current_content = ""
        chunk_counter = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Update hierarchy
            hierarchy = self.detect_hierarchy(line)
            if hierarchy['chapter']:
                current_hierarchy['chapter'] = hierarchy['chapter']
                current_hierarchy['chapter_title'] = hierarchy['chapter_title']
            if hierarchy['section']:
                current_hierarchy['section'] = hierarchy['section']
                current_hierarchy['section_title'] = hierarchy['section_title']
            if hierarchy['article']:
                # When we hit a new article, chunk previous content
                if current_content:
                    sub_chunks = self.intelligent_chunk(current_content, max_chunk_tokens)
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunk_id = f"{filename}_{chunk_counter}"
                        parent_id = f"{current_hierarchy['chapter']}_{current_hierarchy['section']}_{current_hierarchy['article']}"
                        
                        metadata = ChunkMetadata(
                            chapter=current_hierarchy['chapter'] or 'unknown',
                            section=current_hierarchy['section'] or 'unknown',
                            article=current_hierarchy['article'] or 'unknown',
                            paragraph=i + 1,
                            chunk_id=chunk_id,
                            parent_id=parent_id,
                            level=self._calculate_level(current_hierarchy),
                            content_type='text'
                        )
                        
                        chunks.append(ProcessedChunk(
                            content=sub_chunk,
                            metadata=metadata,
                            original_content=current_content if i == 0 else "",
                            tokens=len(self.encoding.encode(sub_chunk))
                        ))
                        chunk_counter += 1
                    
                    current_content = ""
                
                current_hierarchy['article'] = hierarchy['article']
            
            current_content += line + "\n"
        
        # Process remaining content
        if current_content:
            sub_chunks = self.intelligent_chunk(current_content, max_chunk_tokens)
            for i, sub_chunk in enumerate(sub_chunks):
                chunk_id = f"{filename}_{chunk_counter}"
                parent_id = f"{current_hierarchy['chapter']}_{current_hierarchy['section']}_{current_hierarchy['article']}"
                
                metadata = ChunkMetadata(
                    chapter=current_hierarchy['chapter'] or 'unknown',
                    section=current_hierarchy['section'] or 'unknown',
                    article=current_hierarchy['article'] or 'unknown',
                    paragraph=i + 1,
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    level=self._calculate_level(current_hierarchy),
                    content_type='text'
                )
                
                chunks.append(ProcessedChunk(
                    content=sub_chunk,
                    metadata=metadata,
                    original_content=current_content if i == 0 else "",
                    tokens=len(self.encoding.encode(sub_chunk))
                ))
                chunk_counter += 1
        
        return chunks
    
    def _calculate_level(self, hierarchy: Dict[str, Any]) -> int:
        # Calculate hierarchical level (0=document, 1=chapter, 2=section, 3=article, 4=paragraph)"""
        if hierarchy['article']:
            return 3
        elif hierarchy['section']:
            return 2
        elif hierarchy['chapter']:
            return 1
        else:
            return 0
    
    def process_all_documents(self, folder_path: str, output_json: str = "preprocessed_data.json"):
        # Process all PDFs in folder and save to JSON
        print("=" * 80)
        print("LEGAL DOCUMENT PREPROCESSING SYSTEM")
        print("=" * 80)
        
        # Load documents
        print(f"\n📁 Loading PDFs from: {folder_path}")
        documents = self.load_pdfs_from_folder(folder_path)
        print(f"✓ Loaded {len(documents)} documents\n")
        
        # Process each document
        all_chunks = []
        for i, doc in enumerate(documents, 1):
            print(f"\n{'='*80}")
            print(f"Processing document {i}/{len(documents)}: {doc['filename']}")
            print(f"{'='*80}")
            
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
            
            print(f"✓ Generated {len(chunks)} chunks from {doc['filename']}")
        
        # Save to JSON
        print(f"\n💾 Saving to {output_json}...")
        output_data = {
            'metadata': {
                'total_documents': len(documents),
                'total_chunks': len(all_chunks),
                'total_tokens': sum(chunk.tokens for chunk in all_chunks),
                'documents': [doc['filename'] for doc in documents]
            },
            'chunks': [
                {
                    'content': chunk.content,
                    'metadata': asdict(chunk.metadata),
                    'tokens': chunk.tokens,
                    'original_content': chunk.original_content
                }
                for chunk in all_chunks
            ]
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Preprocessing complete!")
        print(f"\n📊 Summary:")
        print(f"   - Documents processed: {len(documents)}")
        print(f"   - Total chunks: {len(all_chunks)}")
        print(f"   - Total tokens: {sum(chunk.tokens for chunk in all_chunks):,}")
        print(f"   - Average tokens per chunk: {sum(chunk.tokens for chunk in all_chunks) // len(all_chunks)}")
        
        return output_data


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = LegalDocumentPreprocessor(
        openai_api_key="PASTE_YOUR_OPENAI_APIKEY_HERE",
        model="gpt-4"
    )
    
    # Process all documents
    output_data = preprocessor.process_all_documents(
        folder_path="/mnt/d/__projects_main/dspproject/legalnorm", ######## user-changable ########
        output_json="preprocessed_legal_norms.json"
    )
    
    # Optional: Inspect results
    print("\n" + "="*80)
    print("SAMPLE PROCESSED CHUNKS")
    print("="*80)
    for i, chunk_data in enumerate(output_data['chunks'][:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Hierarchy: Chapter {chunk_data['metadata']['chapter']} > "
              f"Section {chunk_data['metadata']['section']} > "
              f"Article {chunk_data['metadata']['article']}")
        print(f"Tokens: {chunk_data['tokens']}")
        print(f"Content preview: {chunk_data['content'][:200]}...")