# Legal Reasoning Project, NCCU (2025)
# formatting.py: Format structured legal nodes into bridgeable records.

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
except Exception:
    pd = None

class Formatter:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.stats: Dict[str, int] = {
            "total": 0,
            "with_paragraph": 0,
            "processed_successfully": 0,
        }

    def log(self, message: str):
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def normalize_article_number(self, article: Any) -> str:
        a = str(article).strip()
        if '-' in a:
            a = a.replace('-', '之')
        if '_' in a:
            a = a.replace('_', '之')
        return a

    def format_paragraph(self, paragraph: Any) -> Tuple[str, Optional[int]]:
        if paragraph is None:
            return "", None
        try:
            pnum = int(paragraph)
            if pnum > 0:
                return f"第{pnum}項", pnum
        except Exception:
            pass
        return "", None

    def generate_node_id(self, law_name: str, article: str, paragraph: Optional[int] = None) -> str:
        safe_law = re.sub(r"[^\w\u4e00-\u9fff]", '', law_name)
        safe_article = re.sub(r"[^\w\u4e00-\u9fff]", '', article)
        node_id = f"law_{safe_law}_art_{safe_article}"
        if paragraph:
            node_id += f"_para_{paragraph}"
        return node_id

    def generate_match_keys(self, law_name: str, article: str, para_str: str) -> List[str]:
        """Generate a deterministic, deduplicated list of alias match keys."""
        candidates: List[str] = []
        full_v1 = f"{law_name} 第{article}條{para_str}".strip()
        candidates.append(full_v1)
        full_v2 = f"{law_name} 第{article}條"
        candidates.append(full_v2)
        candidates.append(full_v1.replace(' ', ''))
        candidates.append(full_v2.replace(' ', ''))

        # Normalize and deduplicate while preserving order
        seen = set()
        aliases: List[str] = []
        for cand in candidates:
            if not cand:
                continue
            norm = re.sub(r"\s+", ' ', cand).strip()
            if norm not in seen:
                seen.add(norm)
                aliases.append(norm)

        return aliases

    def generate_rag_text(self, law_name: str, article: str, para_str: str, content: str, metadata: Dict[str, Any]) -> str:
        parts = [f"【法規】{law_name} 第{article}條"]
        if para_str:
            parts[0] += f" {para_str}"
        if metadata.get('chapter'):
            parts.append(f"【章】第{metadata['chapter']}章")
        if metadata.get('section'):
            parts.append(f"【節】第{metadata['section']}節")
        parts.append(f"【內容】{content}")
        return "\n".join(parts)

    def process_single(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            meta = entry.get('metadata', {}) or {}
            content = entry.get('content', '') or ''
            law_name = meta.get('law_name', '').replace('.pdf', '').strip()
            article = str(meta.get('article', '')).strip()
            paragraph = meta.get('paragraph')

            if not law_name or not article:
                self.log(f"跳過：缺少 law_name 或 article -> {meta.get('chunk_id')}")
                return None

            self.stats['total'] += 1

            article_norm = self.normalize_article_number(article)
            para_str, para_num = self.format_paragraph(paragraph)
            if para_num:
                self.stats['with_paragraph'] += 1

            node_id = self.generate_node_id(law_name, article_norm, para_num)
            match_keys = self.generate_match_keys(law_name, article_norm, para_str)
            rag_text = self.generate_rag_text(law_name, article_norm, para_str, content, meta)

            processed = {
                'node_id': node_id,
                'law_name': law_name,
                'article': article_norm,
                'paragraph': para_num,
                'chapter': meta.get('chapter'),
                'section': meta.get('section'),
                'content': content,
                'rag_text': rag_text,
                'match_keys': match_keys,
                'original_metadata': {
                    'chunk_id': meta.get('chunk_id'),
                    'parent_id': meta.get('parent_id'),
                    'confidence_rate': meta.get('confidence_rate'),
                    'tokens': entry.get('tokens')
                }
            }

            self.stats['processed_successfully'] += 1
            return processed

        except Exception as e:
            self.log(f"處理單筆錯誤: {e}")
            return None

    def process_file(self, input_path: str, output_path: str, output_format: str = 'json') -> List[Dict[str, Any]]:
        self.log(f"開始處理: {input_path}")
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # 支援 JSON array 或 JSONL file
        data: List[Dict[str, Any]] = []
        try:
            with open(p, 'r', encoding='utf-8') as f:
                # 試著一次讀入整個 JSON（常見情境）
                data = json.load(f)
                if isinstance(data, dict):
                    # 有些情況檔案是以物件包裹陣列
                    # 嘗試尋找常見鍵
                    for k in ('data', 'items', 'records'):
                        if k in data and isinstance(data[k], list):
                            data = data[k]
                            break
        except json.JSONDecodeError:
            # 退化為 JSONL（每行一個 JSON 物件）
            self.log("檔案非純 JSON 陣列，嘗試以 JSONL 逐行解析...")
            data = []
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                    except Exception:
                        # 忽略不合法行
                        continue

        self.log(f"讀取到 {len(data)} 筆節點資料")

        processed: List[Dict[str, Any]] = []
        for i, entry in enumerate(data, 1):
            if i % 100 == 0:
                self.log(f"處理進度: {i}/{len(data)}")
            r = self.process_single(entry)
            if r:
                processed.append(r)

        # Save
        self.save_output(processed, output_path, output_format)
        self.print_stats()
        return processed

    def save_output(self, records: List[Dict[str, Any]], output_path: str, output_format: str = 'json'):
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)

        if not records:
            # Write empty file and return
            if output_format == 'json':
                with open(outp, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            else:
                open(outp, 'w', encoding='utf-8').close()
            self.log(f"無紀錄可儲存，已建立空檔：{outp}")
            sample_path = outp.parent / f"{outp.stem}_sample.txt"
            self.save_sample(records, sample_path)
            return

        if output_format == 'json':
            with open(outp, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            self.log(f"已儲存 JSON: {outp}")

        elif output_format == 'jsonl':
            with open(outp, 'w', encoding='utf-8') as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self.log(f"已儲存 JSONL: {outp}")

        elif output_format == 'csv':
            if pd is None:
                # fallback: write simple CSV
                import csv
                keys = list(records[0].keys())
                with open(outp, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for rec in records:
                        row = rec.copy()
                        if isinstance(row.get('match_keys'), list):
                            row['match_keys'] = '|'.join(row['match_keys'])
                        writer.writerow(row)
                self.log(f"已儲存 CSV (fallback): {outp}")
            else:
                df = pd.DataFrame(records)
                if 'match_keys' in df.columns:
                    df['match_keys'] = df['match_keys'].apply(lambda x: '|'.join(x) if isinstance(x, list) else '')
                df.to_csv(outp, index=False, encoding='utf-8-sig')
                self.log(f"已儲存 CSV: {outp}")

        # sample view
        sample_path = outp.parent / f"{outp.stem}_sample.txt"
        self.save_sample(records, sample_path)

    def save_sample(self, records: List[Dict[str, Any]], sample_path: Path):
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('法律文件格式化範例輸出\n')
            f.write('=' * 80 + '\n\n')

            if not records:
                f.write('無可顯示的紀錄。\n')
                self.log(f"已儲存範例檔（空）: {sample_path}")
                return

            for idx, rec in enumerate(records[:3]):
                f.write(f"\n【範例 {idx+1}】\n")
                f.write(f"Node ID: {rec.get('node_id')}\n")
                f.write(f"法規名稱: {rec.get('law_name')}\n")
                art = rec.get('article')
                f.write(f"條號: 第{art}條")
                if rec.get('paragraph'):
                    f.write(f" 第{rec.get('paragraph')}項")
                f.write('\n\n')
                f.write('匹配鍵 (Match Keys):\n')
                for k in rec.get('match_keys', []):
                    f.write(f"  - {k}\n")
                f.write('\nRAG文本:\n')
                f.write(rec.get('rag_text', '') + '\n')
                f.write('-' * 80 + '\n')

        self.log(f"已儲存範例檔: {sample_path}")

    def print_stats(self):
        self.log('\n' + '=' * 60)
        self.log('處理統計')
        self.log('=' * 60)
        self.log(f"總節點數: {self.stats['total']}")
        self.log(f"成功處理: {self.stats['processed_successfully']}")
        self.log(f"包含項次: {self.stats['with_paragraph']}")
        if self.stats['total'] > 0:
            rate = self.stats['processed_successfully'] / self.stats['total'] * 100
            self.log(f"成功率: {rate:.1f}%")
        self.log('=' * 60 + '\n')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='格式化 all_documents.json 為橋接格式')
    parser.add_argument('--input', '-i', default='all_documents.json', help='輸入 JSON 檔案 (預設: all_documents.json)')
    parser.add_argument('--output', '-o', default='legal_content.json', help='輸出檔案路徑 (預設: _legal_content.json)')
    parser.add_argument('--format', '-f', default='json', choices=['json', 'jsonl', 'csv'], help='輸出格式')
    parser.add_argument('--sample-count', type=int, default=3, help='儲存範例數量（預設 3）')
    args = parser.parse_args()

    fmt = Formatter(verbose=True)
    processed = fmt.process_file(args.input, args.output, args.format)
    print(f"\n完成: {len(processed)} 筆處理結果，輸出: {args.output}")

if __name__ == '__main__':
    main()