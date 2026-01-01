import json
import re
import unicodedata
from collections import defaultdict
from difflib import get_close_matches

# ==========================================
# 1. 領域知識與別名地圖 (Domain Knowledge)
# ==========================================

LAW_NAME_MAP = {
    # 既有映射
    "勞工安全衛生法": "職業安全衛生法",
    "職業全衛生法": "職業安全衛生法",
    "勞工安全衛生設施規則": "職業安全衛生設施規則",
    "勞工安全衛生組織管理及自動檢查辦法": "職業安全衛生管理辦法",
    "勞工健康保護規則": "勞工健康保護規則",
    "勞工職業災害保險及保護法": "勞工職業災害保險及保護法",
    "營造安全衛生設施標準": "營造安全衛生設施標準",
    "職業安全衛生教育訓練規則": "職業安全衛生教育訓練規則",
    "就業服務法": "就業服務法",
    # [V9 新增] 錯字修正
    "職業安全衛生設置規則": "職業安全衛生設施規則",
    "屋內線路裝置規則": "屋內線路裝置規則", # 確保一致
    "電業法": "電業法",
}

BLACKLIST_KEYWORDS = [
    "製造業", "食品", "粉條", "麵條", "加工", 
    "事故", "死亡", "受傷", "罹災", "原因", "分析"
]

# ==========================================
# 2. 核心工具集
# ==========================================

def parse_chinese_number(cn_str):
    """中文數字轉阿拉伯數字 (保持 V8 的穩定邏輯)"""
    if not cn_str: return ""
    if cn_str.isdigit(): return cn_str
        
    cn_map = {'○': 0, '〇': 0, '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, 
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    
    if len(cn_str) == 1 and cn_str in cn_map:
        return str(cn_map[cn_str])

    total = 0
    tmp = 0
    for char in cn_str:
        if char in cn_map:
            tmp = cn_map[char]
        elif char == '十':
            if tmp == 0: tmp = 1
            total += tmp * 10
            tmp = 0
        elif char == '百':
            if tmp == 0: tmp = 1
            total += tmp * 100
            tmp = 0
    total += tmp
    return str(total)

def text_chinese_to_arabic(text):
    pattern = re.compile(r'[一二三四五六七八九十百○〇零]+')
    def replace_func(match):
        return parse_chinese_number(match.group(0))
    return pattern.sub(replace_func, text)

def advanced_normalize_v9(text):
    """
    標準化 v9：去除引號 + 補全後綴
    """
    if not text: return ""
    
    # 1. NFKC 正規化
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 轉小寫與去空白
    text = text.lower()
    text = re.sub(r'\s+', '', text)
    
    # 3. [V9] 去除開頭結尾的標點符號 (解決 「職業安全... 的問題)
    text = text.strip("「」『』\"' ")

    # 4. 移除代碼雜訊
    text = re.sub(r'\(\d+\)', '', text)

    # 5. 中文數字轉阿拉伯
    text = text_chinese_to_arabic(text)

    # 6. 法規名稱校正
    for alias, standard in LAW_NAME_MAP.items():
        if text.startswith(alias):
            text = text.replace(alias, standard, 1)
            break

    # 7. 處理「之」字號
    text = re.sub(r'條之(\d+)', r'-\1條', text) 
    text = re.sub(r'(\d+)之(\d+)', r'\1-\2', text)
    text = text.replace("_", "-")

    # 8. 補 "第" 字
    def add_prefix(match):
        char = match.group(1)
        num_part = match.group(2)
        if char == "第": return match.group(0)
        return f"{char}第{num_part}條"
    text = re.sub(r'([\u4e00-\u9fa5])([\d\-]+)條', add_prefix, text)

    # 9. 移除款目
    text = re.sub(r'第\d+款', '', text)
    text = re.sub(r'第\d+目', '', text)
    
    # 10. [V9] 處理不完整後綴 (e.g., "第59條第2") -> 視為 "第2項"
    # Regex: 結尾是 "第+數字"，且後面沒有任何單位
    text = re.sub(r'(第\d+)$', r'\1項', text)
    
    # 11. 移除括號
    text = re.sub(r'[\(（].*?[\)）]', '', text)

    return text

# ==========================================
# 3. 索引建構 (含 Missing Law 偵測準備)
# ==========================================

def extract_article_from_content(content):
    if not content: return None
    norm_content = advanced_normalize_v9(content[:20])
    match = re.search(r'第([\d\-]+)條', norm_content)
    if match: return match.group(1)
    return None

def build_diagnostic_index(legal_content_path):
    print("🏗️ 正在建構 v9 診斷型索引...")
    
    with open(legal_content_path, 'r', encoding='utf-8') as f:
        legal_data = json.load(f)

    index_full = {}    
    index_article = {} 
    existing_laws = set() # [V9] 用來記錄資料庫裡到底有哪些法
    
    article_aggregator = defaultdict(list)
    article_meta = {}

    for entry in legal_data:
        raw_law = str(entry.get('law_name', ''))
        
        # 正規化法規名稱並存入集合
        norm_law_name = advanced_normalize_v9(raw_law)
        # 移除可能的 "第x條" 後綴，只留法名
        norm_law_name = re.sub(r'第[\d\-]+條.*', '', norm_law_name)
        if norm_law_name:
            existing_laws.add(norm_law_name)

        raw_art = str(entry.get('article', '')) 
        raw_para = entry.get('paragraph', '')
        para_str = str(raw_para) if raw_para not in [0, "0", None, "None", ""] else ""

        # Content Sniffing
        sniffed_art = extract_article_from_content(entry.get('content', ''))
        
        target_arts = set()
        target_arts.add(raw_art)
        if sniffed_art and sniffed_art != raw_art:
            target_arts.add(sniffed_art)

        for art in target_arts:
            base_key = f"{raw_law}第{art}條"
            
            if para_str:
                full_key = f"{base_key}第{para_str}項"
                norm_full = advanced_normalize_v9(full_key)
                index_full[norm_full] = entry
                
                if para_str == "1":
                    norm_base = advanced_normalize_v9(base_key)
                    if norm_base not in index_full:
                        index_full[norm_base] = entry
            else:
                norm_base = advanced_normalize_v9(base_key)
                index_full[norm_base] = entry

            for k in entry.get('match_keys', []):
                index_full[advanced_normalize_v9(k)] = entry

            if raw_law and art:
                norm_art_key = advanced_normalize_v9(base_key)
                prefix = f"[第{para_str}項] " if para_str else ""
                article_aggregator[norm_art_key].append(prefix + entry.get('content', ''))
                
                if norm_art_key not in article_meta:
                    node = entry.copy()
                    node['paragraph'] = "AGGREGATED"
                    node['node_id'] = f"{norm_art_key}_AGGREGATED"
                    article_meta[norm_art_key] = node

    for key, contents in article_aggregator.items():
        node = article_meta[key]
        node['content'] = "\n".join(contents)
        node['is_aggregated'] = True
        index_article[key] = node

    print(f"📚 資料庫收錄法規數: {len(existing_laws)}")
    return index_full, index_article, existing_laws

# ==========================================
# 4. 主執行流程
# ==========================================

def execute_mapping_v9(kg_file, legal_content_file, output_file):
    index_full, index_article, existing_laws = build_diagnostic_index(legal_content_file)
    all_index_keys = list(index_full.keys()) + list(index_article.keys())
    
    with open(kg_file, 'r', encoding='utf-8') as f:
        kg = json.load(f)
        
    mapped_count = 0
    total_reg = 0
    
    # 錯誤分類統計
    missing_law_logs = defaultdict(int) # 法規不存在
    missing_article_logs = []           # 法規存在但條文對不上
    
    nodes = kg.get('nodes', [])
    
    print("🚀 開始 V9 Mapping (最終診斷版)...")

    for node in nodes:
        label = str(node.get('label', '')).strip()
        norm_label = advanced_normalize_v9(label)
        
        # Filter
        is_valid = True
        for bad in BLACKLIST_KEYWORDS:
            if bad in label: is_valid = False
        if "第" not in norm_label or "條" not in norm_label: is_valid = False
        if len(label) > 60: is_valid = False
        
        if not is_valid: continue

        total_reg += 1
        target_node = None
        match_method = "unknown"

        # 策略 1: Exact Match
        if norm_label in index_full:
            target_node = index_full[norm_label]
            match_method = "exact_v9"
            
        # 策略 2: Article Rollup
        if not target_node:
            match = re.match(r'(.*?第[\d\-]+條)', norm_label)
            if match:
                rollup_key = match.group(1)
                if rollup_key in index_article:
                    target_node = index_article[rollup_key]
                    match_method = "rollup_v9"
                    
        # 策略 3: Fuzzy Match
        if not target_node:
            law_match = re.match(r'(.*?)第', norm_label)
            if law_match:
                current_law = law_match.group(1)
                candidate_keys = [k for k in all_index_keys if k.startswith(current_law)]
                matches = get_close_matches(norm_label, candidate_keys, n=1, cutoff=0.85)
                if matches:
                    best_match = matches[0]
                    target_node = index_full.get(best_match) or index_article.get(best_match)
                    match_method = f"fuzzy_v9 ({best_match})"

        if target_node:
            node['legal_ref_id'] = target_node.get('node_id')
            node['full_text'] = target_node.get('content')
            node['mapping_method'] = match_method
            node['normalized_label'] = norm_label
            mapped_count += 1
        else:
            # --- V9 診斷邏輯 ---
            # 嘗試提取法規名稱
            match_law = re.match(r'(.*?)第', norm_label)
            if match_law:
                law_name = match_law.group(1)
                # 檢查該法規是否存在於 index 中
                # 我們用 fuzzy check 確保不是因為小錯字 (e.g. 職安法 vs 職業安全衛生法 已經在 map 處理過，這裡比對 normalized name)
                
                # 檢查 existing_laws 裡有沒有這個法
                # 這裡做一個簡單的 substring check 或 exact check
                is_law_exist = False
                for exist_law in existing_laws:
                    if law_name in exist_law or exist_law in law_name:
                        is_law_exist = True
                        break
                
                if not is_law_exist:
                    missing_law_logs[law_name] += 1
                else:
                    missing_article_logs.append(f"{label} (Law Found, Article Missing)")
            else:
                 missing_article_logs.append(f"{label} (Parse Error)")

    print(f"📊 V9 最終統計結果:")
    print(f"    - 有效法規節點: {total_reg}")
    print(f"    - 成功匹配: {mapped_count}")
    print(f"    - 成功率: {mapped_count/total_reg*100:.2f}%")
    
    print("\n🔍 未匹配原因診斷:")
    if missing_law_logs:
        print(f"    🔴 [嚴重] 資料庫完全缺失以下法規 (請補充 legal_content.json):")
        for law, count in missing_law_logs.items():
            print(f"       - {law}: {count} 個節點受影響")
            
    if missing_article_logs:
        print(f"    🟠 [警告] 法規存在但條文匹配失敗 (前 15 個):")
        for log in missing_article_logs[:15]:
            print(f"       - {log}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    execute_mapping_v9(
        'knowledge_graph_connected.json', 
        'legal_content.json', 
        'knowledge_graph_final.json'
    )