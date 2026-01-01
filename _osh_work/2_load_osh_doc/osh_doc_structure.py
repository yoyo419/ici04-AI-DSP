# Legal Reasoning Project, NCCU (2025)
# osh_doc_structure.py: Process occupational safety incident documenting PDFs into structured JSON

# Note: This code cannot perfectly extract all incidents due to the variability in document formatting.
# However, it ensures that each extracted incident contains almost complete elements.
# Those not well extracted will be manually revised in subsequent updates.
# 災害類型跟媒介物分類的參考資料：https://mse.ntust.edu.tw/var/file/19/1019/img/790/850509961.pdf

import os
import re
import json
import fitz  # PyMuPDF
from typing import Dict, List, Optional
from openai import OpenAI
from pathlib import Path

class IncidentPDFProcessor:
    def __init__(self, api_key: str):
        """
        初始化處理器
        
        Args:
            api_key: OpenAI API密鑰
        """
        self.client = OpenAI(api_key=api_key)
        
        # 從文件載入分類定義
        self.incident_types = self._load_incident_types()
        self.medium_types = self._load_medium_types()
        
    def _load_incident_types(self) -> Dict:
        """載入災害類型分類"""
        return {
            "1": "墜落, 滾落",
            "2": "跌倒",
            "3": "衝撞",
            "4": "物體飛落",
            "5": "物體倒塌, 崩塌",
            "6": "被撞",
            "7": "被夾, 被捲",
            "8": "被切, 割, 擦傷",
            "9": "踩踏",
            "10": "溺斃",
            "11": "與高溫, 低溫接觸",
            "12": "與有害物等之接觸",
            "13": "感電",
            "14": "爆炸",
            "15": "物體破裂",
            "16": "火災",
            "17": "不當動作",
            "18": "其他",
            "19": "無法歸類者",
            "21": "公路交通事故",
            "22": "鐵路交通事故",
            "23": "船舶, 航空等交通事故",
            "29": "其他交通事故"
        }
    
    def _load_medium_types(self) -> Dict:
        """載入媒介物分類"""
        return {
            "general": {
                "1": "動力機械",
                "2": "裝卸運搬機械",
                "3": "其他設備",
                "4": "營建物及施工設備",
                "5": "物質材料",
                "6": "貨物",
                "7": "環境",
                "9": "其他類"
            },
            "normal": {
                "11": "原動機", "12": "動力傳導裝置", "13": "木材加工用機械",
                "14": "營造用機械", "15": "一般動力機械", "21": "起重機械",
                "22": "動力運搬機械", "23": "交通工具", "31": "壓力容器類",
                "32": "化學設備", "33": "熔接設備", "34": "爐窯等",
                "35": "電氣設備", "36": "人力機械工具", "37": "用具",
                "39": "其他設備", "41": "營建物及施工設備", "51": "危險物, 有害物",
                "52": "材料", "61": "運搬物體", "71": "環境",
                "91": "其他媒介物", "92": "無媒介物", "99": "不能分類"
            },
            "specific": {
                "111": "原動機", "121": "傳動軸", "122": "傳動輪", "123": "齒輪",
                "129": "其他", "131": "圓鋸", "132": "帶鋸", "133": "鉋面鋸",
                "139": "其他", "141": "牽引機類設備", "142": "動力鏟類設備",
                "143": "打樁機, 拔樁機", "149": "其他", "151": "車床",
                "152": "鑽床", "153": "研磨床", "154": "沖床, 剪床",
                "155": "鍛壓鎚", "156": "離心機", "157": "混合機, 粉碎機",
                "158": "輥筒機", "159": "其他", "211": "起重機",
                "212": "移動式起重機", "213": "人字臂起重機", "214": "升降機, 提升機",
                "215": "船舶裝卸裝置", "216": "吊籠", "217": "機械運材、索道機械、集材裝置",
                "218": "固定式起重機", "219": "其他", "221": "卡車",
                "222": "堆高機", "223": "事業內, 軌道設", "224": "輸送帶",
                "229": "其他", "231": "汽車, 公共汽車", "232": "火車",
                "233": "其他", "311": "鍋爐", "312": "壓力容器",
                "319": "其他", "321": "化學設備", "331": "氣體熔接",
                "332": "電弧熔接", "339": "其他", "341": "爐窯等",
                "351": "輸配電線路", "352": "電力設備", "353": "其他",
                "361": "人力起重機", "362": "人力運搬機", "363": "人力機械",
                "364": "手工具", "371": "梯子等", "372": "吊掛鉤具",
                "379": "其他", "391": "其他設備", "411": "施工架",
                "412": "支撐架", "413": "樓梯, 梯道", "414": "開口部份",
                "415": "屋頂, 屋架, 樑", "416": "工作台, 踏板", "417": "通路",
                "418": "營建物", "419": "其他", "511": "爆炸性物質",
                "512": "引火性物質", "513": "可燃性氣體", "514": "有害物",
                "515": "輻射線", "519": "其他", "521": "金屬材料",
                "522": "木材, 竹材", "523": "石頭, 砂, 小石子", "529": "其他",
                "611": "已包裝貨物", "612": "未包裝機械", "711": "土砂, 岩石",
                "712": "立木", "713": "水", "714": "特殊環境",
                "715": "高低溫環境", "719": "其他", "911": "其他媒介物",
                "999": "不能分類"
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        從PDF提取文字內容
        
        Args:
            pdf_path: PDF檔案路徑
            
        Returns:
            提取的文字內容
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text += page.get_text()
        
        doc.close()
        return full_text
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        從文字中提取各個事件段落
        使用多重策略確保最大化事件提取率
        """
        incidents = []
        
        print("  策略 1：使用標準格式分割（一、行業種類）")
        # 策略1：標準的「一、行業種類」分割
        pattern1 = r'(?:一、|1、|一，)\s*行業[種类]類[：:]\s*([^\n]+).*?(?=(?:一、|1、|一，)\s*行業|$)'
        matches1 = list(re.finditer(pattern1, text, re.DOTALL))
        print(f"    找到 {len(matches1)} 個事件")
        
        if len(matches1) >= 5:  # 如果找到足夠多的事件，使用這個策略
            for i, match in enumerate(matches1, 1):
                incident_text = match.group(0)
                incident_data = self._parse_incident_text(incident_text)
                if incident_data and len(incident_data) >= 3:
                    incidents.append(incident_data)
            return incidents
        
        print("  策略 2：使用寬鬆格式分割（包含變體）")
        # 策略2：更寬鬆的分割，包含各種可能的格式變體
        pattern2 = r'一[、．,:：]\s*行業[種类]類[：:、．,\s]*[^\n]+.*?(?=一[、．,:：]\s*行業|$)'
        matches2 = list(re.finditer(pattern2, text, re.DOTALL | re.IGNORECASE))
        print(f"    找到 {len(matches2)} 個事件")
        
        if len(matches2) >= 5:
            for i, match in enumerate(matches2, 1):
                incident_text = match.group(0)
                incident_data = self._parse_incident_text(incident_text)
                if incident_data and len(incident_data) >= 3:
                    incidents.append(incident_data)
            return incidents
        
        print("  策略 3：基於章節標題分割")
        # 策略3：尋找包含「從事...作業」的標題作為分界點
        pattern3 = r'從事.{2,50}作業.*?災害.*?(?=從事.{2,50}作業|$)'
        matches3 = list(re.finditer(pattern3, text, re.DOTALL))
        print(f"    找到 {len(matches3)} 個潛在事件區塊")

        # 用於策略內去重
        temp_signatures = set()

        for i, match in enumerate(matches3, 1):
            # 向前擴展，尋找「一、行業種類」
            start_pos = max(0, match.start() - 500)
            extended_text = text[start_pos:match.end()]
            
            # 檢查是否包含必要的欄位
            if '行業' in extended_text and '災害' in extended_text:
                incident_data = self._parse_incident_text(extended_text)
                if incident_data and len(incident_data) >= 3:
                    # 策略內去重
                    temp_sig = incident_data.get('description', '')[:100]
                    temp_sig = re.sub(r'[\s、。，]', '', temp_sig)
                    if temp_sig not in temp_signatures:
                        temp_signatures.add(temp_sig)
                        incidents.append(incident_data)

        if len(incidents) >= 5:
            return incidents
        
        print("  策略 4：固定長度分割（最後手段）")
        # 策略4：如果前面都失敗，使用固定長度分割
        # 平均每個事件約 2000-3000 字
        chunk_size = 2500
        overlap = 500
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if '行業' in chunk:  # 確保包含基本資訊
                incident_data = self._parse_incident_text(chunk)
                if incident_data and len(incident_data) >= 3:
                    # 更嚴格的去重檢查
                    new_sig = re.sub(r'[\s、。，]', '', 
                                incident_data.get('industry', '')[:30] + 
                                incident_data.get('description', '')[:80])
                    
                    # 檢查是否與已有事件重複
                    is_duplicate = False
                    for existing in incidents:
                        exist_sig = re.sub(r'[\s、。，]', '', 
                                        existing.get('industry', '')[:30] + 
                                        existing.get('description', '')[:80])
                        if new_sig and exist_sig and new_sig == exist_sig:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        incidents.append(incident_data)
        
        # 去重：根據關鍵欄位判斷是否為重複事件
        print(f"  去重前找到 {len(incidents)} 個事件")
        unique_incidents = []
        seen_signatures = set()

        for incident in incidents:
            # 建立事件的唯一簽名（使用前50字的關鍵欄位）
            signature_parts = [
                incident.get('industry', '')[:50],
                incident.get('incident', '')[:30],
                incident.get('description', '')[:100]
            ]
            signature = '|'.join(signature_parts).lower().strip()
            
            # 移除空白和標點符號，使比對更準確
            signature = re.sub(r'[\s、。，！？；：]', '', signature)
            
            if signature and signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_incidents.append(incident)
            else:
                if signature:
                    print(f"    ⚠️  偵測到重複事件，已跳過")

        print(f"  去重後剩餘 {len(unique_incidents)} 個有效事件")
        return unique_incidents
    
    def _parse_incident_text(self, text: str) -> Dict[str, str]:
        """解析單個事件文字，使用超級寬鬆的匹配規則"""
        data = {}
        
        # 超級寬鬆的正則表達式模式
        patterns = {
            'industry': [
                r'[一1１][、．,:：，。]\s*行業[種类]類[：:、．,，。\s]*([^\n]{2,50})',
                r'行業[種类]類[：:、．,，。\s]*([^\n]{2,50})',
                r'一[^二]*?行業[^：:]*?[：:]([^\n]+)',
            ],
            'incident': [
                r'[二2２][、．,:：，。]\s*災害[類类]型[：:、．,，。\s]*([^\n]+)',
                r'災害[類类]型[：:、．,，。\s]*([^\n]+)',
                r'二[^三]*?災害[^：:]*?[：:]([^\n]+)',
            ],
            'medium_type': [
                r'[三3３][、．,:：，。]\s*媒\s*介\s*物[：:、．,，。\s]*([^\n]+)',
                r'媒\s*介\s*物[：:、．,，。\s]*([^\n]+)',
                r'三[^四五]*?媒[^：:]{0,5}[：:]([^\n]+)',
            ],
            'description': [
                r'[五5５][、．,:：，。]\s*災害發生經過[：:、．,，。\s]*(.+?)(?=[六七八6７８][、．,:：，。]|$)',
                r'災害發生經過[：:、．,，。\s]*(.+?)(?=災害[原因防]|六[、．]|$)',
                r'五[^六七八]*?經過[^：:]{0,10}[：:](.+?)(?=六[、．]|災害原因|$)',
            ],
            'cause_analysis': [
                r'[六6６][、．,:：，。]\s*災害原因分析[：:、．,，。\s]*(.+?)(?=[七八7８][、．,:：，。]|$)',
                r'災害原因分析[：:、．,，。\s]*(.+?)(?=災害防|七[、．]|$)',
                r'六[^七八]*?原因[^：:]{0,10}[：:](.+?)(?=七[、．]|災害防|$)',
            ],
            'preventive_measures': [
                # 匹配到「八、」或檔案結尾，但不匹配「現場」（因為可能在對策內容中出現）
                r'[七7７][、．,:：，。]\s*災害防[止]?對策[：:、．,，。\s]*(.+?)(?=八[、．,:：，。]\s*(?:災害示意圖|現場示意圖|照片)|$)',
                r'災害防[止]?對策[：:、．,，。\s]*(.+?)(?=八[、．,:：，。]\s*(?:災害示意圖|現場示意圖|照片)|$)',
                r'七[^八九十]*?對策[^：:]{0,10}[：:](.+?)(?=八[、．,:：，。]\s*(?:災害示意圖|現場示意圖|照片)|$)',
                # 備用：匹配到檔案結尾
                r'[七7７][、．,:：，。]\s*災害防[止]?對策[：:、．,，。\s]*(.+)',
            ]
        }
        
        # 對每個欄位嘗試多個模式
        for key, pattern_list in patterns.items():
            found = False
            for pattern in pattern_list:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    # 清理內容：移除過多的空白和換行（但保留基本格式）
                    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # 多個換行變成兩個
                    content = re.sub(r'[ \t]+', ' ', content)  # 多個空格變成一個
                    
                    # 對 preventive_measures 不做長度限制
                    if key != 'preventive_measures':
                        content = content[:5000]  # 其他欄位限制長度
                    else:
                        content = content[:50000]  # preventive_measures 允許更長（約50KB）
                    
                    data[key] = content
                    found = True
                    break
            
            if not found and key not in ['medium_type']:  # medium_type 可以沒有
                print(f"      ⚠ 未找到欄位 '{key}'")
        
        # 檢查是否至少有 3 個必要欄位
        required_fields = ['industry', 'incident', 'description']
        found_required = sum(1 for field in required_fields if field in data and data[field])
        
        if found_required >= 2:  # 降低門檻：至少2個必要欄位
            return data
        else:
            return None
    
    def extract_basic_fields(self, incident_text: str) -> Dict:
        """
        提取基本欄位（不需要AI的欄位）
        
        Args:
            incident_text: 單一事故的文本
            
        Returns:
            包含基本欄位的字典
        """
        result = {}
        
        # 提取行業種類
        industry_match = re.search(r'一、行業種類[：:]\s*(.+?)(?=\n|二、)', incident_text)
        if industry_match:
            result['industry'] = industry_match.group(1).strip()
        
        # 提取災害類型
        incident_match = re.search(r'二、災害類型[：:]\s*(.+?)(?=\n|三、)', incident_text)
        if incident_match:
            result['incident'] = incident_match.group(1).strip()
        
        # 提取媒介物
        medium_match = re.search(r'三、媒介物[：:]\s*(.+?)(?=\n|四、)', incident_text)
        if medium_match:
            result['medium_type'] = medium_match.group(1).strip()
        
        # 提取災害發生經過
        desc_match = re.search(r'五、災害發生經過[：:]\s*(.+?)(?=六、|$)', incident_text, re.DOTALL)
        if desc_match:
            result['description'] = desc_match.group(1).strip()
        
        # 提取災害原因分析
        cause_match = re.search(r'六、災害原因分析[：:]\s*(.+?)(?=七、|$)', incident_text, re.DOTALL)
        if cause_match:
            result['cause_analysis'] = cause_match.group(1).strip()
        
        # 提取災害防止對策
        prevent_match = re.search(r'七、災害防止對策[：:]\s*(.+?)(?=八、|$)', incident_text, re.DOTALL)
        if prevent_match:
            result['preventive_measures'] = prevent_match.group(1).strip()
        
        return result
    
    def classify_incident_type(self, incident: str) -> tuple:
        """
        使用OpenAI API分類災害類型
        
        Args:
            incident: 災害類型文字
            
        Returns:
            (災害類型, 災害類型ID)
        """
        if not incident or incident.strip() == "":
            return "", ""
            
        prompt = f"""請根據以下災害類型分類表，判斷「{incident}」最符合哪一個類別。

災害類型分類表：
{json.dumps(self.incident_types, ensure_ascii=False, indent=2)}

重要指示：
1. 仔細比對輸入的災害類型與分類表中的描述
2. 選擇最相符的類別
3. 只回傳JSON格式，不要有任何其他文字
4. JSON格式：{{"type": "類型名稱", "id": "編號"}}

範例：
輸入：墜落
輸出：{{"type": "墜落, 滾落", "id": "1"}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是職業安全專家，專門負責災害分類。請嚴格按照指示回傳JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0
            )
            
            content = response.choices[0].message.content.strip()
            # 清理可能的 markdown 格式
            content = content.replace('```json', '').replace('```', '').strip()
            
            result = json.loads(content)
            return result.get('type', ''), result.get('id', '')
        except Exception as e:
            print(f"分類災害類型時發生錯誤: {e}")
            print(f"輸入: {incident}")
            return "", ""
    
    def classify_medium_type(self, medium: str) -> tuple:
        """
        使用OpenAI API分類媒介物
        
        Args:
            medium: 媒介物文字
            
        Returns:
            (大類別, 大類別ID, 類別, 類別ID, 項目, 項目ID)
        """
        if not medium or medium.strip() == "":
            return "", "", "", "", "", ""
            
        prompt = f"""請根據以下媒介物分類表，判斷「{medium}」最符合哪一個類別。

媒介物大類別：
{json.dumps(self.medium_types['general'], ensure_ascii=False, indent=2)}

媒介物類別：
{json.dumps(self.medium_types['normal'], ensure_ascii=False, indent=2)}

媒介物項目：
{json.dumps(self.medium_types['specific'], ensure_ascii=False, indent=2)}

重要指示：
1. 需要找出三個層級的分類：大類別、類別、項目
2. 大類別ID是個位數（1-9）
3. 類別ID是十位數（11-99）
4. 項目ID是百位數（111-999）
5. 只回傳JSON格式，不要有任何其他文字

JSON格式：
{{
    "general": "大類別名稱",
    "general_id": "大類別ID",
    "normal": "類別名稱",
    "normal_id": "類別ID",
    "specific": "項目名稱",
    "specific_id": "項目ID"
}}

範例：
輸入：固定式起重機
輸出：
{{
    "general": "裝卸運搬機械",
    "general_id": "2",
    "normal": "起重機械",
    "normal_id": "21",
    "specific": "固定式起重機",
    "specific_id": "218"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是職業安全專家，專門負責媒介物分類。請嚴格按照指示回傳JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0
            )
            
            content = response.choices[0].message.content.strip()
            # 清理可能的 markdown 格式
            content = content.replace('```json', '').replace('```', '').strip()
            
            result = json.loads(content)
            return (
                result.get('general', ''), result.get('general_id', ''),
                result.get('normal', ''), result.get('normal_id', ''),
                result.get('specific', ''), result.get('specific_id', '')
            )
        except Exception as e:
            print(f"分類媒介物時發生錯誤: {e}")
            print(f"輸入: {medium}")
            return "", "", "", "", "", ""
    
    def generate_description_summary(self, description: str) -> str:
        """
        使用OpenAI API生成災害發生經過摘要
        
        Args:
            description: 災害發生經過完整描述
            
        Returns:
            摘要文字（60-100字）
        """
        if not description or description.strip() == "":
            return ""
            
        prompt = f"""請將以下災害發生經過濃縮成60-100字的摘要。

災害發生經過：
{description}

重要指示：
1. 摘要必須簡潔明確，包含關鍵資訊
2. 字數控制在60-100字之間
3. 只回傳摘要文字，不要有任何其他說明
4. 不要使用引號或其他格式

範例：
輸入：據○○有限公司所僱勞工林○○稱：103年5月15日約12時37分許，移動式起重機操作手陳○○(罹災者)操作履帶移動式起重機，從事橋墩全套管基樁工程之挖掘作業時，突然聽到碰一聲，看到該起重機之桁架與吊車左邊連結處斷裂，且桁架向右傾壓到駕駛室，致駕駛室內起重機操作手陳○○當場死亡。
輸出：陳罹災者操作履帶移動式起重機進行橋墩全套管基樁工程之挖掘作業時，桁架與吊車左側連結處斷裂，導致桁架傾壓駕駛室，陳當場死亡。
重要關鍵資訊：履帶移動式起重機 (媒介物)、橋墩全套管基樁工程 (重要事件細節)、桁架與吊車左側連結處斷裂 (重要事件細節)
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是職業安全文件撰寫專家，擅長濃縮災害報告。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            # 移除可能的引號
            summary = summary.strip('"\'')
            return summary
        except Exception as e:
            print(f"生成摘要時發生錯誤: {e}")
            return ""
    
    def generate_cause_summary(self, cause_analysis: str) -> str:
        """
        使用OpenAI API生成災害原因摘要
        
        Args:
            cause_analysis: 災害原因分析完整內容
            
        Returns:
            結構化的原因摘要
        """
        if not cause_analysis or cause_analysis.strip() == "":
            return ""
            
        prompt = f"""請分析以下災害原因，並以結構化方式摘要。

災害原因分析：
{cause_analysis}

重要指示：
1. 必須明確指出主體（勞工或雇主）
2. 說明具體行為（未架設、未使用、未禁止、未辦理、未訂定等）
3. 包含相關設備、規則或活動
4. 用「、」分隔各項原因
5. 只回傳摘要文字，不要有其他說明
6. 如果原文沒有明確提到主體，請根據行為類型聰明判斷

行為類型判斷原則：
- 未架設設備、未使用防護具 → 通常是勞工
- 未禁止危險行為、未辦理訓練、未訂定規則 → 通常是雇主

範例格式：
「勞工未架設施工架與工作臺、勞工未使用安全帶與安全帽等防護工具、雇主未禁止勞工搭載堆高機除乘坐席外的位置、雇主未辦理勞工安全衛生教育訓練、雇主未訂定適合之安全衛生工作守則」

請直接回傳摘要："""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是職業安全分析專家，擅長識別災害成因並歸納責任主體。請嚴格按照指示格式輸出。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            # 移除可能的引號
            summary = summary.strip('"\'')
            return summary
        except Exception as e:
            print(f"生成原因摘要時發生錯誤: {e}")
            return ""
    
    def extract_regulations(self, preventive_measures: str) -> str:
        """
        使用OpenAI API提取法規條文
        
        Args:
            preventive_measures: 災害防止對策完整內容
            
        Returns:
            以逗號分隔的法規條文列表
        """
        if not preventive_measures or preventive_measures.strip() == "":
            return ""
            
        prompt = f"""請從以下災害防止對策中提取所有法規條文名稱。

災害防止對策：
{preventive_measures}

重要指示：
1. 提取所有完整的法規條文名稱（包含法律名稱和條文編號）
2. 用半形逗號加空格 ", " 分隔各法規
3. 保持原文的法規名稱格式
4. 只回傳法規列表，不要有其他說明文字
5. 如果有「暨」連接多個法條，請保持完整

範例格式：
「勞工安全衛生法第25條第1項, 勞工安全衛生法第14條第2項, 勞工安全衛生組織管理及自動檢查辦法第12條之1第2項暨勞工安全衛生法第14條第3項」

請直接回傳法規列表："""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是法規文件處理專家，擅長提取和整理法規條文。請嚴格按照指示格式輸出。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0
            )
            
            regulations = response.choices[0].message.content.strip()
            # 移除可能的引號
            regulations = regulations.strip('"\'')
            # 移除所有空格（包含全形和半形空格）
            regulations = regulations.replace(' ', '').replace('　', '')
            return regulations
        except Exception as e:
            print(f"提取法規時發生錯誤: {e}")
            return ""
    
    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        處理完整的PDF檔案
        
        Args:
            pdf_path: PDF檔案路徑
            output_path: 輸出JSON檔案路徑（可選）
            
        Returns:
            所有事故案例的列表
        """
        print(f"正在讀取PDF檔案: {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)
        
        print("正在分割事故案例...")
        incidents = self.extract_sections(text)
        print(f"找到 {len(incidents)} 個事故案例")
        
        results = []
        processed_signatures = set()

        for i, incident_data in enumerate(incidents, 1):
            print(f"\n處理第 {i}/{len(incidents)} 個事故案例...")
            
            # 建立簽名檢查是否重複
            check_sig = (
                incident_data.get('industry', '')[:30] + '|' +
                incident_data.get('incident', '')[:20] + '|' +
                incident_data.get('description', '')[:80]
            )
            check_sig = re.sub(r'[\s、。，！？；：]', '', check_sig)
            
            if check_sig in processed_signatures:
                print(f"  ⚠️  偵測到重複事件（最終檢查），跳過處理")
                continue
            
            processed_signatures.add(check_sig)
            
            print(f"  開始處理事件：{incident_data.get('industry', '未知')[:20]}...")
            print(f"  已取得欄位：{list(incident_data.keys())}")
            result = self.process_incident_data(incident_data)
            if result:
                results.append(result)
        
        # 儲存結果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n結果已儲存至: {output_path}")
        
        return results

    def process_folder(self, folder_path: str, output_folder: Optional[str] = None):
        """
        處理資料夾中的所有 PDF 檔案
        
        Args:
            folder_path: 包含 PDF 檔案的資料夾路徑
            output_folder: 輸出 JSON 檔案的資料夾路徑（可選，預設為與 PDF 相同位置）
        """
        from pathlib import Path
        
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ 錯誤：資料夾不存在 - {folder_path}")
            return
        
        if not folder.is_dir():
            print(f"❌ 錯誤：路徑不是資料夾 - {folder_path}")
            return
        
        # 找出所有 PDF 檔案
        pdf_files = list(folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ 在 {folder_path} 中沒有找到 PDF 檔案")
            return
        
        print(f"📁 找到 {len(pdf_files)} 個 PDF 檔案")
        print("=" * 60)
        
        # 設定輸出資料夾
        if output_folder:
            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = folder
        
        # 統計資訊
        total_incidents = 0
        successful_files = 0
        failed_files = []
        
        # 處理每個 PDF 檔案
        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"\n{'='*60}")
            print(f"📄 [{idx}/{len(pdf_files)}] 處理檔案: {pdf_file.name}")
            print(f"{'='*60}")
            
            try:
                # 生成輸出檔案名稱
                output_file = output_dir / f"{pdf_file.stem}_結構化資料.json"
                
                # 處理單一 PDF
                results = self.process_pdf(str(pdf_file), str(output_file))
                
                if results:
                    total_incidents += len(results)
                    successful_files += 1
                    print(f"✅ 成功處理：{pdf_file.name}")
                    print(f"   提取事件數：{len(results)} 個")
                    print(f"   輸出檔案：{output_file.name}")
                else:
                    failed_files.append(pdf_file.name)
                    print(f"⚠️  未能提取任何事件：{pdf_file.name}")
            
            except Exception as e:
                failed_files.append(pdf_file.name)
                print(f"❌ 處理失敗：{pdf_file.name}")
                print(f"   錯誤訊息：{str(e)}")
        
        # 輸出總結
        print(f"\n{'='*60}")
        print(f"📊 處理完成統計")
        print(f"{'='*60}")
        print(f"總檔案數：{len(pdf_files)}")
        print(f"成功處理：{successful_files}")
        print(f"失敗檔案：{len(failed_files)}")
        print(f"總事件數：{total_incidents}")
        
        if failed_files:
            print(f"\n❌ 失敗的檔案清單：")
            for filename in failed_files:
                print(f"   - {filename}")
        
        print(f"\n📁 所有輸出檔案位於：{output_dir}")
    
    def process_incident_data(self, incident_data: Dict[str, str]) -> Dict:
        """
        處理已解析的事件資料（從 extract_sections 來的）
        
        Args:
            incident_data: 已解析的事件資料字典
            
        Returns:
            完整的結構化 JSON 資料
        """
        result = {}
        
        # 直接使用已解析的基本欄位
        result['industry'] = incident_data.get('industry', '')
        result['incident'] = incident_data.get('incident', '')
        result['medium_type'] = incident_data.get('medium_type', '')
        result['description'] = incident_data.get('description', '')
        result['cause_analysis'] = incident_data.get('cause_analysis', '')
        result['preventive_measures'] = incident_data.get('preventive_measures', '')
        
        # 使用 OpenAI 分類災害類型
        if result['incident']:
            print("  正在分類災害類型...")
            incident_type, incident_type_id = self.classify_incident_type(result['incident'])
            result['incident_type'] = incident_type
            result['incident_type_id'] = incident_type_id
        
        # 使用 OpenAI 分類媒介物
        if result['medium_type']:
            print("  正在分類媒介物...")
            general, general_id, normal, normal_id, specific, specific_id = \
                self.classify_medium_type(result['medium_type'])
            result['medium_type_general'] = general
            result['medium_type_general_id'] = general_id
            result['medium_type_normal'] = normal
            result['medium_type_normal_id'] = normal_id
            result['medium_type_specific'] = specific
            result['medium_type_specific_id'] = specific_id
        else:
            # 沒有媒介物資料時，設定空字串
            result['medium_type_general'] = ''
            result['medium_type_general_id'] = ''
            result['medium_type_normal'] = ''
            result['medium_type_normal_id'] = ''
            result['medium_type_specific'] = ''
            result['medium_type_specific_id'] = ''
        
        # 生成描述摘要
        if result['description']:
            print("  正在生成災害發生經過摘要...")
            result['description_summary'] = self.generate_description_summary(result['description'])
        
        # 生成原因摘要
        if result['cause_analysis']:
            print("  正在生成災害原因摘要...")
            result['cause_summary'] = self.generate_cause_summary(result['cause_analysis'])
        
        # 提取法規
        if result['preventive_measures']:
            print("  正在提取法規條文...")
            result['preventive_regulations'] = self.extract_regulations(result['preventive_measures'])
        
        return result

def main():
    """主程式"""
    from pathlib import Path
    
    # ================== 設定區 ==================
    # 設定 OpenAI API 密鑰
    API_KEY = "sk-YOUR_API_KEY"

    # 選擇處理模式：'folder' 或 'single'
    MODE = 'folder'
    # MODE = 'single' 

    # 資料夾模式設定
    FOLDER_PATH = "./osh_case_folder_1"
    OUTPUT_FOLDER = "./extraction_output"  # None 表示輸出到與 PDF 相同位置，也可指定路徑
    
    # 單一檔案模式設定
    SINGLE_PDF_PATH = "./osh_case_folder_3/109年從事電梯系統更新作業發生感電災害致死重大職業災害案例.pdf"
    # ============================================
    
    print("🚀 職業災害 PDF 資訊抽取系統")
    print("=" * 60)
    
    # 建立處理器
    processor = IncidentPDFProcessor(API_KEY)
    
    if MODE == 'folder':
        # 資料夾模式
        print(f"📂 模式：批次處理資料夾")
        print(f"📁 輸入資料夾：{FOLDER_PATH}")
        if OUTPUT_FOLDER:
            print(f"📁 輸出資料夾：{OUTPUT_FOLDER}")
        else:
            print(f"📁 輸出資料夾：與 PDF 檔案相同位置")
        print("=" * 60)
        
        processor.process_folder(FOLDER_PATH, OUTPUT_FOLDER)
    
    elif MODE == 'single':
        # 單一檔案模式
        print(f"📄 模式：處理單一檔案")
        pdf_path = Path(SINGLE_PDF_PATH)
        output_path = pdf_path.parent / f"{pdf_path.stem}_結構化資料.json"
        
        print(f"📄 輸入檔案：{pdf_path.name}")
        print(f"📄 輸出檔案：{output_path.name}")
        print("=" * 60)
        
        results = processor.process_pdf(str(pdf_path), str(output_path))
        
        if results:
            print(f"\n✅ 處理完成！共處理 {len(results)} 個事故案例")
        else:
            print(f"\n⚠️  未能提取任何事件")
    
    else:
        print(f"❌ 錯誤：無效的模式 '{MODE}'，請設定為 'folder' 或 'single'")


if __name__ == "__main__":
    main()