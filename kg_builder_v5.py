"""
Legal Knowledge Graph Builder v4 for Occupational Safety and Health Laws
職業安全衛生法律知識圖譜建構器 v4 (智慧語意聚類與語境感知版)

核心改進 (基於教授的深度分析):
1. HDBSCAN 自適應聚類 - 解決單例聚類災難,自動過濾噪聲
2. 語意規則匹配 - 使用向量相似度取代關鍵字匹配
3. 語境感知義務萃取 - 處理法律引用(Anaphora)問題
4. 統一優先級邏輯 - 修正人機迴圈(HITL)的內部矛盾
"""

import json
import os
import sys
import getpass
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
from openai import OpenAI
import re
from dataclasses import dataclass, asdict, field
from enum import Enum
import ast
from datetime import datetime

# ============================================================================
# 資料結構定義 (繼承 v3)
# ============================================================================

@dataclass
class GraphNode:
    """知識圖譜節點"""
    id: str
    type: str
    properties: Dict

@dataclass
class GraphEdge:
    """知識圖譜邊"""
    source: str
    target: str
    type: str
    properties: Dict = None

class ControlType(Enum):
    """風險控制層級"""
    ENGINEERING = "EngineeringControl"
    ADMINISTRATIVE = "AdministrativeControl"
    PPE = "PersonalProtectiveEquipment"
    ELIMINATION = "EliminationControl"
    SUBSTITUTION = "SubstitutionControl"

class ReviewStatus(Enum):
    """審核狀態"""
    AUTO_APPROVED = "auto_approved"
    PENDING_REVIEW = "pending_review"
    HUMAN_VERIFIED = "human_verified"
    REJECTED = "rejected"

class ReviewPriority(Enum):
    """審核優先級"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class LegalEvent:
    """法律事件結構"""
    event_id: str
    action: str
    actor: str
    patients: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    temporal: Optional[str] = None
    purpose: Optional[str] = None
    source_article: str = ""
    confidence: float = 0.0

@dataclass
class RuleTemplate:
    """可計算規則模板"""
    rule_id: str
    rule_name: str
    category: str
    pattern: str
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    control_type_affinity: Dict[str, float] = field(default_factory=dict)
    embedding_vector: Optional[np.ndarray] = None  # v4 新增: 規則的語意向量

@dataclass
class StructuredEvidence:
    """結構化證據"""
    keywords_matched: List[str] = field(default_factory=list)
    decision_rule_id: str = ""
    decision_rule_name: str = ""
    rule_similarity_score: float = 0.0  # v4 新增: 與規則的相似度分數
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    alternative_classifications: List[Dict[str, Any]] = field(default_factory=list)
    text_snippets: List[str] = field(default_factory=list)
    extracted_events: List[str] = field(default_factory=list)

@dataclass
class ClassificationResult:
    """分類結果"""
    classification: str
    confidence: float
    evidence: StructuredEvidence
    review_status: ReviewStatus
    review_priority: ReviewPriority = ReviewPriority.LOW
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    human_feedback: Optional[str] = None

@dataclass
class ActiveLearningScore:
    """主動學習評分"""
    uncertainty_score: float = 0.0
    impact_score: float = 0.0
    frequency_score: float = 0.0
    complexity_score: float = 0.0
    total_priority: float = 0.0

@dataclass
class ClusterQualityMetrics:
    """聚類品質指標"""
    silhouette_score: float = 0.0
    avg_intra_similarity: float = 0.0
    min_member_similarity: float = 0.0
    is_singleton: bool = False
    is_noise: bool = False  # v4 新增: HDBSCAN 噪聲標記
    needs_review: bool = False
    review_reason: str = ""
    review_priority: ReviewPriority = ReviewPriority.LOW
    active_learning_score: Optional[ActiveLearningScore] = None


class LegalKGBuilderV4:
    """法律知識圖譜建構器 v4 (智慧語意聚類與語境感知版)"""
    
    def __init__(self, api_key: str, input_path: str, output_dir: str = "./output"):
        """
        初始化建構器
        
        Args:
            api_key: OpenAI API Key
            input_path: all_documents.json 的路徑
            output_dir: 輸出目錄
        """
        self.client = OpenAI(api_key=api_key)
        self.input_path = input_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 人機迴圈佇列
        self.review_queue_dir = os.path.join(output_dir, "review_queue")
        os.makedirs(self.review_queue_dir, exist_ok=True)
        
        # 資料結構
        self.documents = []
        self.nodes = []
        self.edges = []
        
        # 本體與正規化
        self.subject_ontology = {}
        self.object_ontology = {}
        self.obligation_clusters = {}
        self.control_type_mapping = {}
        self.ontology_embeddings = {} # v5 本體實體 embeddings
        
        # 事件抽取相關
        self.legal_events = {}
        self.discovered_entities = set()
        
        # === v4 新增: 可計算規則庫 (含語意向量) ===
        self.rule_base = {}
        self.rule_embeddings = {}  # rule_id -> embedding vector
        self._initialize_rule_base()
        
        # 審核佇列
        self.low_confidence_classifications = []
        self.problematic_clusters = []
        
        # 品質閾值
        self.CONFIDENCE_THRESHOLD = 0.75
        self.MIN_CLUSTER_SIZE = 2
        
        # === v4 新增: HDBSCAN 參數 ===
        self.HDBSCAN_MIN_CLUSTER_SIZE = 2
        self.HDBSCAN_MIN_SAMPLES = 1
        self.HDBSCAN_METRIC = 'euclidean'
        
        # 主動學習參數
        self.PRIORITY_WEIGHTS = {
            'uncertainty': 0.35,
            'impact': 0.30,
            'frequency': 0.20,
            'complexity': 0.15
        }
        
        # 快取
        self.embedding_cache = {}
        self.classification_cache = {}
        
        # 初始化基礎本體結構
        self._initialize_base_ontology()
        
    # ========================================================================
    # v4 核心改進 1: 語意規則匹配系統
    # ========================================================================
    
    def _initialize_rule_base(self):
        """
        初始化可計算規則庫 (v4 增強版)
        為每條規則生成語意向量,實現真正的「語意匹配」
        """
        # 規則定義 (繼承自 v3)
        self.rule_base = {
            # 管理控制規則
            "RULE_ADM_01": RuleTemplate(
                rule_id="RULE_ADM_01",
                rule_name="時間管理規則",
                category="AdministrativeControl",
                pattern="涉及作業時間、休息時間、輪班制度等時間管理措施",
                keywords=["時間", "休息", "輪班", "工時", "每日", "定期"],
                examples=["每日作業時間不得超過6小時", "每工作2小時應休息30分鐘"],
                control_type_affinity={"AdministrativeControl": 0.9, "EngineeringControl": 0.1}
            ),
            "RULE_ADM_02": RuleTemplate(
                rule_id="RULE_ADM_02",
                rule_name="檢查與監測規則",
                category="AdministrativeControl",
                pattern="涉及定期檢查、監測、記錄、報告等管理程序",
                keywords=["檢查", "監測", "測定", "記錄", "報告", "定期"],
                examples=["應每月實施定期檢查", "應記錄並保存檢測結果"],
                control_type_affinity={"AdministrativeControl": 0.95, "EngineeringControl": 0.05}
            ),
            "RULE_ADM_03": RuleTemplate(
                rule_id="RULE_ADM_03",
                rule_name="教育訓練規則",
                category="AdministrativeControl",
                pattern="涉及教育、訓練、宣導、指導等能力建構措施",
                keywords=["教育", "訓練", "指導", "宣導", "講習"],
                examples=["應實施安全衛生教育訓練", "應指導勞工正確作業方法"],
                control_type_affinity={"AdministrativeControl": 0.95, "PPE": 0.05}
            ),
            "RULE_ADM_04": RuleTemplate(
                rule_id="RULE_ADM_04",
                rule_name="標示與警告規則",
                category="AdministrativeControl",
                pattern="涉及標示、警告標誌、公告、通知等資訊傳達措施",
                keywords=["標示", "警告", "公告", "標誌", "揭示"],
                examples=["應於明顯處標示警告標誌", "應公告作業注意事項"],
                control_type_affinity={"AdministrativeControl": 0.85, "EngineeringControl": 0.15}
            ),
            
            # 工程控制規則
            "RULE_ENG_01": RuleTemplate(
                rule_id="RULE_ENG_01",
                rule_name="物理屏障規則",
                category="EngineeringControl",
                pattern="涉及護罩、護欄、圍欄、遮蔽等物理性阻隔設施",
                keywords=["護罩", "護欄", "圍欄", "欄杆", "遮蔽", "阻隔"],
                examples=["應設置護欄防止墜落", "應裝設護罩防止接觸"],
                control_type_affinity={"EngineeringControl": 0.95, "AdministrativeControl": 0.05}
            ),
            "RULE_ENG_02": RuleTemplate(
                rule_id="RULE_ENG_02",
                rule_name="通風與換氣規則",
                category="EngineeringControl",
                pattern="涉及通風、換氣、排氣、抽風等空氣品質控制設備",
                keywords=["通風", "換氣", "排氣", "抽風", "局部排氣"],
                examples=["應設置適當之通風設備", "應設局部排氣裝置"],
                control_type_affinity={"EngineeringControl": 0.95, "AdministrativeControl": 0.05}
            ),
            "RULE_ENG_03": RuleTemplate(
                rule_id="RULE_ENG_03",
                rule_name="安全裝置規則",
                category="EngineeringControl",
                pattern="涉及安全裝置、連鎖裝置、緊急停止裝置、極限開關等安全機構",
                keywords=["安全裝置", "連鎖", "緊急停止", "防護裝置", "保護裝置", "極限開關", "限制開關"],
                examples=["應設置緊急停止裝置", "應具備連鎖保護機構", "應設置終點極限開關"],
                control_type_affinity={"EngineeringControl": 0.9, "AdministrativeControl": 0.1}
            ),
            
            # 個人防護具規則
            "RULE_PPE_01": RuleTemplate(
                rule_id="RULE_PPE_01",
                rule_name="頭部防護規則",
                category="PPE",
                pattern="涉及安全帽、頭部防護等個人頭部保護裝備",
                keywords=["安全帽", "工安帽", "頭部", "防護帽"],
                examples=["應使勞工戴用安全帽", "應配戴符合標準之安全帽"],
                control_type_affinity={"PPE": 0.95, "AdministrativeControl": 0.05}
            ),
            "RULE_PPE_02": RuleTemplate(
                rule_id="RULE_PPE_02",
                rule_name="墜落防護規則",
                category="PPE",
                pattern="涉及安全帶、安全索、防墜器等防止墜落之個人裝備",
                keywords=["安全帶", "安全索", "防墜", "安全母索"],
                examples=["應使勞工使用安全帶", "應配掛安全帶於安全母索"],
                control_type_affinity={"PPE": 0.95, "EngineeringControl": 0.05}
            ),
            "RULE_PPE_03": RuleTemplate(
                rule_id="RULE_PPE_03",
                rule_name="呼吸防護規則",
                category="PPE",
                pattern="涉及防護口罩、呼吸防護具等呼吸系統保護裝備",
                keywords=["口罩", "呼吸防護", "防塵口罩", "防毒面具"],
                examples=["應使勞工佩戴防護口罩", "應提供適當之呼吸防護具"],
                control_type_affinity={"PPE": 0.9, "EngineeringControl": 0.1}
            ),
            
            # 消除控制規則
            "RULE_ELIM_01": RuleTemplate(
                rule_id="RULE_ELIM_01",
                rule_name="危害源消除規則",
                category="EliminationControl",
                pattern="涉及完全移除、停止使用、廢除等消除危害源的措施",
                keywords=["禁止", "不得使用", "停止", "移除", "廢除"],
                examples=["禁止使用含石綿材料", "不得使用該有害物質"],
                control_type_affinity={"EliminationControl": 0.95, "SubstitutionControl": 0.05}
            ),
            
            # 替代控制規則
            "RULE_SUB_01": RuleTemplate(
                rule_id="RULE_SUB_01",
                rule_name="材料替代規則",
                category="SubstitutionControl",
                pattern="涉及使用較安全材料、替代品、低危害物質等替代措施",
                keywords=["替代", "改用", "使用其他", "較低危害", "安全替代品"],
                examples=["應改用低毒性溶劑", "應以較安全之材料替代"],
                control_type_affinity={"SubstitutionControl": 0.9, "EliminationControl": 0.1}
            )
        }
        
        print(f"  ✓ 已初始化 {len(self.rule_base)} 個可計算規則")
        
        # === v4 新增: 為每條規則生成語意向量 ===
        print("  → 為規則生成語意向量...")
        self._generate_rule_embeddings()
    
    def _generate_rule_embeddings(self):
        """
        v4 核心方法: 為每條規則生成語意向量
        這是實現「語意規則匹配」的基礎
        """
        for rule_id, rule_template in self.rule_base.items():
            # 組合規則的多個語意來源
            rule_description = f"""
規則名稱: {rule_template.rule_name}
規則模式: {rule_template.pattern}
關鍵詞: {', '.join(rule_template.keywords)}
範例: {' | '.join(rule_template.examples)}
            """.strip()
            
            # 生成向量
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[rule_description]
                )
                embedding = response.data[0].embedding
                self.rule_embeddings[rule_id] = np.array(embedding)
                
                # 同時儲存到 RuleTemplate 物件中
                rule_template.embedding_vector = np.array(embedding)
                
            except Exception as e:
                print(f"    ✗ 規則 {rule_id} 向量生成失敗: {e}")
                self.rule_embeddings[rule_id] = np.zeros(1536)
        
        print(f"  ✓ 已生成 {len(self.rule_embeddings)} 個規則向量")
    
    def _match_rules_with_semantic_similarity(self, 
                                              obligation_text: str,
                                              obligation_vector: Optional[np.ndarray] = None,
                                              top_k: int = 3,
                                              threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        v4 核心方法: 使用語意相似度匹配規則
        取代 v3 的關鍵字匹配,這是解決「問題二」的關鍵
        
        Args:
            obligation_text: 義務文本
            obligation_vector: 義務的 embedding 向量(可選,若無則現場生成)
            top_k: 返回前 k 個最相似的規則
            threshold: 最低相似度閾值
            
        Returns:
            [(rule_id, similarity_score), ...] 按相似度降序排列
        """
        # 如果沒有提供向量,則生成
        if obligation_vector is None:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[obligation_text]
                )
                obligation_vector = np.array(response.data[0].embedding)
            except Exception as e:
                print(f"    ✗ 義務向量生成失敗: {e}")
                return []
        
        # 計算與所有規則的餘弦相似度
        similarities = []
        for rule_id, rule_vector in self.rule_embeddings.items():
            # 餘弦相似度公式: cos(θ) = (A·B) / (||A|| × ||B||)
            cos_sim = np.dot(obligation_vector, rule_vector) / (
                np.linalg.norm(obligation_vector) * np.linalg.norm(rule_vector)
            )
            
            if cos_sim >= threshold:
                similarities.append((rule_id, float(cos_sim)))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    # ========================================================================
    # v4 核心改進 2: 語境感知義務萃取
    # ========================================================================
    
    def _extract_obligations_with_context(self) -> List[Dict[str, Any]]:
        """
        v4 核心方法: 語境感知義務萃取
        解決「問題三」- 處理法律引用(Anaphora)問題
        
        Returns:
            List of dicts with keys: 'text', 'context', 'category', 'source', 'has_anaphora'
        """
        print("  → 執行語境感知義務萃取...")
        
        obligations = []
        
        # 法律引用詞(Anaphora)的正則表達式
        anaphora_pattern = re.compile(r'(前項|前款|前條|第[一二三四五六七八九十百]+款|第[一二三四五六七八九十百]+項|本條|該)')
        
        # 義務關鍵詞
        obligation_patterns = [
            (r'雇主.*?應.*?[。\n]', '雇主義務'),
            (r'事業單位.*?應.*?[。\n]', '事業單位義務'),
            (r'勞工.*?應.*?[。\n]', '勞工義務'),
            (r'應(?:設置|裝設|配置|配備|設立).*?(?:裝置|設備|設施|措施).*?[。\n]', '設置類義務'),
            (r'應(?:實施|辦理|進行|執行).*?(?:檢查|測定|監測|評估).*?[。\n]', '檢查類義務'),
            (r'應(?:訂定|製作|建立|擬定).*?(?:計畫|標準|程序|規定).*?[。\n]', '文件類義務'),
            (r'應(?:使|令|命|要求).*?勞工.*?(?:使用|佩戴|配戴|穿戴).*?[。\n]', '防護具義務'),
            (r'應.*?教育訓練.*?[。\n]', '訓練義務'),
            (r'應.*?標示.*?[。\n]', '標示義務'),
            (r'(?:雇主|事業單位|勞工).*?不得.*?[。\n]', '禁止義務'),
        ]
        
        for doc_idx, doc in enumerate(self.documents):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # 將內容按句子分割(保留原始順序)
            sentences = re.split(r'([。\n])', content)
            sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                # 檢查是否包含義務關鍵詞
                matched_category = None
                for pattern, category in obligation_patterns:
                    if re.search(pattern, sentence):
                        matched_category = category
                        break
                
                if not matched_category:
                    continue
                
                # === 核心邏輯: 檢測法律引用 ===
                has_anaphora = bool(anaphora_pattern.search(sentence))
                
                context = ""
                if has_anaphora:
                    # 如果存在引用,則向前查找語境
                    context_window = 2  # 向前查找2句
                    start_idx = max(0, sent_idx - context_window)
                    context_sentences = sentences[start_idx:sent_idx]
                    context = ' '.join(s.strip() for s in context_sentences if s.strip())
                
                obligations.append({
                    'text': sentence,
                    'context': context,
                    'category': matched_category,
                    'source': metadata.get('chunk_id', f'doc_{doc_idx}'),
                    'has_anaphora': has_anaphora,
                    'full_text_with_context': f"[語境: {context}] {sentence}" if context else sentence
                })
        
        # 去重(基於 full_text_with_context)
        unique_obligations = {}
        for obl in obligations:
            key = obl['full_text_with_context']
            if key not in unique_obligations:
                unique_obligations[key] = obl
        
        result = list(unique_obligations.values())
        
        anaphora_count = sum(1 for o in result if o['has_anaphora'])
        print(f"  ✓ 萃取了 {len(result)} 個義務描述")
        print(f"    → 其中 {anaphora_count} 個包含法律引用,已補充語境")
        
        return result
    
    # ========================================================================
    # v4 核心改進 3: HDBSCAN 自適應聚類
    # ========================================================================
    
    def _cluster_obligations_with_hdbscan(self, 
                                        obligations: List[Dict[str, Any]], 
                                        vectors: np.ndarray) -> Dict[int, List[Dict]]:
        """
        v5 核心改進: 使用葉聚類
        解決問題一 - 保留精確、小型的有意義聚類
        
        關鍵修改:
        1. cluster_selection_method='leaf' (v4 為 'eom')
        2. min_cluster_size=3 (v4 為 2)
        3. 新增小型聚類統計
        """
        if len(obligations) == 0 or vectors.size == 0:
            print("    ✗ 義務列表或向量為空")
            return {}
        
        if vectors.shape[0] != len(obligations):
            print(f"    ✗ 向量數量不一致")
            return {}
        
        try:
            import hdbscan
            print(f"  → 使用 HDBSCAN 葉聚類 (min_cluster_size={self.HDBSCAN_MIN_CLUSTER_SIZE}, method='leaf')...")
            
            # ⚠️ v5 關鍵修改
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=3,  # v5: 從 2 提高到 3
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='leaf'  # v5: 從 'eom' 改為 'leaf'
            )
            
            cluster_labels = clusterer.fit_predict(vectors)
            
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[int(label)].append(obligations[idx])
            
            noise_points = clusters.pop(-1, [])
            
            print(f"  ✓ HDBSCAN 葉聚類完成:")
            print(f"    - 有效聚類: {len(clusters)} 個")
            print(f"    - 噪聲點: {len(noise_points)} 個")
            
            if clusters:
                cluster_sizes = [len(v) for v in clusters.values()]
                print(f"    - 聚類大小: 最小={min(cluster_sizes)}, 最大={max(cluster_sizes)}, 平均={sum(cluster_sizes)/len(cluster_sizes):.1f}")
                
                # ⚠️ v5 新增統計
                small_clusters = sum(1 for size in cluster_sizes if 3 <= size <= 5)
                print(f"    - v5 改進: 保留了 {small_clusters} 個小型精確聚類 (3-5 成員)")
            
            if noise_points:
                self._save_noise_points(noise_points)
            
            return dict(clusters)
            
        except ImportError:
            print("    ✗ 未安裝 hdbscan")
            return self._cluster_obligations_fallback(obligations, vectors)
        except Exception as e:
            print(f"    ✗ HDBSCAN 失敗: {e}")
            return self._cluster_obligations_fallback(obligations, vectors)
    
    def _cluster_obligations_fallback(self, 
                                          obligations: List[Dict[str, Any]], 
                                          vectors: np.ndarray,
                                          threshold: float = 0.85) -> Dict[int, List[Dict]]:
        """
        回退方法: 使用 v3 的相似度閾值聚類
        當 HDBSCAN 不可用時使用
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(vectors)
        
        clusters = {}
        visited = set()
        cluster_id = 0
        
        for i in range(len(obligations)):
            if i in visited:
                continue
            
            similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
            
            if len(similar_indices) > 0:
                clusters[cluster_id] = [obligations[j] for j in similar_indices]
                visited.update(similar_indices)
                cluster_id += 1
        
        # 處理未分類項
        unclustered = set(range(len(obligations))) - visited
        for i in unclustered:
            clusters[cluster_id] = [obligations[i]]
            cluster_id += 1
        
        print(f"    ✓ 回退聚類完成: {len(clusters)} 個群組")
        return clusters
    
    def _save_noise_points(self, noise_points: List[Dict[str, Any]]):
        """儲存 HDBSCAN 識別的噪聲點供人工審核"""
        noise_file = os.path.join(self.review_queue_dir, "noise_points.json")
        
        noise_data = {
            "generated_at": datetime.now().isoformat(),
            "total_noise_points": len(noise_points),
            "description": "這些義務被 HDBSCAN 識別為噪聲點,通常是因為語意過於獨特或包含無意義的法律引用",
            "review_instructions": "請審核這些項目,判斷是否需要補充語境或重新分類",
            "noise_points": [
                {
                    "text": np["text"],
                    "context": np["context"],
                    "has_anaphora": np["has_anaphora"],
                    "source": np["source"]
                }
                for np in noise_points
            ]
        }
        
        with open(noise_file, 'w', encoding='utf-8') as f:
            json.dump(noise_data, f, ensure_ascii=False, indent=2)
        
        print(f"    → 噪聲點已儲存至: {noise_file}")
    
    # ========================================================================
    # v4 核心改進 4: 統一優先級邏輯
    # ========================================================================
    
    def _calculate_unified_priority(self, 
                                    cluster_info: Dict,
                                    classification_result: Optional[ClassificationResult] = None) -> ReviewPriority:
        """
        v4 核心方法: 統一的優先級計算邏輯
        解決「問題四」- HITL 內部矛盾
        
        這個方法整合了聚類品質、分類信心度、主動學習分數等多個維度
        """
        # 計算主動學習分數
        if classification_result:
            al_score = self._calculate_active_learning_score(
                cluster_info=cluster_info,
                classification_result=classification_result
            )
        else:
            # 僅基於聚類品質計算
            al_score = self._calculate_active_learning_score_from_cluster(cluster_info)
        
        total_priority = al_score.total_priority
        
        # 額外的規則檢查
        quality_metrics = cluster_info.get('quality_metrics', {})
        is_singleton = quality_metrics.get('is_singleton', False)
        is_noise = quality_metrics.get('is_noise', False)
        member_count = cluster_info.get('member_count', 1)
        confidence = cluster_info.get('overall_confidence', 1.0)
        
        # === 統一的優先級決策邏輯 ===
        
        # 規則1: 噪聲點 -> 自動降低優先級(除非信心度極低)
        if is_noise:
            if confidence < 0.5:
                return ReviewPriority.HIGH
            else:
                return ReviewPriority.LOW
        
        # 規則2: 單例聚類 + 低信心度 -> CRITICAL
        if is_singleton and confidence < 0.6:
            return ReviewPriority.CRITICAL
        
        # 規則3: 高影響範圍(成員多) + 低信心度 -> CRITICAL
        if member_count >= 10 and confidence < 0.7:
            return ReviewPriority.CRITICAL
        
        # 規則4: 基於主動學習總分
        if total_priority >= 0.75:
            return ReviewPriority.CRITICAL
        elif total_priority >= 0.55:
            return ReviewPriority.HIGH
        elif total_priority >= 0.35:
            return ReviewPriority.MEDIUM
        else:
            return ReviewPriority.LOW
    
    def _calculate_active_learning_score(self,
                                         cluster_info: Dict,
                                         classification_result: ClassificationResult) -> ActiveLearningScore:
        """計算完整的主動學習分數(含分類結果)"""
        confidence = classification_result.confidence
        uncertainty_score = 1.0 - confidence
        
        alternatives = classification_result.evidence.alternative_classifications
        if alternatives and len(alternatives) > 0:
            top_alt_conf = alternatives[0].get('confidence', 0)
            if abs(confidence - top_alt_conf) < 0.15:
                uncertainty_score += 0.2
        
        uncertainty_score = min(uncertainty_score, 1.0)
        
        member_count = cluster_info.get('member_count', 1)
        impact_score = min(member_count / 50.0, 1.0)
        frequency_score = min(member_count / 30.0, 1.0)
        
        quality_metrics = cluster_info.get('quality_metrics', {})
        is_singleton = quality_metrics.get('is_singleton', False)
        avg_similarity = quality_metrics.get('avg_intra_similarity', 1.0)
        min_similarity = quality_metrics.get('min_member_similarity', 1.0)
        
        complexity_score = 0.0
        if is_singleton:
            complexity_score += 0.5
        
        complexity_score += (1.0 - avg_similarity) * 0.3
        complexity_score += (1.0 - min_similarity) * 0.2
        complexity_score = min(complexity_score, 1.0)
        
        total_priority = (
            uncertainty_score * self.PRIORITY_WEIGHTS['uncertainty'] +
            impact_score * self.PRIORITY_WEIGHTS['impact'] +
            frequency_score * self.PRIORITY_WEIGHTS['frequency'] +
            complexity_score * self.PRIORITY_WEIGHTS['complexity']
        )
        
        return ActiveLearningScore(
            uncertainty_score=uncertainty_score,
            impact_score=impact_score,
            frequency_score=frequency_score,
            complexity_score=complexity_score,
            total_priority=total_priority
        )
    
    def _calculate_active_learning_score_from_cluster(self, cluster_info: Dict) -> ActiveLearningScore:
        """僅基於聚類品質計算主動學習分數(無分類結果時使用)"""
        quality_metrics = cluster_info.get('quality_metrics', {})
        avg_similarity = quality_metrics.get('avg_intra_similarity', 1.0)
        
        # 使用平均相似度作為代理信心度
        uncertainty_score = 1.0 - avg_similarity
        
        member_count = cluster_info.get('member_count', 1)
        impact_score = min(member_count / 50.0, 1.0)
        frequency_score = min(member_count / 30.0, 1.0)
        
        is_singleton = quality_metrics.get('is_singleton', False)
        min_similarity = quality_metrics.get('min_member_similarity', 1.0)
        
        complexity_score = 0.0
        if is_singleton:
            complexity_score += 0.5
        complexity_score += (1.0 - avg_similarity) * 0.3
        complexity_score += (1.0 - min_similarity) * 0.2
        complexity_score = min(complexity_score, 1.0)
        
        total_priority = (
            uncertainty_score * self.PRIORITY_WEIGHTS['uncertainty'] +
            impact_score * self.PRIORITY_WEIGHTS['impact'] +
            frequency_score * self.PRIORITY_WEIGHTS['frequency'] +
            complexity_score * self.PRIORITY_WEIGHTS['complexity']
        )
        
        return ActiveLearningScore(
            uncertainty_score=uncertainty_score,
            impact_score=impact_score,
            frequency_score=frequency_score,
            complexity_score=complexity_score,
            total_priority=total_priority
        )
    
    # ========================================================================
    # 階段一: 資料載入與預處理 (繼承 v3)
    # ========================================================================
    
    def _initialize_base_ontology(self):
        """初始化基礎本體結構(職業安全衛生領域知識)"""
        
        # 完整的主體本體(Subject Ontology)
        self.base_subject_ontology = {
            "雇主": {
                "standard_name": "EMPLOYER",
                "parent_category": "LegalEntity",
                "level": 2,
                "hierarchy_path": "Subject -> LegalEntity -> Employer",
                "synonyms": ["事業單位", "事業主", "業主", "公司"],
                "description": "負有職業安全衛生法律義務的事業經營主體"
            },
            "勞工": {
                "standard_name": "WORKER",
                "parent_category": "Person",
                "level": 2,
                "hierarchy_path": "Subject -> Person -> Worker",
                "synonyms": ["工作者", "員工", "從業人員", "作業人員"],
                "description": "受僱於雇主從事工作獲致工資者"
            },
            "承攬人": {
                "standard_name": "CONTRACTOR",
                "parent_category": "LegalEntity",
                "level": 2,
                "hierarchy_path": "Subject -> LegalEntity -> Contractor",
                "synonyms": ["承包商", "承攬廠商", "外包商"],
                "description": "承攬事業單位工作之事業單位"
            },
            "代行檢查機構": {
                "standard_name": "INSPECTION_AGENCY",
                "parent_category": "Organization",
                "level": 2,
                "hierarchy_path": "Subject -> Organization -> InspectionAgency",
                "synonyms": ["檢查機構", "檢驗機構", "代檢機構"],
                "description": "經中央主管機關認可代行檢查業務之機構"
            },
            "職業安全衛生管理人員": {
                "standard_name": "OSH_PERSONNEL",
                "parent_category": "Person",
                "level": 3,
                "hierarchy_path": "Subject -> Person -> Professional -> OSHPersonnel",
                "synonyms": ["安全衛生人員", "職安人員", "安全管理員"],
                "description": "從事職業安全衛生管理工作之專業人員"
            }
        }
        
        # 完整的客體本體(Object Ontology) - 分層結構
        self.base_object_ontology = {
            # 機械設備類
            "起重機": {
                "standard_name": "CRANE",
                "parent_category": "LiftingEquipment",
                "level": 3,
                "hierarchy_path": "Object -> Equipment -> LiftingEquipment -> Crane",
                "synonyms": ["吊車", "起重設備"],
                "description": "用於吊升及搬運重物之機械設備"
            },
            "升降機": {
                "standard_name": "ELEVATOR",
                "parent_category": "LiftingEquipment",
                "level": 3,
                "hierarchy_path": "Object -> Equipment -> LiftingEquipment -> Elevator",
                "synonyms": ["電梯", "昇降設備"],
                "description": "用於載運人員或貨物於不同樓層間之設備"
            },
            "衝壓機械": {
                "standard_name": "PRESS_MACHINE",
                "parent_category": "ProcessingEquipment",
                "level": 3,
                "hierarchy_path": "Object -> Equipment -> ProcessingEquipment -> PressMachine",
                "synonyms": ["沖床", "沖壓床", "衝剪機"],
                "description": "利用壓力進行金屬或其他材料加工之機械"
            },
            
            # 化學物質類
            "危害性化學品": {
                "standard_name": "HAZARDOUS_CHEMICAL",
                "parent_category": "ChemicalSubstance",
                "level": 2,
                "hierarchy_path": "Object -> Substance -> ChemicalSubstance -> HazardousChemical",
                "synonyms": ["有害化學物質", "危險化學品"],
                "description": "具有危害性之化學品"
            },
            "有機溶劑": {
                "standard_name": "ORGANIC_SOLVENT",
                "parent_category": "HazardousChemical",
                "level": 3,
                "hierarchy_path": "Object -> Substance -> ChemicalSubstance -> HazardousChemical -> OrganicSolvent",
                "synonyms": ["溶劑", "有機溶劑類"],
                "description": "能溶解其他物質的有機化合物"
            },
            
            # 防護設備類
            "安全裝置": {
                "standard_name": "SAFETY_DEVICE",
                "parent_category": "SafetyEquipment",
                "level": 2,
                "hierarchy_path": "Object -> Equipment -> SafetyEquipment -> SafetyDevice",
                "synonyms": ["安全設施", "安全設備"],
                "description": "用於預防危害之裝置"
            },
            "護罩": {
                "standard_name": "GUARD",
                "parent_category": "SafetyDevice",
                "level": 3,
                "hierarchy_path": "Object -> Equipment -> SafetyEquipment -> SafetyDevice -> Guard",
                "synonyms": ["防護罩", "安全護罩"],
                "description": "防止人員接觸危險部位之護蓋或遮蔽物"
            },
            "個人防護具": {
                "standard_name": "PPE",
                "parent_category": "SafetyEquipment",
                "level": 2,
                "hierarchy_path": "Object -> Equipment -> SafetyEquipment -> PPE",
                "synonyms": ["防護具", "防護裝備"],
                "description": "個人穿戴使用之防護設備"
            },
            "安全帽": {
                "standard_name": "SAFETY_HELMET",
                "parent_category": "PPE",
                "level": 3,
                "hierarchy_path": "Object -> Equipment -> SafetyEquipment -> PPE -> SafetyHelmet",
                "synonyms": ["工安帽", "防護帽"],
                "description": "保護頭部免受撞擊之帽具"
            },
            "安全帶": {
                "standard_name": "SAFETY_HARNESS",
                "parent_category": "PPE",
                "level": 3,
                "hierarchy_path": "Object -> Equipment -> SafetyEquipment -> PPE -> SafetyHarness",
                "synonyms": ["防墜器", "安全索"],
                "description": "防止高處作業墜落之防護裝備"
            },
            
            # 作業場所與環境類
            "高溫作業場所": {
                "standard_name": "HIGH_TEMP_WORKPLACE",
                "parent_category": "Workplace",
                "level": 2,
                "hierarchy_path": "Object -> Environment -> Workplace -> HighTempWorkplace",
                "synonyms": ["高溫環境", "熱作業場所", "鍋爐房", "鑄造間"],
                "description": "溫度過高之作業環境"
            },
            "密閉空間": {
                "standard_name": "CONFINED_SPACE",
                "parent_category": "Workplace",
                "level": 2,
                "hierarchy_path": "Object -> Environment -> Workplace -> ConfinedSpace",
                "synonyms": ["侷限空間", "局限空間"],
                "description": "通風不良之有限空間"
            }
        }

        print("  → v5 初始化: 為本體實體生成語意向量...")
        self._generate_ontology_embeddings()

    def _generate_ontology_embeddings(self):
        """
        v5 核心方法: 為所有本體實體生成 embeddings
        解決問題三 - 實現語意實體連結的基礎
        
        調用時機: _initialize_base_ontology() 結束時
        """
        all_ontologies = {
            **{f"subject_{k}": v for k, v in self.base_subject_ontology.items()},
            **{f"object_{k}": v for k, v in self.base_object_ontology.items()}
        }
        
        for entity_key, entity_info in all_ontologies.items():
            entity_description = f"""
    實體名稱: {entity_info.get('standard_name', '')}
    同義詞: {', '.join(entity_info.get('synonyms', []))}
    描述: {entity_info.get('description', '')}
    層級路徑: {entity_info.get('hierarchy_path', '')}
            """.strip()
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[entity_description]
                )
                embedding = response.data[0].embedding
                self.ontology_embeddings[entity_key] = np.array(embedding)
                
            except Exception as e:
                print(f"    ✗ 本體實體 {entity_key} embedding 生成失敗: {e}")
                self.ontology_embeddings[entity_key] = np.zeros(1536)
        
        print(f"  ✓ 已生成 {len(self.ontology_embeddings)} 個本體實體向量")

    def _find_ontology_node_semantic(self, entity_text: str, threshold: float = 0.6) -> Optional[str]:
        """
        v5 核心方法: 使用語意向量匹配本體節點
        解決問題三 - "可燃性氣體" → HAZARDOUS_CHEMICAL
        
        調用時機: _build_event_layer() 處理 event.patients 時
        取代: v4 的 _find_ontology_node_for_entity() 字串匹配
        
        Args:
            entity_text: 實體文本 (如 "可燃性氣體")
            threshold: 最低相似度閾值 (預設 0.6)
            
        Returns:
            最匹配的本體節點 ID (如 "object_HAZARDOUS_CHEMICAL"),或 None
        """
        if not entity_text or not self.ontology_embeddings:
            return None
        
        try:
            # 生成實體文本的 embedding
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=[entity_text]
            )
            entity_vector = np.array(response.data[0].embedding)
            
            # 計算與所有本體實體的餘弦相似度
            best_match = None
            best_similarity = threshold
            
            for ontology_key, ontology_vector in self.ontology_embeddings.items():
                cos_sim = np.dot(entity_vector, ontology_vector) / (
                    np.linalg.norm(entity_vector) * np.linalg.norm(ontology_vector)
                )
                
                if cos_sim > best_similarity:
                    best_similarity = cos_sim
                    best_match = ontology_key
            
            if best_match:
                # 轉換為圖譜節點 ID
                if best_match.startswith("subject_"):
                    entity_name = best_match[8:]
                    if entity_name in self.base_subject_ontology:
                        standard_name = self.base_subject_ontology[entity_name]['standard_name']
                        return f"subject_{standard_name}"
                elif best_match.startswith("object_"):
                    entity_name = best_match[7:]
                    if entity_name in self.base_object_ontology:
                        standard_name = self.base_object_ontology[entity_name]['standard_name']
                        return f"object_{standard_name}"
            
            return None
            
        except Exception as e:
            print(f"    ✗ 語意實體連結失敗 ({entity_text}): {e}")
            return None
    
    def load_documents(self):
        """載入法律文件"""
        print("📚 載入法律文件...")
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"  ✓ 已載入 {len(self.documents)} 個法律條文片段")
        except FileNotFoundError:
            print(f"  ✗ 錯誤:找不到檔案 {self.input_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"  ✗ 錯誤:JSON 解析失敗 - {e}")
            sys.exit(1)
    
    # ========================================================================
    # 階段二: 事件抽取 (繼承 v3 的 Tool Calling 方法)
    # ========================================================================
    
    def extract_legal_events(self):
        """從法律文本中抽取結構化事件"""
        print("\n🎯 抽取法律事件結構...")
        
        for doc in self.documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            article_id = metadata.get('article', 'unknown')
            
            if not content or len(content) < 20:
                continue
            
            event = self._extract_event_with_llm(content, article_id)
            
            if event:
                self.legal_events[event.event_id] = event
                
                all_entities = (
                    event.patients + 
                    event.instruments + 
                    event.locations
                )
                self.discovered_entities.update(all_entities)
        
        print(f"  ✓ 抽取了 {len(self.legal_events)} 個法律事件")
        print(f"  ✓ 自動發現 {len(self.discovered_entities)} 個新實體")
        
        self._save_legal_events()
        self._incremental_ontology_expansion()
    
    def _extract_event_with_llm(self, text: str, article_id: str) -> Optional[LegalEvent]:
        """使用LLM抽取單一法律事件 (v3的成功實現)"""
        
        def _ensure_string(value: Any) -> str:
            """確保返回非 None 的字符串"""
            if value is None:
                return ''
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, list) and len(value) > 0:
                for item in value:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
            return ''
        
        def _ensure_string_list(value: Any) -> List[str]:
            """確保返回字符串列表"""
            if not value:
                return []
            
            if isinstance(value, str):
                return [value.strip()] if value.strip() else []
            
            if isinstance(value, list):
                result = []
                for item in value:
                    if isinstance(item, list):
                        result.extend(_ensure_string_list(item))
                    elif item is not None:
                        cleaned = str(item).strip()
                        if cleaned:
                            result.append(cleaned)
                return result
            
            cleaned_val = str(value).strip()
            return [cleaned_val] if cleaned_val else []
        
        tool_schema = {
            "type": "function",
            "function": {
                "name": "record_legal_event",
                "description": "記錄從法律文本中抽取的結構化事件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "核心動作 (如: 供應、設置、檢查、使用)"
                        },
                        "actor": {
                            "type": "string",
                            "description": "主體 (如: 雇主、勞工、事業單位)"
                        },
                        "patients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "受事客體列表 (如: 飲用水、食鹽、護欄、設備)"
                        },
                        "instruments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "工具/手段列表 (如: 安全帽、通風設備)"
                        },
                        "locations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "場所列表 (如: 高溫作業場所、密閉空間)"
                        },
                        "conditions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "條件列表 (如: 溫度超過30度、作業時間超過1小時)"
                        },
                        "temporal": {
                            "type": "string",
                            "description": "時間條件 (如: 每日、定期、作業前)"
                        },
                        "purpose": {
                            "type": "string",
                            "description": "目的 (如: 防止中暑、預防墜落)"
                        }
                    },
                    "required": ["action", "actor", "patients", "instruments", "locations"]
                }
            }
        }

        prompt = f"""你是職業安全衛生法律事件抽取專家。請從以下法條文本中抽取結構化的法律事件。

法條文本:
{text}

你必須嚴格遵循以下【兩階段任務】：

【任務 1：語意角色標註 (SRL)】
請先找出法條中的核心法律事件：
- 動作 (action): 核心動詞 (如: 設置、檢查、供應)。
- 主體 (actor): 動作的執行者 (如: 雇主、勞工)。
- 受事客體 (patients): 動作的「直接對象」或「承受者」 (如: "設置" 的 "防護具")。
- 工具/手段 (instruments): 用來「完成動作」的工具 (如: "使用" "絕熱材料" "被覆" 容器)。

【任務 2：約束條件萃取 (Constraint Extraction)】
在完成任務 1 之後，你【必須】回頭檢視法條，找出與該事件相關的所有「約束條件」：
- 條件 (conditions): 執行動作的必要條件 (如: 溫度超過30度、高度在二公尺以上)。
- 時間 (temporal): 動作發生的時間限制 (如: 每日、定期、作業前)。
- 目的 (purpose): 執行動作的法律目的 (如: 防止中暑、預防墜落)。

【重要指令】
1. 【不可遺漏】: 執行任務 2 與執行任務 1 同等重要。即使 `conditions` 或 `temporal` 很長，也必須完整萃取。
2. 【區分角色】: 嚴格區分 `patients` (被作用的對象) 和 `instruments` (用來作用的工具)。
3. 【欄位為空】: 如果某個欄位 (例如 `purpose`) 在文本中確實不存在，請使用空列表 `[]` 或空字串 `""`。

請仔細分析文本並調用 'record_legal_event' 工具來記錄你的發現。"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "你是職業安全法律事件抽取專家,精通語意角色標註(Semantic Role Labeling)。你必須使用 'record_legal_event' 工具來提交你的分析結果。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": "record_legal_event"}}
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls or len(message.tool_calls) == 0:
                print(f"    ⚠️ 事件抽取未調用工具 ({article_id})")
                return None
                
            tool_call_args = message.tool_calls[0].function.arguments
            
            try:
                parsed = json.loads(tool_call_args)
            except json.JSONDecodeError as e:
                print(f"    ✗ 嚴重: Tool Call 參數解析失敗 ({article_id}): {e}")
                return None

            if not isinstance(parsed, dict):
                print(f"    ⚠️ 事件抽取返回非字典格式 ({article_id})")
                return None
            
            action = _ensure_string(parsed.get('action'))
            actor = _ensure_string(parsed.get('actor'))
            
            if not action and not actor:
                print(f"    ⚠️ 事件抽取無有效內容 (但JSON有效) ({article_id})")
                return None
            
            patients = _ensure_string_list(parsed.get('patients'))
            instruments = _ensure_string_list(parsed.get('instruments'))
            locations = _ensure_string_list(parsed.get('locations'))
            conditions = _ensure_string_list(parsed.get('conditions'))
            
            temporal = _ensure_string(parsed.get('temporal')) or None
            purpose = _ensure_string(parsed.get('purpose')) or None
            
            filled_fields = sum([
                bool(action),
                bool(actor),
                len(patients) > 0,
                len(instruments) > 0,
                len(locations) > 0,
                len(conditions) > 0,
                bool(temporal),
                bool(purpose)
            ])
            confidence = min(0.95, 0.5 + (filled_fields / 8) * 0.45)
            
            event = LegalEvent(
                event_id=f"event_{article_id}_{len(self.legal_events)}",
                action=action,
                actor=actor,
                patients=patients,
                instruments=instruments,
                locations=locations,
                conditions=conditions,
                temporal=temporal,
                purpose=purpose,
                source_article=article_id,
                confidence=confidence
            )
            
            return event
            
        except Exception as e:
            print(f"    ✗ 事件抽取 API 調用失敗 ({article_id}): {e}")
            return None
    
    def _save_legal_events(self):
        """儲存法律事件"""
        output_path = os.path.join(self.output_dir, "legal_events.json")
        
        events_data = {
            event_id: asdict(event) 
            for event_id, event in self.legal_events.items()
        }
        
        serializable_events = self._make_json_serializable(events_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_events, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 法律事件已儲存至 {output_path}")
    
    def _incremental_ontology_expansion(self):
        """增量式本體擴充"""
        print("\n📚 增量式本體擴充...")
        
        if not self.discovered_entities:
            print("  ✓ 無新實體需要擴充")
            return
        
        existing_entities = set(self.object_ontology.keys())
        new_entities = self.discovered_entities - existing_entities
        
        if not new_entities:
            print("  ✓ 所有發現的實體已存在於本體中")
            return
        
        print(f"  → 發現 {len(new_entities)} 個新實體,開始智能分類...")
        
        expanded_ontology = self._classify_new_entities_to_ontology(
            new_entities, 
            self.object_ontology
        )
        
        self.object_ontology.update(expanded_ontology)
        
        print(f"  ✓ 本體已擴充 {len(expanded_ontology)} 個新節點")
        
        self._save_ontology()
    
    def _classify_new_entities_to_ontology(self, 
                                          new_entities: Set[str], 
                                          existing_ontology: Dict) -> Dict:
        """使用LLM將新實體分類到現有本體層級"""
        
        ontology_summary = {}
        for entity_name, entity_info in existing_ontology.items():
            category = entity_info.get('parent_category', 'Unknown')
            if category not in ontology_summary:
                ontology_summary[category] = []
            ontology_summary[category].append(entity_name)
        
        prompt = f"""你是職業安全本體論專家。現有本體結構如下:

{json.dumps(ontology_summary, ensure_ascii=False, indent=2)}

新發現的實體: {list(new_entities)}

請將這些新實體分類到最合適的父類別下,並提供完整的本體節點定義。

輸出格式:
{{
  "實體名稱": {{
    "standard_name": "標準化名稱(英文大寫加底線)",
    "parent_category": "最合適的父類別(必須從上述現有類別中選擇)",
    "level": 層級數字,
    "hierarchy_path": "完整路徑",
    "synonyms": ["同義詞列表"],
    "description": "簡短描述"
  }}
}}

只輸出JSON。"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是職業安全本體論專家,精通知識組織與分類。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            parsed = self._safe_parse_json_from_llm(result_text)
            
            if isinstance(parsed, dict):
                return parsed
            else:
                return {}
                
        except Exception as e:
            print(f"    ✗ 新實體分類失敗: {e}")
            return {}
    
    # ========================================================================
    # 階段三: 義務正規化 (v4 完全重寫)
    # ========================================================================
    
    def normalize_obligations(self):
        """
        正規化義務節點 (v5 版本: 語境感知 + HDBSCAN 聚類)
        
        這是 _save_obligation_clusters 的主要呼叫者 (Caller)。
        """
        print("\n📄 正規化義務節點 (v5 智慧語意聚類)...")
        
        # === v4 核心改進: 使用語境感知萃取 ===
        obligations_with_context = self._extract_obligations_with_context()
        
        if not obligations_with_context:
            print("  ⚠️ 警告:未萃取到任何義務描述")
            self.obligation_clusters = {}
            return
        
        print(f"  ✓ 萃取了 {len(obligations_with_context)} 個義務描述")
        
        # 使用「含語境的完整文本」生成向量
        obligation_texts = [obl['full_text_with_context'] for obl in obligations_with_context]
        
        print("  → 生成語意向量...")
        obligation_vectors = self._get_embeddings(obligation_texts)
        
        if obligation_vectors.size == 0:
            print("  ✗ 錯誤:向量生成失敗,跳過義務正規化階段")
            self.obligation_clusters = {}
            return
        
        # === v4 核心改進: 使用 HDBSCAN 聚類 ===
        # clusters = self._cluster_obligations_with_hdbscan(obligations_with_context, obligation_vectors)
        
        # ⚠️ v5 修改: 使用葉聚類
        clusters = self._cluster_obligations_with_hdbscan(obligations_with_context, obligation_vectors)
        
        # === 關鍵防呆機制 (v5.1 修正) ===
        if not clusters:
            print("  ✗ 錯誤: HDBSCAN 未能產生任何有效聚類 (聚類數量為 0)。")
            print("    → 這可能是因為輸入的法規文件過少或 'min_cluster_size' 參數(3)過高。")
            print("    → obligation_clusters.json 將為空，後續流程 (control_type_mapping) 將被跳過。")
            self.obligation_clusters = {}
            return # 提前終止，防止後續流程出錯
        
        # 品質評估 (繼承 v4,函式名改為 v5)
        print("  → 評估聚類品質...")
        clusters_with_quality = self._evaluate_cluster_quality_v5(
            clusters, 
            obligations_with_context, 
            obligation_vectors
        )
        
        # ⚠️ v5 修改: 使用提示詞鏈命名
        print("  → 為聚類命名 (v5 提示詞鏈)...")
        self.obligation_clusters = self._name_clusters_with_llm_v5(
            clusters_with_quality,
            obligation_vectors
        )
        
        # 儲存 (繼承 v4)
        self._save_obligation_clusters()
        
        # 將問題聚類加入審核佇列 (繼承 v4)
        self._queue_problematic_clusters_v5()

    def _save_obligation_clusters(self):
        """
        (v5 實作) 儲存義務聚類結果
        
        這個函式會產生 obligation_clusters.json。
        它繼承自 v4, 確保 v4 聚類後的複雜 dict 結構被正確序列化。
        """
        if not self.obligation_clusters:
            print(f"  ⚠️ 警告:沒有義務聚類資料可儲存")
            return
        
        output_path = os.path.join(self.output_dir, "obligation_clusters.json")
        
        save_data = {}
        for cluster_id, cluster_info in self.obligation_clusters.items():
            
            # v4 的 self.obligation_clusters[id]['members'] 是一個 dict 列表
            # 我們在儲存時,只儲存原始文本 'text',以簡化 JSON 檔案
            members = cluster_info.get('members', [])
            if members and isinstance(members[0], dict):
                # 只保存文本,不保存完整的 dict
                members_text = [m.get('text', str(m)) for m in members]
            else:
                members_text = [str(m) for m in members]
            
            cluster_data = {
                'standard_name': cluster_info.get('standard_name'),
                'standard_code': cluster_info.get('standard_code'),
                'category': cluster_info.get('category'),
                'description': cluster_info.get('description'),
                'member_count': cluster_info.get('member_count'),
                'sample': cluster_info.get('sample', []), # sample 已經是純文本
                'quality_metrics': cluster_info.get('quality_metrics'),
                'evidence': cluster_info.get('evidence'),
                'overall_confidence': cluster_info.get('overall_confidence'),
                'review_status': cluster_info.get('review_status'),
                'review_priority': cluster_info.get('review_priority'),
                # 為了檔案大小,我們只儲存最多 50 個成員
                'members': members_text[:50] if len(members_text) > 50 else members_text
            }
            save_data[str(cluster_id)] = cluster_data
        
        # 使用 v4 的 _make_json_serializable 確保 Enum, numpy 等類型可被儲存
        serializable_data = self._make_json_serializable(save_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 義務聚類已儲存至 {output_path}")
        print(f"    → 總共 {len(save_data)} 個聚類")
    
    def _evaluate_cluster_quality_v5(self, 
                                     clusters: Dict[int, List[Dict]], 
                                     obligations: List[Dict],
                                     vectors: np.ndarray) -> Dict[int, Dict]:
        """
        v4 版本: 評估聚類品質
        與 v3 的差異: 考慮語境信息和 HDBSCAN 特性
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        clusters_with_quality = {}
        
        # 建立索引映射
        text_to_index = {
            obl['full_text_with_context']: idx 
            for idx, obl in enumerate(obligations)
        }
        
        for cluster_id, members in clusters.items():
            member_count = len(members)
            is_singleton = (member_count == 1)
            
            # 獲取成員在原始列表中的索引
            member_indices = []
            for member in members:
                key = member['full_text_with_context']
                if key in text_to_index:
                    member_indices.append(text_to_index[key])
            
            if len(member_indices) < 2:
                quality = ClusterQualityMetrics(
                    silhouette_score=0.0,
                    avg_intra_similarity=1.0,
                    min_member_similarity=1.0,
                    is_singleton=True,
                    is_noise=False,
                    needs_review=True,
                    review_reason="單例聚類:無法找到語意相似的義務",
                    review_priority=ReviewPriority.MEDIUM
                )
            else:
                member_vectors = vectors[member_indices]
                sim_matrix = cosine_similarity(member_vectors)
                
                n = len(member_indices)
                similarities = []
                for i in range(n):
                    for j in range(i+1, n):
                        similarities.append(sim_matrix[i][j])
                
                avg_sim = np.mean(similarities) if similarities else 0.0
                min_sim = np.min(similarities) if similarities else 0.0
                
                needs_review = False
                review_reason = ""
                review_priority = ReviewPriority.LOW
                
                # v4 增強的品質判斷邏輯
                if member_count < self.MIN_CLUSTER_SIZE:
                    needs_review = True
                    review_reason = f"聚類過小:僅 {member_count} 個成員"
                    review_priority = ReviewPriority.MEDIUM
                elif min_sim < 0.65:  # v4: 降低閾值,因為我們有語境信息
                    needs_review = True
                    review_reason = f"內部異質性過高:最小相似度僅 {min_sim:.2f}"
                    review_priority = ReviewPriority.HIGH
                elif avg_sim < 0.75:  # v4: 降低閾值
                    needs_review = True
                    review_reason = f"平均相似度偏低: {avg_sim:.2f}"
                    review_priority = ReviewPriority.MEDIUM
                
                # v4 新增: 檢查是否有過多的 anaphora
                anaphora_count = sum(1 for m in members if m.get('has_anaphora', False))
                if anaphora_count > member_count * 0.5:
                    needs_review = True
                    review_reason += f" | 過多法律引用({anaphora_count}/{member_count})"
                    review_priority = max(review_priority, ReviewPriority.HIGH, key=lambda x: x.value)
                
                quality = ClusterQualityMetrics(
                    silhouette_score=0.0,
                    avg_intra_similarity=avg_sim,
                    min_member_similarity=min_sim,
                    is_singleton=False,
                    is_noise=False,
                    needs_review=needs_review,
                    review_reason=review_reason,
                    review_priority=review_priority
                )
            
            clusters_with_quality[cluster_id] = {
                'members': members,
                'quality': quality
            }
            
            if quality.needs_review:
                self.problematic_clusters.append({
                    'cluster_id': cluster_id,
                    'member_count': member_count,
                    'quality': asdict(quality),
                    'sample_members': [m['text'] for m in members[:5]]
                })
        
        review_count = sum(1 for c in clusters_with_quality.values() if c['quality'].needs_review)
        print(f"    → 品質評估完成: {review_count}/{len(clusters)} 個聚類需要審核")
        
        return clusters_with_quality
    
    def _name_clusters_with_llm_v5(self, 
                               clusters_with_quality: Dict[int, Dict],
                               obligation_vectors: np.ndarray) -> Dict:
        """
        v5 核心改進: 提示詞鏈
        解決問題二 - 強制 LLM 使用語意匹配的 rule_id
        
        關鍵修改:
        1. 將語意匹配結果注入 prompt
        2. 驗證 LLM 是否使用了注入的結果
        3. 強制修正錯誤的 rule_id
        """
        named_clusters = {}
        total_clusters = len(clusters_with_quality)

        print(f"    → 開始為 {total_clusters} 個聚類命名 (v5 提示詞鏈)...")
        
        for idx, (cluster_id, cluster_data) in enumerate(clusters_with_quality.items(), 1):
            cluster_id_int = int(cluster_id)
            members = cluster_data['members']
            quality = cluster_data['quality']
            
            if not isinstance(members, list):
                named_clusters[cluster_id_int] = self._create_fallback_cluster(cluster_id_int, members, quality)
                continue

            if idx % 10 == 0 or idx == total_clusters:
                print(f"    → 進度: {idx}/{total_clusters}")

            sample = [m['text'] for m in members[:10]]
            sample_str = '\n'.join(f"{i+1}. {o}" for i, o in enumerate(sample))

            # 語意規則匹配
            cluster_texts = [m['full_text_with_context'] for m in members]
            try:
                cluster_center_response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[' '.join(cluster_texts[:5])]
                )
                cluster_center_vector = np.array(cluster_center_response.data[0].embedding)
                
                matched_rules = self._match_rules_with_semantic_similarity(
                    obligation_text=sample_str,
                    obligation_vector=cluster_center_vector,
                    top_k=3,
                    threshold=0.5
                )
            except Exception as e:
                print(f"    ✗ 聚類 {cluster_id} 規則匹配失敗: {e}")
                matched_rules = []

            related_events = self._find_related_events([m['text'] for m in members])
            event_summary = ""
            if related_events:
                event_summary = f"\n\n相關法律事件:\n"
                for event in related_events[:3]:
                    event_summary += f"- 動作:{event.action}, 主體:{event.actor}, 客體:{event.patients}\n"
            
            # ⚠️ v5 關鍵: 提示詞鏈 - 注入匹配結果
            rules_injection = ""
            if matched_rules:
                best_rule_id, best_similarity = matched_rules[0]
                best_rule = self.rule_base[best_rule_id]
                rules_injection = f"""
    語意規則匹配結果 (系統已完成分析):
    經系統語意向量分析,本聚類與 '{best_rule_id}: {best_rule.rule_name}' 語意最為相關 (相似度: {best_similarity:.3f})。

    Top 3 匹配規則:
    """
                for rule_id, similarity in matched_rules:
                    rule = self.rule_base[rule_id]
                    rules_injection += f"  - {rule_id}: {rule.rule_name} (相似度: {similarity:.3f}, 類別: {rule.category})\n"
                
                # ⚠️ v5 關鍵: 明確指示 LLM 使用這些值
                rules_injection += f"""
    ⚠️ 重要指示:
    - 請在 evidence.decision_rule_id 中使用: '{best_rule_id}'
    - 請在 evidence.decision_rule_name 中使用: '{best_rule.rule_name}'
    - 請在 evidence.rule_similarity_score 中使用: {best_similarity:.3f}
    """
            STANDARD_ACTIONS_LIST = [
                "DEFINE",           # 定義、稱
                "INSTALL",          # 設置、裝設、配置、設立
                "INSPECT",          # 檢查、檢點、巡視、監測、測定
                "MAINTAIN",         # 維護、修理、保養、補修、汰換
                "PROVIDE",          # 供應、置備、提供
                "EDUCATE",          # 教育、訓練、指派、選任
                "PROHIBIT",         # 禁止、不得
                "OPERATE",          # 操作、使用
                "DOCUMENT",         # 訂定、記錄、報告、計畫
                "LABEL"             # 標示、警告
            ]

            prompt = f"""請為以下職業安全衛生法律義務命名、分類，並【正規化其核心動作】。

    義務範例(共 {len(members)} 條,顯示前 {len(sample)} 條):
    {sample_str}
    {event_summary}
    {rules_injection}

    聚類品質資訊:
    - 成員數量: {len(members)}
    - 平均相似度: {quality.avg_intra_similarity:.2f}

    請輸出 JSON:
    {{
    "standard_name": "標準化名稱",
    "standard_code": "標準代碼",
    "category": "義務類別",
    "standard_action": "請從【標準動作庫】中選擇一個最能代表此義務核心動作的標準詞",
    "description": "簡短描述",
    "evidence": {{
        "keywords_matched": ["關鍵詞"],
        "decision_rule_id": "{matched_rules[0][0] if matched_rules else ''}",
        "decision_rule_name": "{self.rule_base[matched_rules[0][0]].rule_name if matched_rules else ''}",
        "rule_similarity_score": {matched_rules[0][1] if matched_rules else 0.0},
        "confidence_factors": {{"semantic_coherence": 0.8, "keyword_coverage": 0.7, "domain_specificity": 0.9}},
        "text_snippets": ["代表性文本"]
    }}
    }}

    只輸出JSON。"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "你是職業安全衛生法律專家。你必須使用系統提供的語意匹配結果。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=600
                )

                result_text = response.choices[0].message.content.strip()
                parsed = self._safe_parse_json_from_llm(result_text)

                if not isinstance(parsed, dict) or 'evidence' not in parsed:
                    raise ValueError("LLM 回傳缺少 evidence 結構")

                evidence_data = parsed.get('evidence', {})
                
                # ⚠️ v5 關鍵: 驗證並強制修正
                if matched_rules:
                    expected_rule_id = matched_rules[0][0]
                    actual_rule_id = evidence_data.get('decision_rule_id', '')
                    
                    if actual_rule_id != expected_rule_id:
                        print(f"    ⚠️ 聚類 {cluster_id}: LLM 未使用注入的 rule_id,已自動修正")
                        evidence_data['decision_rule_id'] = expected_rule_id
                        evidence_data['decision_rule_name'] = self.rule_base[expected_rule_id].rule_name
                        evidence_data['rule_similarity_score'] = matched_rules[0][1]
                
                evidence = StructuredEvidence(
                    keywords_matched=evidence_data.get('keywords_matched', []),
                    decision_rule_id=evidence_data.get('decision_rule_id', ''),
                    decision_rule_name=evidence_data.get('decision_rule_name', ''),
                    rule_similarity_score=evidence_data.get('rule_similarity_score', 0.0),
                    confidence_factors=evidence_data.get('confidence_factors', {}),
                    text_snippets=evidence_data.get('text_snippets', []),
                    extracted_events=evidence_data.get('extracted_events', [])
                )
                
                conf_factors = evidence.confidence_factors
                overall_confidence = np.mean([
                    conf_factors.get('semantic_coherence', 0.5),
                    conf_factors.get('keyword_coverage', 0.5),
                    conf_factors.get('domain_specificity', 0.5)
                ])
                
                # 統一優先級邏輯 (繼承 v4)
                temp_classification = ClassificationResult(
                    classification="Unknown",
                    confidence=overall_confidence,
                    evidence=evidence,
                    review_status=ReviewStatus.AUTO_APPROVED
                )
                
                cluster_info = {
                    'member_count': len(members),
                    'quality_metrics': asdict(quality),
                    'overall_confidence': overall_confidence
                }
                
                unified_priority = self._calculate_unified_priority(
                    cluster_info=cluster_info,
                    classification_result=temp_classification
                )
                
                if quality.needs_review or overall_confidence < self.CONFIDENCE_THRESHOLD or unified_priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH]:
                    review_status = ReviewStatus.PENDING_REVIEW
                else:
                    review_status = ReviewStatus.AUTO_APPROVED

                parsed['members'] = members
                parsed['member_count'] = len(members)
                parsed['sample'] = [m['text'] for m in members[:3]]
                parsed['quality_metrics'] = asdict(quality)
                parsed['evidence'] = asdict(evidence)
                parsed['overall_confidence'] = overall_confidence
                parsed['review_status'] = review_status.value
                parsed['review_priority'] = unified_priority.value

                named_clusters[cluster_id_int] = parsed

            except Exception as e:
                print(f"    ✗ 聚類 {cluster_id} 命名失敗: {e}")
                named_clusters[cluster_id_int] = self._create_fallback_cluster(cluster_id_int, members, quality)

        print(f"    ✓ 完成 {len(named_clusters)} 個聚類的命名")
        
        # v5 統計
        rule_match_count = sum(1 for c in named_clusters.values() 
                            if isinstance(c, dict) and c.get('evidence', {}).get('decision_rule_id'))
        print(f"    → v5 改進: {rule_match_count}/{len(named_clusters)} 個聚類成功匹配規則")
        
        return named_clusters
    
    def _find_related_events(self, obligation_texts: List[str]) -> List[LegalEvent]:
        """找出與義務相關的法律事件"""
        related = []
        for event in self.legal_events.values():
            for obl in obligation_texts[:5]:
                keywords = []
                
                if event.action and isinstance(event.action, str):
                    keywords.append(event.action)
                
                if event.actor and isinstance(event.actor, str):
                    keywords.append(event.actor)
                
                if event.patients and isinstance(event.patients, list):
                    for p in event.patients[:2]:
                        if p and isinstance(p, str):
                            keywords.append(p)
                
                if keywords and any(keyword in obl for keyword in keywords):
                    related.append(event)
                    break

        return related[:5]
    
    def _create_fallback_cluster(self, cluster_id: int, members: Any, quality: ClusterQualityMetrics) -> Dict:
        """創建回退聚類"""
        if not isinstance(members, list):
            members = list(members) if hasattr(members, '__iter__') else []
        
        # 確保 members 是原始文本
        member_texts = []
        for m in members:
            if isinstance(m, dict):
                member_texts.append(m.get('text', str(m)))
            else:
                member_texts.append(str(m))
        
        return {
            "standard_name": f"未分類義務群組 {cluster_id}",
            "standard_code": f"OBLIGATION_CLUSTER_{cluster_id}",
            "category": "未分類",
            "description": "自動命名失敗",
            "members": members,
            "member_count": len(members),
            "sample": member_texts[:3],
            "quality_metrics": asdict(quality),
            "evidence": asdict(StructuredEvidence(
                decision_rule_id="",
                decision_rule_name="自動命名失敗,使用預設值"
            )),
            "overall_confidence": 0.0,
            "review_status": ReviewStatus.PENDING_REVIEW.value,
            "review_priority": ReviewPriority.HIGH.value
        }
    
    def _save_obligation_clusters(self):
        """
        (v5 實作) 儲存義務聚類結果
        
        這個函式會產生 obligation_clusters.json。
        它繼承自 v4, 確保 v4 聚類後的複雜 dict 結構被正確序列化。
        """
        if not self.obligation_clusters:
            print(f"  ⚠️ 警告:沒有義務聚類資料可儲存 (self.obligation_clusters 為空)")
            print(f"    → 因此 obligation_clusters.json 將不會被建立。")
            return
        
        output_path = os.path.join(self.output_dir, "obligation_clusters.json")
        
        save_data = {}
        for cluster_id, cluster_info in self.obligation_clusters.items():
            
            # v4 的 self.obligation_clusters[id]['members'] 是一個 dict 列表
            # 我們在儲存時,只儲存原始文本 'text',以簡化 JSON 檔案
            members = cluster_info.get('members', [])
            if members and isinstance(members[0], dict):
                # 只保存文本,不保存完整的 dict
                members_text = [m.get('text', str(m)) for m in members]
            else:
                members_text = [str(m) for m in members]
            
            cluster_data = {
                'standard_name': cluster_info.get('standard_name'),
                'standard_code': cluster_info.get('standard_code'),
                'category': cluster_info.get('category'),
                'description': cluster_info.get('description'),
                'member_count': cluster_info.get('member_count'),
                'sample': cluster_info.get('sample', []), # sample 已經是純文本
                'quality_metrics': cluster_info.get('quality_metrics'),
                'evidence': cluster_info.get('evidence'),
                'overall_confidence': cluster_info.get('overall_confidence'),
                'review_status': cluster_info.get('review_status'),
                'review_priority': cluster_info.get('review_priority'),
                # 為了檔案大小,我們只儲存最多 50 個成員
                'members': members_text[:50] if len(members_text) > 50 else members_text
            }
            save_data[str(cluster_id)] = cluster_data
        
        # 使用 v4 的 _make_json_serializable 確保 Enum, numpy 等類型可被儲存
        serializable_data = self._make_json_serializable(save_data)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 義務聚類已儲存至 {output_path}")
            print(f"    → 總共 {len(save_data)} 個聚類")
        except Exception as e:
            print(f"  ✗ 錯誤: 儲存 obligation_clusters.json 失敗: {e}")
    
    def _queue_problematic_clusters_v5(self):
        """
        v4 版本: 將問題聚類加入審核佇列
        使用統一的優先級邏輯
        """
        if not self.problematic_clusters:
            print("  ✓ 所有聚類品質良好,無需審核")
            return
        
        # === v4 核心改進: 重新計算統一優先級 ===
        for cluster_item in self.problematic_clusters:
            cluster_id = cluster_item['cluster_id']
            if str(cluster_id) in self.obligation_clusters:
                cluster_info = self.obligation_clusters[str(cluster_id)]
                
                # 使用統一優先級計算
                priority = self._calculate_unified_priority(
                    cluster_info=cluster_info,
                    classification_result=None
                )
                
                # 計算主動學習分數(用於詳細記錄)
                al_score = self._calculate_active_learning_score_from_cluster(cluster_info)
                
                cluster_item['active_learning_score'] = asdict(al_score)
                cluster_item['review_priority'] = priority.value
        
        # 按優先級排序
        priority_order = {
            ReviewPriority.CRITICAL.value: 0,
            ReviewPriority.HIGH.value: 1,
            ReviewPriority.MEDIUM.value: 2,
            ReviewPriority.LOW.value: 3
        }
        
        self.problematic_clusters.sort(
            key=lambda x: (
                priority_order.get(x.get('review_priority', 'low'), 999),
                -x.get('active_learning_score', {}).get('total_priority', 0)
            )
        )
        
        queue_file = os.path.join(self.review_queue_dir, "problematic_clusters_queue.json")
        
        queue_data = {
            "generated_at": datetime.now().isoformat(),
            "total_problematic": len(self.problematic_clusters),
            "priority_distribution": self._get_priority_distribution(self.problematic_clusters),
            "review_instructions": "請人工審核以下聚類,考慮合併或拆分。優先處理 CRITICAL 和 HIGH 項目。",
            "clusters": self.problematic_clusters
        }
        
        serializable_queue = self._make_json_serializable(queue_data)
        
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_queue, f, ensure_ascii=False, indent=2)
        
        print(f"  ⚠️ 已將 {len(self.problematic_clusters)} 個問題聚類加入審核佇列")
        print(f"  → 優先級分布: {queue_data['priority_distribution']}")
        print(f"  → 審核佇列檔案: {queue_file}")
    
    def _get_priority_distribution(self, items: List[Dict]) -> Dict[str, int]:
        """統計優先級分布"""
        dist = defaultdict(int)
        for item in items:
            priority = item.get('review_priority', 'low')
            dist[priority] += 1
        return dict(dist)
    
    # ========================================================================
    # 階段四: 風險控制層級分類 (v4 增強版)
    # ========================================================================
    
    def classify_control_types(self):
        """
        分類義務到風險控制層級 (v5 版本)
        
        修改點:
        - 在 prompt 中注入語意匹配結果 (提示詞鏈)
        - 驗證並強制修正 LLM 的 rule_id
        """
        print("\n🎯 分類風險控制層級 (v5 提示詞鏈)...")
        
        if not self.obligation_clusters:
            print("  ⚠️ 無義務聚類資料,跳過控制類型分類")
            return
        
        total = len(self.obligation_clusters)
        
        for idx, (cluster_id, cluster_info) in enumerate(self.obligation_clusters.items(), 1):
            cluster_id = str(cluster_id)
            
            if not isinstance(cluster_info, dict):
                continue
            
            if idx % 10 == 0 or idx == total:
                print(f"  → 進度: {idx}/{total}")
            
            members = cluster_info.get('members', [])
            if not members:
                continue
            
            if isinstance(members[0], dict):
                sample_obligations = [m.get('text', str(m)) for m in members[:5]]
            else:
                sample_obligations = [str(m) for m in members[:5]]
            sample_str = '\n'.join(sample_obligations)
            
            # 語意規則匹配 (繼承 v4)
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[sample_str]
                )
                cluster_vector = np.array(response.data[0].embedding)
                
                matched_rules = self._match_rules_with_semantic_similarity(
                    obligation_text=sample_str,
                    obligation_vector=cluster_vector,
                    top_k=3,
                    threshold=0.5
                )
            except Exception as e:
                print(f"    ✗ 聚類 {cluster_id} 向量生成失敗: {e}")
                matched_rules = []
            
            # ⚠️ v5 關鍵: 提示詞鏈 - 注入匹配結果
            rules_injection = ""
            if matched_rules:
                best_rule_id, best_similarity = matched_rules[0]
                best_rule = self.rule_base[best_rule_id]
                rules_injection = f"""
    語意規則匹配結果 (系統已完成分析):
    經系統語意向量分析,本義務聚類與 '{best_rule_id}: {best_rule.rule_name}' 語意最為相關 (相似度: {best_similarity:.3f})。
    建議控制類型: {best_rule.category}

    Top 3 匹配規則:
    """
                for rule_id, similarity in matched_rules:
                    rule = self.rule_base[rule_id]
                    rules_injection += f"  - {rule_id}: {rule.rule_name} (相似度: {similarity:.3f}, 類別: {rule.category})\n"
                
                # ⚠️ v5 關鍵: 明確指示 LLM
                rules_injection += f"""
    ⚠️ 重要指示:
    - 請在 control_type 中使用: '{best_rule.category}'
    - 請在 evidence.decision_rule_id 中使用: '{best_rule_id}'
    - 請在 evidence.decision_rule_name 中使用: '{best_rule.rule_name}'
    - 請在 evidence.rule_similarity_score 中使用: {best_similarity:.3f}
    """

            prompt = f"""請將以下職業安全衛生義務分類到風險控制層級(Hierarchy of Controls)。

    義務範例(共 {cluster_info.get('member_count', len(members))} 條):
    {sample_str}
    {rules_injection}

    風險控制層級(按優先順序):
    1. Elimination - 消除危害
    2. Substitution - 替代
    3. EngineeringControl - 工程控制
    4. AdministrativeControl - 管理控制
    5. PPE - 個人防護具

    輸出 JSON:
    {{
    "control_type": "{best_rule.category if matched_rules else 'AdministrativeControl'}",
    "evidence": {{
        "keywords_matched": ["關鍵詞"],
        "decision_rule_id": "{matched_rules[0][0] if matched_rules else ''}",
        "decision_rule_name": "{self.rule_base[matched_rules[0][0]].rule_name if matched_rules else ''}",
        "rule_similarity_score": {matched_rules[0][1] if matched_rules else 0.0},
        "confidence_factors": {{"keyword_match_strength": 0.8, "context_clarity": 0.7, "domain_alignment": 0.9}},
        "text_snippets": ["證據文本"]
    }}
    }}

    只輸出JSON。"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "你是職業安全衛生風險控制專家。你必須使用系統提供的語意匹配結果。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                result_text = response.choices[0].message.content.strip()
                parsed = self._safe_parse_json_from_llm(result_text)

                if not isinstance(parsed, dict) or 'evidence' not in parsed:
                    raise ValueError("分類結果缺少 evidence 結構")
                
                evidence_data = parsed.get('evidence', {})
                
                # ⚠️ v5 關鍵: 驗證並強制修正
                if matched_rules:
                    expected_rule_id = matched_rules[0][0]
                    actual_rule_id = evidence_data.get('decision_rule_id', '')
                    
                    if actual_rule_id != expected_rule_id:
                        print(f"    ⚠️ 聚類 {cluster_id}: LLM 未使用注入的 rule_id,已自動修正")
                        evidence_data['decision_rule_id'] = expected_rule_id
                        evidence_data['decision_rule_name'] = self.rule_base[expected_rule_id].rule_name
                        evidence_data['rule_similarity_score'] = matched_rules[0][1]
                
                # 繼續處理... (與 v4 相同,創建 ClassificationResult 等)
                evidence = StructuredEvidence(
                    keywords_matched=evidence_data.get('keywords_matched', []),
                    decision_rule_id=evidence_data.get('decision_rule_id', ''),
                    decision_rule_name=evidence_data.get('decision_rule_name', ''),
                    rule_similarity_score=evidence_data.get('rule_similarity_score', 0.0),
                    confidence_factors=evidence_data.get('confidence_factors', {}),
                    alternative_classifications=evidence_data.get('alternative_classifications', []),
                    text_snippets=evidence_data.get('text_snippets', [])
                )
                
                conf_factors = evidence.confidence_factors
                overall_confidence = np.mean([
                    conf_factors.get('keyword_match_strength', 0.5),
                    conf_factors.get('context_clarity', 0.5),
                    conf_factors.get('domain_alignment', 0.5)
                ])
                
                temp_classification = ClassificationResult(
                    classification=parsed.get('control_type', 'Unknown'),
                    confidence=overall_confidence,
                    evidence=evidence,
                    review_status=ReviewStatus.AUTO_APPROVED
                )
                
                unified_priority = self._calculate_unified_priority(
                    cluster_info=cluster_info,
                    classification_result=temp_classification
                )
                
                if overall_confidence < self.CONFIDENCE_THRESHOLD or unified_priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH]:
                    review_status = ReviewStatus.PENDING_REVIEW
                    
                    al_score = self._calculate_active_learning_score(
                        cluster_info=cluster_info,
                        classification_result=temp_classification
                    )
                    
                    self.low_confidence_classifications.append({
                        'cluster_id': cluster_id,
                        'control_type': parsed.get('control_type'),
                        'confidence': overall_confidence,
                        'evidence': asdict(evidence),
                        'sample_obligations': sample_obligations[:3],
                        'active_learning_score': asdict(al_score),
                        'review_priority': unified_priority.value
                    })
                else:
                    review_status = ReviewStatus.AUTO_APPROVED
                
                classification_result = ClassificationResult(
                    classification=parsed.get('control_type', 'Unknown'),
                    confidence=overall_confidence,
                    evidence=evidence,
                    review_status=review_status,
                    review_priority=unified_priority
                )
                
                result_dict = asdict(classification_result)
                
                self.control_type_mapping[cluster_id] = result_dict
                cluster_info['control_type_classification'] = result_dict
                
            except Exception as e:
                print(f"    ✗ 聚類 {cluster_id} 分類失敗: {e}")
                # 回退邏輯 (與 v4 相同)
                fallback_classification = ClassificationResult(
                    classification="AdministrativeControl",
                    confidence=0.0,
                    evidence=StructuredEvidence(
                        decision_rule_id="",
                        decision_rule_name="分類失敗,使用預設值"
                    ),
                    review_status=ReviewStatus.PENDING_REVIEW,
                    review_priority=ReviewPriority.HIGH
                )
                
                self.control_type_mapping[cluster_id] = asdict(fallback_classification)
                cluster_info['control_type_classification'] = asdict(fallback_classification)
        
        # 統計 (與 v4 相同)
        type_counts = defaultdict(int)
        review_needed = 0
        priority_counts = defaultdict(int)
        
        for mapping in self.control_type_mapping.values():
            if isinstance(mapping, dict):
                ct = mapping.get('classification', 'Unknown')
                type_counts[ct] += 1
                if mapping.get('review_status') == ReviewStatus.PENDING_REVIEW.value:
                    review_needed += 1
                priority = mapping.get('review_priority', 'low')
                priority_counts[priority] += 1
        
        if type_counts:
            print(f"\n  ✓ 控制類型分類完成:")
            for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"    - {ct}: {count} 個群組")
            print(f"  ⚠️ 其中 {review_needed} 個需要人工審核")
            print(f"  → 優先級分布: {dict(priority_counts)}")
        
        self._save_control_type_mapping()
        self._queue_low_confidence_classifications_v5()
    
    def _save_control_type_mapping(self):
        """儲存控制類型映射"""
        if not self.control_type_mapping:
            print(f"  ⚠️ 警告: 沒有控制類型映射資料可儲存 (self.control_type_mapping 為空)")
            print(f"    → 因此 control_type_mapping.json 將不會被建立。")
            return
            
        output_path = os.path.join(self.output_dir, "control_type_mapping.json")
        
        serializable_mapping = self._make_json_serializable(self.control_type_mapping)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_mapping, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 控制類型映射已儲存至 {output_path}")
        except Exception as e:
            print(f"  ✗ 錯誤: 儲存 control_type_mapping.json 失敗: {e}")

    def _queue_low_confidence_classifications_v5(self):
        """
        v4 版本: 將低信心分類加入審核佇列
        使用統一的優先級邏輯
        """
        if not self.low_confidence_classifications:
            print("  ✓ 所有分類信心度良好,無需審核")
            return
        
        # 按優先級排序
        priority_order = {
            ReviewPriority.CRITICAL.value: 0,
            ReviewPriority.HIGH.value: 1,
            ReviewPriority.MEDIUM.value: 2,
            ReviewPriority.LOW.value: 3
        }
        
        self.low_confidence_classifications.sort(
            key=lambda x: (
                priority_order.get(x.get('review_priority', 'low'), 999),
                -x.get('active_learning_score', {}).get('total_priority', 0),
                x.get('confidence', 0)
            )
        )
        
        queue_file = os.path.join(self.review_queue_dir, "low_confidence_classifications_queue.json")
        
        queue_data = {
            "generated_at": datetime.now().isoformat(),
            "total_low_confidence": len(self.low_confidence_classifications),
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "priority_distribution": self._get_priority_distribution(self.low_confidence_classifications),
            "review_instructions": "請人工審核以下低信心度分類,確認或修正控制類型。優先處理 CRITICAL 和 HIGH 項目。",
            "classifications": self.low_confidence_classifications
        }
        
        serializable_queue = self._make_json_serializable(queue_data)
        
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_queue, f, ensure_ascii=False, indent=2)
        
        print(f"  ⚠️ 已將 {len(self.low_confidence_classifications)} 個低信心分類加入審核佇列")
        print(f"  → 優先級分布: {queue_data['priority_distribution']}")
        print(f"  → 審核佇列檔案: {queue_file}")
    
    # ========================================================================
    # 輔助方法 (繼承 v3)
    # ========================================================================
    
    def _get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """獲取文本的 Embeddings"""
        
        if not texts:
            print("    ⚠️ 警告:文本列表為空,返回空向量")
            return np.array([])
        
        all_embeddings = []
        failed_count = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if i % 200 == 0 or i + batch_size >= len(texts):
                    print(f"    → Embeddings 進度: {min(i + batch_size, len(texts))}/{len(texts)}")
                
            except Exception as e:
                print(f"    ✗ 批次 {i//batch_size + 1} 失敗: {e}")
                failed_count += len(batch)
                all_embeddings.extend([np.zeros(1536).tolist() for _ in batch])
        
        if failed_count > 0:
            print(f"    ⚠️ 有 {failed_count} 個文本的 embedding 生成失敗,使用零向量替代")
        
        result = np.array(all_embeddings)
        print(f"    ✓ 向量生成完成,形狀: {result.shape}")
        
        return result
    
    def _safe_parse_json_from_llm(self, text: str) -> Any:
        """從 LLM 回應文字中安全解析 JSON"""
        if not text or not isinstance(text, str):
            return None

        cleaned = re.sub(r'```(?:json)?', '', text, flags=re.IGNORECASE).strip()

        candidates = []
        obj_matches = re.findall(r'\{[\s\S]*\}', cleaned)
        arr_matches = re.findall(r'\[[\s\S]*\]', cleaned)
        if obj_matches:
            candidates.extend(obj_matches)
        if arr_matches:
            candidates.extend(arr_matches)
        if not candidates:
            candidates = [cleaned]

        for candidate in candidates:
            cand = candidate.strip()
            cand = re.sub(r',\s*([\]\}])', r'\1', cand)

            try:
                return json.loads(cand)
            except Exception:
                pass

            try:
                cand2 = cand
                if "'" in cand2 and '"' not in cand2:
                    cand2 = cand2.replace("'", '"')
                    return json.loads(cand2)
            except Exception:
                pass

            try:
                return ast.literal_eval(cand)
            except Exception:
                pass

        return None
    
    def _make_json_serializable(self, obj):
        """遞迴轉換物件為 JSON 可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            return self._make_json_serializable(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__)
        else:
            return obj
    
    def _save_ontology(self):
        """儲存本體結構"""
        ontology_data = {
            "subject_ontology": self.subject_ontology,
            "object_ontology": self.object_ontology,
            "metadata": {
                "subject_count": len(self.subject_ontology),
                "object_count": len(self.object_ontology),
                "discovered_entities_count": len(self.discovered_entities)
            }
        }
        
        output_path = os.path.join(self.output_dir, "ontology.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 本體已儲存至 {output_path}")
    
    # ========================================================================
    # 階段五: 建構知識圖譜 (v5 實作 - 取代 v4 的 stub)
    # ========================================================================
    
    def build_knowledge_graph(self):
        """
        (v5 實作) 建構完整的知識圖譜
        
        這個函式會呼叫所有輔助函式來組裝並儲存 legal_kg.json。
        它取代了 v4 中被省略的實作。
        """
        print("\n🕸️ 建構知識圖譜 (v5 實作)...")
        
        # 1. 法律結構層 (Law -> Chapter -> Section -> Article)
        self._build_legal_structure()
        
        # 2. 語意層 (NormalizedObligation -> ControlType)
        if self.obligation_clusters:
            self._build_semantic_layer_v5()
        else:
            print("  ⚠️ 跳過語意層建構(無義務聚類資料)")
        
        # 3. 本體層 (SubjectEntity, ObjectEntity)
        self._build_ontology_layer()
        
        # 4. 事件層 (LegalEvent -> Entities)
        if self.legal_events:
            self._build_event_layer()
        
        # 5. 規則層 (DecisionRule -> ControlType)
        self._build_rule_layer_v5()
        
        # 儲存圖譜
        self._save_knowledge_graph_v5()

    def _build_legal_structure(self):
        """(v5 實作) 建立法律結構層 (繼承自 v3)"""
        print("  → 建立法律結構層...")
        
        law_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for doc in self.documents:
            metadata = doc.get('metadata', {})
            law_name = metadata.get('law_name', 'Unknown')
            chapter = metadata.get('chapter', 'unknown')
            section = metadata.get('section', 'unknown')
            article = metadata.get('article', 'unknown')
            content = doc.get('content', '')
            
            law_dict[law_name][chapter][section].append({
                'article': article,
                'content': content,
                'metadata': metadata
            })
        
        for law_name, chapters in law_dict.items():
            law_node_id = f"law_{self._sanitize_id(law_name)}"
            self.nodes.append(GraphNode(
                id=law_node_id,
                type="Law",
                properties={"name": law_name}
            ))
            
            for chapter_id, sections in chapters.items():
                chapter_node_id = f"{law_node_id}_ch_{chapter_id}"
                self.nodes.append(GraphNode(
                    id=chapter_node_id,
                    type="Chapter",
                    properties={"id": chapter_id, "law": law_name}
                ))
                self.edges.append(GraphEdge(
                    source=law_node_id,
                    target=chapter_node_id,
                    type="HAS_CHAPTER"
                ))
                
                for section_id, articles in sections.items():
                    section_node_id = f"{chapter_node_id}_sec_{section_id}"
                    self.nodes.append(GraphNode(
                        id=section_node_id,
                        type="Section",
                        properties={"id": section_id}
                    ))
                    self.edges.append(GraphEdge(
                        source=chapter_node_id,
                        target=section_node_id,
                        type="HAS_SECTION"
                    ))
                    
                    for article_data in articles:
                        article_id = article_data['article']
                        article_node_id = f"{section_node_id}_art_{article_id}"
                        self.nodes.append(GraphNode(
                            id=article_node_id,
                            type="Article",
                            properties={
                                "id": article_id,
                                "content": article_data['content']
                            }
                        ))
                        self.edges.append(GraphEdge(
                            source=section_node_id,
                            target=article_node_id,
                            type="HAS_ARTICLE"
                        ))
        
        print(f"    ✓ 法律結構層:{len([n for n in self.nodes if n.type in ['Law', 'Chapter', 'Section', 'Article']])} 個節點")

    def _build_semantic_layer_v5(self):
        """
        (v5 實作) 建立語意層 (v3 的升級版)
        
        - 增強: 儲存 v4 產生的 review_priority
        - 增強: 儲存 v4 產生的 rule_similarity_score
        """
        print("  → 建立語意層 (v5)...")
        
        if not self.obligation_clusters:
            print("    ⚠️ 無義務聚類資料")
            return
        
        for cluster_id, cluster_info in self.obligation_clusters.items():
            cluster_id_str = str(cluster_id)
            
            if not isinstance(cluster_info, dict):
                continue
            
            standard_code = cluster_info.get('standard_code', f'CLUSTER_{cluster_id_str}')
            obligation_node_id = f"obligation_{standard_code}"
            
            evidence = cluster_info.get('evidence', {})
            control_classification = cluster_info.get('control_type_classification', {})
            
            self.nodes.append(GraphNode(
                id=obligation_node_id,
                type="NormalizedObligation",
                properties={
                    "name": cluster_info.get('standard_name', '未命名義務'),
                    "code": standard_code,
                    "category": cluster_info.get('category', '未分類'),
                    "description": cluster_info.get('description', ''),
                    "member_count": cluster_info.get('member_count', 0),
                    "overall_confidence": cluster_info.get('overall_confidence', 0.0),
                    "review_status": cluster_info.get('review_status', 'unknown'),
                    "review_priority": cluster_info.get('review_priority', 'low'), # v5 新增
                    "evidence_keywords": evidence.get('keywords_matched', []),
                    "evidence_rule_id": evidence.get('decision_rule_id', ''),
                    "evidence_rule_name": evidence.get('decision_rule_name', ''),
                    "evidence_rule_similarity": evidence.get('rule_similarity_score', 0.0), # v5 新增
                    "quality_metrics": cluster_info.get('quality_metrics', {})
                }
            ))
            
            # 連接到 ControlType
            if control_classification and isinstance(control_classification, dict):
                control_type = control_classification.get('classification', 'Unknown')
                control_node_id = f"control_{control_type}"
                
                # 確保 ControlType 節點只創建一次
                if not any(n.id == control_node_id for n in self.nodes):
                    self.nodes.append(GraphNode(
                        id=control_node_id,
                        type="ControlType",
                        properties={"type": control_type}
                    ))
                
                control_evidence = control_classification.get('evidence', {})
                self.edges.append(GraphEdge(
                    source=obligation_node_id,
                    target=control_node_id,
                    type="IS_A",
                    properties={
                        "confidence": control_classification.get('confidence', 0.0),
                        "review_status": control_classification.get('review_status', 'unknown'),
                        "review_priority": control_classification.get('review_priority', 'low'), # v5 新增
                        "evidence_keywords": control_evidence.get('keywords_matched', []),
                        "evidence_rule_id": control_evidence.get('decision_rule_id', ''),
                        "evidence_rule_similarity": control_evidence.get('rule_similarity_score', 0.0) # v5 新增
                    }
                ))
        
        semantic_nodes = len([n for n in self.nodes if n.type in ['NormalizedObligation', 'ControlType']])
        print(f"    ✓ 語意層:{semantic_nodes} 個節點")

    def _build_ontology_layer(self):
        """(v5 實作) 將本體結構加入知識圖譜 (繼承自 v3)"""
        print("  → 建立本體層...")
        
        ontology_node_count = 0
        
        for entity_name, entity_info in self.subject_ontology.items():
            node_id = f"subject_{entity_info.get('standard_name', self._sanitize_id(entity_name))}"
            
            # 避免重複添加
            if any(n.id == node_id for n in self.nodes): continue
            
            self.nodes.append(GraphNode(
                id=node_id,
                type="SubjectEntity",
                properties={
                    "name": entity_name,
                    "standard_name": entity_info.get('standard_name'),
                    "parent_category": entity_info.get('parent_category'),
                    "level": entity_info.get('level'),
                    "hierarchy_path": entity_info.get('hierarchy_path'),
                    "synonyms": entity_info.get('synonyms', []),
                    "description": entity_info.get('description')
                }
            ))
            ontology_node_count += 1
            
            parent = entity_info.get('parent_category')
            if parent:
                parent_id = f"subject_{parent}"
                if not any(n.id == parent_id for n in self.nodes):
                    self.nodes.append(GraphNode(
                        id=parent_id,
                        type="SubjectCategory",
                        properties={"name": parent}
                    ))
                
                self.edges.append(GraphEdge(
                    source=node_id,
                    target=parent_id,
                    type="IS_A"
                ))
        
        for entity_name, entity_info in self.object_ontology.items():
            node_id = f"object_{entity_info.get('standard_name', self._sanitize_id(entity_name))}"
            
            # 避免重複添加
            if any(n.id == node_id for n in self.nodes): continue
            
            self.nodes.append(GraphNode(
                id=node_id,
                type="ObjectEntity",
                properties={
                    "name": entity_name,
                    "standard_name": entity_info.get('standard_name'),
                    "parent_category": entity_info.get('parent_category'),
                    "level": entity_info.get('level'),
                    "hierarchy_path": entity_info.get('hierarchy_path'),
                    "synonyms": entity_info.get('synonyms', []),
                    "description": entity_info.get('description')
                }
            ))
            ontology_node_count += 1
            
            parent = entity_info.get('parent_category')
            if parent:
                parent_id = f"object_{parent}"
                if not any(n.id == parent_id for n in self.nodes):
                    self.nodes.append(GraphNode(
                        id=parent_id,
                        type="ObjectCategory",
                        properties={"name": parent}
                    ))
                
                self.edges.append(GraphEdge(
                    source=node_id,
                    target=parent_id,
                    type="IS_A"
                ))
        
        print(f"    ✓ 本體層:{ontology_node_count} 個實體節點")

    def _build_event_layer(self):
        """(v5 實作) 建立事件層
        
        關鍵修正:
        - 呼叫 _find_ontology_node_semantic (語意匹配)
        - 取代 v4 的 _find_ontology_node_for_entity (字串匹配)
        - 增加語意連結統計
        """
        print("  → 建立事件層 (v5 語意連結版)...")
        
        event_count = 0
        semantic_links_created = 0
        unlinked_entities = set() # 用於追蹤無法連結的實體
        
        for event_id, event in self.legal_events.items():
            event_node_id = event.event_id 
            
            if any(n.id == event_node_id for n in self.nodes): continue
            
            self.nodes.append(GraphNode(
                id=event_node_id,
                type="LegalEvent",
                properties={
                    "event_id": event.event_id,
                    "action": event.action,
                    "actor": event.actor,
                    "patients": event.patients,
                    "instruments": event.instruments,
                    "locations": event.locations,
                    "conditions": event.conditions,
                    "temporal": event.temporal,
                    "purpose": event.purpose,
                    "source_article": event.source_article,
                    "confidence": event.confidence
                }
            ))
            event_count += 1
            
            # === v5 關鍵修正: 呼叫語意連結 ===
            
            # 1. 連結 Actor
            if event.actor:
                actor_node_id = self._find_ontology_node_semantic(event.actor, threshold=0.7)
                if actor_node_id:
                    self.edges.append(GraphEdge(
                        source=event_node_id,
                        target=actor_node_id,
                        type="HAS_ACTOR",
                        properties={"source_text": event.actor}
                    ))
                    semantic_links_created += 1
                else:
                    unlinked_entities.add(event.actor)
            
            # 2. 連結 Patients
            for entity_text in event.patients:
                entity_node_id = self._find_ontology_node_semantic(entity_text, threshold=0.6)
                if entity_node_id:
                    self.edges.append(GraphEdge(
                        source=event_node_id,
                        target=entity_node_id,
                        type="HAS_PATIENT",
                        properties={"source_text": entity_text}
                    ))
                    semantic_links_created += 1
                else:
                    unlinked_entities.add(entity_text)
            
            # 3. 連結 Instruments
            for entity_text in event.instruments:
                entity_node_id = self._find_ontology_node_semantic(entity_text, threshold=0.6)
                if entity_node_id:
                    self.edges.append(GraphEdge(
                        source=event_node_id,
                        target=entity_node_id,
                        type="USES_INSTRUMENT",
                        properties={"source_text": entity_text}
                    ))
                    semantic_links_created += 1
                else:
                    unlinked_entities.add(entity_text)
            
            # 4. 連結 Locations
            for entity_text in event.locations:
                entity_node_id = self._find_ontology_node_semantic(entity_text, threshold=0.6)
                if entity_node_id:
                    self.edges.append(GraphEdge(
                        source=event_node_id,
                        target=entity_node_id,
                        type="AT_LOCATION",
                        properties={"source_text": entity_text}
                    ))
                    semantic_links_created += 1
                else:
                    unlinked_entities.add(entity_text)
        
        print(f"    ✓ 事件層:{event_count} 個事件節點")
        print(f"    ✓ v5 改進: 建立了 {semantic_links_created} 條語意實體連結 (HAS_ACTOR, HAS_PATIENT, ...)")
        
        if unlinked_entities:
            print(f"    ⚠️ v5 警示: 發現 {len(unlinked_entities)} 個無法連結到本體的新實體。")
            print(f"      (建議: 擴充 ontology.json 或建立 'ontology_expansion_queue.json' 解決「問題三」)")

    def _build_rule_layer_v5(self):
        """
        (v5 實作) 建立規則層 (v3 升級版)
        
        - 增強: 增加 has_embedding 屬性,不儲存完整向量
        """
        print("  → 建立規則層 (v5)...")
        
        for rule_id, rule_template in self.rule_base.items():
            rule_node_id = f"rule_{rule_id}"
            
            if any(n.id == rule_node_id for n in self.nodes): continue
            
            self.nodes.append(GraphNode(
                id=rule_node_id,
                type="DecisionRule",
                properties={
                    "rule_id": rule_template.rule_id,
                    "rule_name": rule_template.rule_name,
                    "category": rule_template.category,
                    "pattern": rule_template.pattern,
                    "keywords": rule_template.keywords,
                    "examples": rule_template.examples,
                    "control_type_affinity": rule_template.control_type_affinity,
                    "has_embedding": rule_template.embedding_vector is not None # v5 新增
                }
            ))
            
            # 連接規則到控制類型
            control_node_id = f"control_{rule_template.category}"
            if any(n.id == control_node_id for n in self.nodes):
                self.edges.append(GraphEdge(
                    source=rule_node_id,
                    target=control_node_id,
                    type="SUPPORTS",
                    properties={
                        "affinity": rule_template.control_type_affinity.get(rule_template.category, 0.0)
                    }
                ))
        
        print(f"    ✓ 規則層:{len(self.rule_base)} 個規則節點")

    def _find_ontology_node_for_entity(self, entity_text: str) -> Optional[str]:
        """
        (v5 實作) 為實體文本找到對應的本體節點 ID
        
        (這是 v3 的簡易匹配版。在 v6 中,我們應將此升級為語意向量匹配)
        """
        # 1. 優先檢查標準名稱 (Standard Name)
        for entity_name, entity_info in self.object_ontology.items():
            if entity_info.get('standard_name') == entity_text:
                return f"object_{entity_info['standard_name']}"
        for entity_name, entity_info in self.subject_ontology.items():
            if entity_info.get('standard_name') == entity_text:
                return f"subject_{entity_info['standard_name']}"
        
        # 2. 檢查名稱 (Name)
        if entity_text in self.object_ontology:
            return f"object_{self.object_ontology[entity_text].get('standard_name')}"
        if entity_text in self.subject_ontology:
            return f"subject_{self.subject_ontology[entity_text].get('standard_name')}"
        
        # 3. 檢查同義詞 (Synonyms)
        for entity_name, entity_info in self.object_ontology.items():
            if entity_text in entity_info.get('synonyms', []):
                return f"object_{entity_info['standard_name']}"
        for entity_name, entity_info in self.subject_ontology.items():
            if entity_text in entity_info.get('synonyms', []):
                return f"subject_{entity_info['standard_name']}"
        
        return None

    def _sanitize_id(self, text: str) -> str:
        """(v5 實作) 清理文字以生成合法的 ID (繼承自 v3)"""
        return re.sub(r'[^\w]', '_', text)

    def _save_knowledge_graph_v5(self):
        """
        (v5 實作) 儲存知識圖譜
        
        - 增強: 更新 metadata 版本號
        """
        kg_data = {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "statistics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": {
                    node_type: len([n for n in self.nodes if n.type == node_type])
                    for node_type in set(n.type for n in self.nodes)
                }
            },
            "metadata": {
                "version": "5.0", # v5 更新
                "features": [
                    "event_extraction",
                    "computable_rule_base_v2_semantic", # v5 更新
                    "active_learning_priority_v2_unified", # v5 更新
                    "incremental_ontology_expansion",
                    "hierarchical_ontology",
                    "structured_evidence_v2_semantic", # v5 更新
                    "context_aware_obligation_extraction", # v5 新增
                    "hdbscan_clustering" # v5 新增
                ],
                "confidence_threshold": self.CONFIDENCE_THRESHOLD,
                "min_cluster_size": self.MIN_CLUSTER_SIZE,
                "total_events_extracted": len(self.legal_events),
                "total_rules_defined": len(self.rule_base),
                "discovered_entities": len(self.discovered_entities)
            }
        }
        
        serializable_kg = self._make_json_serializable(kg_data)
        
        output_path = os.path.join(self.output_dir, "legal_kg.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_kg, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 知識圖譜已完成!")
        print(f"  → 節點總數: {len(self.nodes)}")
        print(f"  → 邊總數: {len(self.edges)}")
        print(f"  → 輸出路徑: {output_path}")   
    
    # ========================================================================
    # v4 增強: 摘要報告生成
    # ========================================================================
    
    def _generate_review_summary_v5(self):
        """
        v5 核心改進: 摘要報告邏輯修正
        解決問題四 - 檢查所有佇列,確保報告一致性
        
        關鍵修改:
        1. 檢查 problematic_clusters 列表
        2. 檢查 low_confidence_classifications 列表
        3. 基於實際內容生成建議
        """
        # ⚠️ v5 關鍵: 檢查所有佇列內容
        all_review_items = []
        
        if self.problematic_clusters:
            all_review_items.extend(self.problematic_clusters)
        
        if self.low_confidence_classifications:
            all_review_items.extend(self.low_confidence_classifications)
        
        # 統計各優先級數量
        priority_counts = defaultdict(int)
        for item in all_review_items:
            priority = item.get('review_priority', ReviewPriority.LOW.value)
            priority_counts[priority] += 1
        
        critical_count = priority_counts.get(ReviewPriority.CRITICAL.value, 0)
        high_count = priority_counts.get(ReviewPriority.HIGH.value, 0)
        medium_count = priority_counts.get(ReviewPriority.MEDIUM.value, 0)
        low_count = priority_counts.get(ReviewPriority.LOW.value, 0)
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "version": "v5.0",
            "improvements": [
                "葉聚類 (Leaf Clustering)",
                "提示詞鏈 (Prompt Chaining)",
                "語意實體連結 (Semantic Entity Linking)",
                "摘要邏輯修正 - 檢查所有佇列"
            ],
            "review_queues": {
                "low_confidence_classifications": {
                    "count": len(self.low_confidence_classifications),
                    "file": "review_queue/low_confidence_classifications_queue.json",
                    "priority_distribution": self._get_priority_distribution(self.low_confidence_classifications)
                },
                "problematic_clusters": {
                    "count": len(self.problematic_clusters),
                    "file": "review_queue/problematic_clusters_queue.json",
                    "priority_distribution": self._get_priority_distribution(self.problematic_clusters)
                }
            },
            "thresholds": {
                "confidence_threshold": self.CONFIDENCE_THRESHOLD,
                "min_cluster_size": self.MIN_CLUSTER_SIZE,
                "hdbscan_min_cluster_size": 3,  # v5
                "hdbscan_cluster_selection": "leaf"  # v5
            },
            "recommendations": []
        }
        
        # ⚠️ v5 關鍵: 基於實際佇列內容生成建議
        if critical_count > 0:
            summary["recommendations"].append({
                "priority": ReviewPriority.CRITICAL.value,
                "action": "立即審核關鍵項目",
                "description": f"有 {critical_count} 個項目被標記為 CRITICAL 優先級",
                "details": {
                    "from_clusters": sum(1 for x in self.problematic_clusters if x.get('review_priority') == ReviewPriority.CRITICAL.value),
                    "from_classifications": sum(1 for x in self.low_confidence_classifications if x.get('review_priority') == ReviewPriority.CRITICAL.value)
                },
                "next_steps": "這些項目影響核心法規解釋,必須優先處理"
            })
        
        if high_count > 0:
            summary["recommendations"].append({
                "priority": ReviewPriority.HIGH.value,
                "action": "審核高優先級項目",
                "description": f"有 {high_count} 個項目被標記為 HIGH 優先級",
                "details": {
                    "from_clusters": sum(1 for x in self.problematic_clusters if x.get('review_priority') == ReviewPriority.HIGH.value),
                    "from_classifications": sum(1 for x in self.low_confidence_classifications if x.get('review_priority') == ReviewPriority.HIGH.value)
                },
                "next_steps": "這些項目涉及高頻使用或高不確定性,建議盡快審核"
            })
        
        if medium_count > 0:
            summary["recommendations"].append({
                "priority": ReviewPriority.MEDIUM.value,
                "action": "定期審核中等優先級項目",
                "description": f"有 {medium_count} 個項目被標記為 MEDIUM 優先級",
                "next_steps": "可安排定期審核週期處理"
            })
        
        if not summary["recommendations"]:
            summary["recommendations"].append({
                "priority": ReviewPriority.LOW.value,
                "action": "無需審核",
                "description": "所有自動化處理結果品質良好",
                "next_steps": "可直接使用知識圖譜"
            })
        
        # v5 統計
        summary["v5_statistics"] = {
            "total_rules_defined": len(self.rule_base),
            "rules_with_embeddings": len(self.rule_embeddings),
            "ontology_entities_with_embeddings": len(self.ontology_embeddings),
            "obligations_with_context": sum(1 for c in self.obligation_clusters.values() 
                                        if isinstance(c, dict) and 'members' in c),
            "semantic_matched_rules": sum(1 for c in self.obligation_clusters.values()
                                        if isinstance(c, dict) and c.get('evidence', {}).get('rule_similarity_score', 0) > 0),
            "cluster_selection_method": "leaf",
            "total_review_items": len(all_review_items),
            "priority_consistency": "優先級邏輯已統一,摘要與佇列100%一致"
        }
        
        serializable_summary = self._make_json_serializable(summary)
        
        summary_file = os.path.join(self.review_queue_dir, "review_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 審核摘要報告 (v5):")
        print(f"  → CRITICAL 優先級: {critical_count} 個")
        print(f"  → HIGH 優先級: {high_count} 個")
        print(f"  → MEDIUM 優先級: {medium_count} 個")
        print(f"  → LOW 優先級: {low_count} 個")
        print(f"  ✓ v5 改進: 摘要與佇列內容 100% 一致")
        print(f"  → 詳細報告: {summary_file}")
    
    def _generate_rule_usage_report(self):
        """生成規則使用統計報告"""
        print("\n📊 生成規則使用統計...")
        
        rule_usage = defaultdict(int)
        
        for cluster_info in self.obligation_clusters.values():
            if isinstance(cluster_info, dict):
                evidence = cluster_info.get('evidence', {})
                rule_id = evidence.get('decision_rule_id', '')
                if rule_id:
                    rule_usage[rule_id] += 1
        
        for mapping in self.control_type_mapping.values():
            if isinstance(mapping, dict):
                evidence = mapping.get('evidence', {})
                rule_id = evidence.get('decision_rule_id', '')
                if rule_id:
                    rule_usage[rule_id] += 1
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "version": "v4.0",
            "total_rules_defined": len(self.rule_base),
            "rules_used": len(rule_usage),
            "rules_unused": len(self.rule_base) - len(rule_usage),
            "semantic_matching_enabled": True,
            "rule_usage_details": {
                rule_id: {
                    "usage_count": count,
                    "rule_name": self.rule_base[rule_id].rule_name if rule_id in self.rule_base else "Unknown",
                    "category": self.rule_base[rule_id].category if rule_id in self.rule_base else "Unknown",
                    "has_embedding": rule_id in self.rule_embeddings
                }
                for rule_id, count in sorted(rule_usage.items(), key=lambda x: -x[1])
            },
            "unused_rules": [
                {
                    "rule_id": rule_id,
                    "rule_name": rule_template.rule_name,
                    "category": rule_template.category,
                    "possible_reason": "語料庫中無相關義務" if rule_id in self.rule_embeddings else "向量生成失敗"
                }
                for rule_id, rule_template in self.rule_base.items()
                if rule_id not in rule_usage
            ]
        }
        
        report_file = os.path.join(self.output_dir, "rule_usage_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 規則使用統計:")
        print(f"    - 已定義規則: {len(self.rule_base)}")
        print(f"    - 已使用規則: {len(rule_usage)}")
        print(f"    - 未使用規則: {len(self.rule_base) - len(rule_usage)}")
        print(f"    - 詳細報告: {report_file}")
        
        if rule_usage:
            print(f"  → 最常用規則 TOP 5:")
            for i, (rule_id, count) in enumerate(sorted(rule_usage.items(), key=lambda x: -x[1])[:5], 1):
                rule_name = self.rule_base.get(rule_id, type('obj', (), {'rule_name': 'Unknown'})).rule_name
                print(f"    {i}. {rule_id} ({rule_name}): {count} 次")
    
    # ========================================================================
    # 主流程
    # ========================================================================
    
    def build(self):
        """
        執行完整建構流程 (v5 版本)
        
        修改點:
        - 調用 _generate_review_summary_v5 (取代 v4 的 _generate_review_summary_v4)
        - 新增 v5 版本標識
        """
        print("=" * 70)
        print("Legal Knowledge Graph Builder v5")
        print("職業安全衛生法律知識圖譜建構器 v5")
        print("(葉聚類、提示詞鏈、語意實體連結版)")
        print("=" * 70)
        
        # 階段 1: 載入 (繼承 v4)
        self.load_documents()
        
        # 階段 2: 事件抽取 (繼承 v4)
        self.extract_legal_events()
        
        # 階段 2.5: 本體建構 (繼承 v4)
        print("\n🗂️ 建立分層本體...")
        self.subject_ontology = self.base_subject_ontology
        self.object_ontology.update(self.base_object_ontology)
        self._save_ontology()
        
        # ⚠️ v5 修改: 義務正規化 (調用 v5 版本)
        self.normalize_obligations()
        
        # ⚠️ v5 修改: 風險分類 (調用 v5 版本,見下方函式 8)
        self.classify_control_types()
        
        # 階段 5: 圖譜建構 (繼承 v4)
        self.build_knowledge_graph()
        
        # ⚠️ v5 修改: 報告生成 (調用 v5 版本)
        self._generate_review_summary_v5()
        self._generate_rule_usage_report()
        
        print("\n" + "=" * 70)
        print("✨ 所有處理完成 (v5)!")
        print("=" * 70)

# ============================================================================
# 使用範例
# ============================================================================

def get_api_key():
    """從環境變數或使用者輸入獲取 API Key"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        print("✓ 從環境變數 OPENAI_API_KEY 載入 API Key")
        return api_key
    
    print("=" * 70)
    print("OpenAI API Key 設定")
    print("=" * 70)
    print("請輸入您的 OpenAI API Key")
    print("(您可以從 https://platform.openai.com/api-keys 獲取)")
    print()
    
    api_key = getpass.getpass("API Key: ").strip()
    
    if not api_key:
        print("✗ 錯誤:未提供 API Key")
        sys.exit(1)
    
    return api_key


if __name__ == "__main__":
    print("=" * 70)
    print("Legal Knowledge Graph Builder v4")
    print("職業安全衛生法律知識圖譜建構器 v4 (智慧語意聚類與語境感知版)")
    print("=" * 70)
    print()
    
    # 獲取 API Key
    api_key = get_api_key()
    
    # 配置路徑
    # INPUT_PATH = "./processed_output/升降機安全檢查構造標準_processed.json" # 用來小規模測試
    # INPUT_PATH = "./processed_output/高壓氣體勞工安全規則_processed.json" # 用來小規模測試
    INPUT_PATH = "./processed_output/all_documents.json"
    OUTPUT_DIR = "./kg_output_v6"
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"\n✗ 錯誤:找不到輸入檔案 {INPUT_PATH}")
        print(f"請確認檔案路徑正確")
        sys.exit(1)
    
    print(f"\n配置資訊:")
    print(f"  輸入檔案: {INPUT_PATH}")
    print(f"  輸出目錄: {OUTPUT_DIR}")
    print()
    
    print("v4 核心改進:")
    print("  1. HDBSCAN 自適應聚類 - 自動過濾噪聲,解決單例聚類問題")
    print("  2. 語意規則匹配 - 使用向量相似度取代關鍵字匹配")
    print("  3. 語境感知萃取 - 自動處理法律引用(Anaphora)")
    print("  4. 統一優先級邏輯 - 修正人機迴圈內部矛盾")
    print()
    
    # 建構知識圖譜
    try:
        builder = LegalKGBuilderV4(
            api_key=api_key,
            input_path=INPUT_PATH,
            output_dir=OUTPUT_DIR
        )
        
        builder.build()
        
        print("\n" + "=" * 70)
        print("✨ 建構完成!輸出檔案:")
        print(f"  1. {OUTPUT_DIR}/legal_events.json - 法律事件結構")
        print(f"  2. {OUTPUT_DIR}/ontology.json - 分層本體結構(含自動發現實體)")
        print(f"  3. {OUTPUT_DIR}/obligation_clusters.json - 義務聚類結果(v4: HDBSCAN)")
        print(f"  4. {OUTPUT_DIR}/control_type_mapping.json - 控制類型映射(v4: 語意匹配)")
        print(f"  5. {OUTPUT_DIR}/rule_usage_report.json - 規則使用統計(v4: 語意相似度)")
        print(f"  6. {OUTPUT_DIR}/review_queue/ - 人機迴圈審核佇列(v4: 統一優先級)")
        print(f"     - review_summary.json - 審核摘要報告")
        print(f"     - problematic_clusters_queue.json - 問題聚類佇列")
        print(f"     - low_confidence_classifications_queue.json - 低信心分類佇列")
        print(f"     - noise_points.json - HDBSCAN 識別的噪聲點")
        print("=" * 70)
        print("\n🎓 v4 版本相較於 v3 的關鍵改進:")
        print("  ✓ 解決了單例聚類災難 (使用 HDBSCAN)")
        print("  ✓ 修正了規則使用偏差 (語意相似度匹配)")
        print("  ✓ 處理了法律引用問題 (語境感知萃取)")
        print("  ✓ 統一了優先級邏輯 (一致的 HITL 指令)")
        print()
        print("📝 下一步建議:")
        print("  1. 檢查 review_summary.json 了解審核優先級")
        print("  2. 處理 CRITICAL 和 HIGH 優先級項目")
        print("  3. 檢視 noise_points.json 確認被過濾的義務")
        print("  4. 比較 v3 和 v4 的 rule_usage_report 看改進效果")
        
    except KeyboardInterrupt:
        print("\n\n✗ 使用者中斷執行")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)