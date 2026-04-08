#!/usr/bin/env python3
"""
包含：
1) xmlcount_reward_func - XML 格式檢查（保留）
2) scoring_function     - 改成只評多選題
3) enhanced_scoring_function_v2 - 仍可用，套在多選題分數上
"""

import re
import itertools
import numpy as np
import math
from typing import List
import torch

import cn2an


############################ format reward ##############################

# 原本這個空函式只有 docstring，為避免混淆，改為整段註解（不刪）
# def count_xml(text) -> float:
#     """
#     計算 XML 標籤的正確性得分
#     Args:
#         text: 模型输出文本
#     Returns:
#         float: 得分 (0.0 - 1.0)
#     """

TAG_REAS_OPEN  = re.compile(r'<\s*reasoning\s*>', re.I)
TAG_REAS_CLOSE = re.compile(r'<\s*/\s*reasoning\s*>', re.I)
TAG_ANS_OPEN   = re.compile(r'<\s*answer\s*>', re.I)
TAG_ANS_CLOSE  = re.compile(r'<\s*/\s*answer\s*>', re.I)
TAG_EOA        = re.compile(r'<\s*eo[as]\s*>', re.I)  # 支援 <eoa> 或 <eos>
BRACKET_ANY    = re.compile(r'\[(法条|法條|刑期|罪名|金额|金額|正确答案|正確答案)\]')

def count_xml(text) -> float:
    score = 0.0

    if len(TAG_REAS_OPEN.findall(text)) == 1:
        score += 0.125
    if len(TAG_REAS_CLOSE.findall(text)) == 1:
        score += 0.125
    if len(TAG_ANS_OPEN.findall(text)) == 1:
        score += 0.125
    if len(TAG_ANS_CLOSE.findall(text)) == 1:
        score += 0.125

    if BRACKET_ANY.search(text):
        score += 0.125

    if len(TAG_EOA.findall(text)) >= 1:
        score += 0.125
        # 結尾多餘內容扣分：以最後一個 <eoa>/<eos> 為準
        last = None
        for m in TAG_EOA.finditer(text):
            last = m
        if last:
            tail = text[last.end():]
            penalty = len(tail.strip()) * 0.01
            score -= min(penalty, 0.375)

    # 推理＋答案都有內容再加分
    try:
        reasoning = re.search(r'<\s*reasoning\s*>(.*?)<\s*/\s*reasoning\s*>', text, re.S|re.I)
        answer   = re.search(r'<\s*answer\s*>(.*?)<\s*/\s*answer\s*>', text, re.S|re.I)
        if reasoning and reasoning.group(1).strip() and answer and answer.group(1).strip() and TAG_EOA.search(answer.group(1)):
            score += 0.25
    except Exception:
        pass

    return score


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    scores = [count_xml(c) for c in contents]
    for i, score in enumerate(scores):
        print(f"內容 {i+1} XML格式分數: {score:.3f}")
    return scores


################################ helpers (保留，選擇題模式不會呼叫) ###############################

# def extract_articles(text):
#     """提取法条编号（保留，選擇題模式不使用）"""
#     matches = re.findall(r'第(\d+(?:-\d+)?)條', text)
#     return matches

# def extract_pairs(text):
#     """提取被告-罪名对（保留，選擇題模式不使用）"""
#     return re.findall(r'\(([^)]+)\)犯\(([^)]+)\)', text)

# def extract_penalties(text):
#     """提取刑期信息（保留，選擇題模式不使用）"""
#     penalties = {}
#     defendants = re.findall(r'\(([^)]+)\)犯\(([^)]+)\)', text)
#     for name, _ in defendants:
#         pattern = fr'\({re.escape(name)}\).*?有期徒刑時長：(\d+)\)'
#         match = re.search(pattern, text)
#         if match:
#             fixed_term = match.group(1)
#             penalties[name] = int(fixed_term)
#     return penalties

# def calculate_term_score(predicted: int, actual: int) -> float:
#     """計算刑期分數（保留，選擇題模式不使用）"""
#     if predicted == actual:
#         return 1.0
#     if actual == 0:
#         return 0.0 if predicted > 0 else 1.0
#     if predicted == 0:
#         return 0.0
#     error_ratio = abs(predicted - actual) / actual
#     penalty_factor = 1.2 if predicted > actual else 1.0
#     score = max(0, 1 - error_ratio * penalty_factor)
#     return score

# def penalty_reward(response: str, answer: str) -> float:
#     """刑期預測獎勵函數（保留，選擇題模式不使用）"""

#     if not response or not answer:
#         return 0.0
#     resp_penalties = extract_penalties(response)
#     ans_penalties = extract_penalties(answer)
#     if not resp_penalties or not ans_penalties:
#         return 0.0
#     total_score = 0
#     total_defendants = len(ans_penalties)
#     for name, ans_term in ans_penalties.items():
#         if name in resp_penalties:
#             resp_term = resp_penalties[name]
#             score = calculate_term_score(resp_term, ans_term)
#             print(f"\n{name} 刑期評分詳情:")
#             print(f"有期徒刑: 預測={resp_term}, 實際={ans_term}, 得分={score:.3f}")
#             total_score += score
#     return total_score / total_defendants if total_defendants > 0 else 0

# def expand_brackets(text):
#     """擴展方括號表達式（保留，選擇題模式不使用）"""
#     while "[" in text and "]" in text:
#         match = re.search(r"\[(.*?)\]", text)
#         if not match:
#             break
#         options = match.group(1).split("、")
#         expanded = [text[:match.start()] + opt + text[match.end():] for opt in options]
#         text = expanded
#     if isinstance(text, list):
#         return list(itertools.product(*[t.split("、") if "、" in t else [t] for t in text]))
#     return [text]

# def extract_law_articles(text):
#     """
#     提取法條編號（保留，選擇題模式不使用）
#     修正：使用命名群組 'content'。原本 match.group(1) 容易取錯，先註解保留。
#     """
#     pattern = r'\[(?:法条|法條)\](?P<content>.*?)<\s*eo[as]\s*>'
#     match = re.search(pattern, text, re.DOTALL)
#     if not match:
#         return []
#     content = match.group('content').strip()
#     # content = match.group(1).strip()  # 會取到「法条/法條」，因此註解保留

#     prediction_law_chunks = re.split(r'[,，、\s]+', content)
#     prediction_law_index_digit_list = []
#     for prediction_law_chunk in prediction_law_chunks:
#         if not prediction_law_chunk:
#             continue
#         prediction_law_chunk = prediction_law_chunk.replace("萬元", "元")
#         prediction_law_chunk = re.sub(r'第(.*?)款', "", prediction_law_chunk)
#         prediction_law_chunk = re.sub(r'第(.*?)條', lambda m: m.group(1), prediction_law_chunk)
#         prediction_law_chunk = re.sub(r'第(.*?)条',  lambda m: m.group(1), prediction_law_chunk)
#         if cn2an:
#             try:
#                 prediction_law_chunk = cn2an.transform(prediction_law_chunk, "cn2an")
#             except:
#                 pass
#         nums = re.findall(r"\d+", prediction_law_chunk)
#         if not nums:
#             continue
#         val = int(nums[0])
#         if val <= 490:
#             prediction_law_index_digit_list.append(val)
#     return prediction_law_index_digit_list

# def normalize_accusation(accusation):
#     """規範化罪名（保留，選擇題模式不使用）"""
#     accusation = accusation.strip()
#     if accusation.endswith('罪') and len(accusation) > 1:
#         special_cases = ['犯罪所得', '犯罪分子', '經濟犯']
#         if not any(case in accusation for case in special_cases):
#             accusation = accusation[:-1]
#     return accusation

# def extract_criminal_charges(text):
#     """提取罪名（保留，選擇題模式不使用）"""
#     pattern = r'\[罪名\](.*?)<\s*eo[as]\s*>'
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         content = match.group(1).strip()
#         raw_charges = [x.strip() for x in re.split(r'[;；、,\uFF0C]+', content) if x.strip()]
#         normalized_charges = [normalize_accusation(charge) for charge in raw_charges]
#         return normalized_charges
#     return []

# def extract_sentence_number(text):
#     """提取刑期（月数）（保留，選擇題模式不使用）"""
#     pattern = r'\[刑期\](.*?)<\s*eo[as]\s*>'
#     match = re.search(pattern, text, re.DOTALL)
#     if not match:
#         return None
#     sentence_text = match.group(1).strip()
#     if "无期" in sentence_text or "無期" in sentence_text:
#         return "无期"
#     if "死刑" in sentence_text:
#         return "死刑"
#     if cn2an:
#         try:
#             sentence_text = cn2an.transform(sentence_text, "cn2an")
#         except:
#             pass
#     m2 = re.search(r'(\d+)\s*年\s*(\d+)\s*月', sentence_text)
#     if m2:
#         return int(m2.group(1))*12 + int(m2.group(2))
#     month_matches = re.findall(r"\d+个月", sentence_text)
#     if not month_matches:
#         month_matches = re.findall(r"\d+月", sentence_text)
#     if month_matches:
#         return int(re.findall(r"\d+", month_matches[0])[0])
#     year_matches = re.findall(r"\d+年", sentence_text)
#     if year_matches:
#         return int(re.findall(r"\d+", year_matches[0])[0]) * 12
#     return None

# def extract_crime_amount(text):
#     """提取涉案金額（保留，選擇題模式不使用）"""
#     pattern = r'\[(金额|金額)\](.*?)<\s*eo[as]\s*>'
#     match = re.search(pattern, text, re.DOTALL)
#     if not match:
#         return None
#     content = match.group(1).strip()
#     content = content.replace("元", "")
#     digits = re.findall(r"\d+\.?\d*", content)
#     if not digits:
#         return None
#     return float(digits[0])

def extract_correct_option(text):
    """
    提取正確選項（選擇題會用到）
    支援：A–E / a–e、以及（A）/(A) 格式
    """
    pattern = r'\[(正确答案|正確答案)\](.*?)<\s*eo[as]\s*>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    m = re.search(r'\b([A-Ea-e])\b', content)
    if not m:
        m = re.search(r'[（(]([A-Ea-e])[)）]', content)
    return m.group(1).upper() if m else None


################################ scoring (選擇題專用) ###############################

# def compute_f1_score(predicted, reference):
#     """保留（選擇題模式不使用）"""
#     predicted = [str(p) for p in predicted]
#     reference = [str(r) for r in reference]
#     predicted = [p.strip().lower() for p in predicted]
#     reference = [r.strip().lower() for r in reference]
#     true_positives = len(set(predicted) & set(reference))
#     precision = true_positives / len(predicted) if predicted else 0
#     recall = true_positives / len(reference) if reference else 0
#     if precision + recall == 0:
#         return 0
#     f1 = 2 * precision * recall / (precision + recall)
#     return f1


def scoring_function(prompts, completions, answer, **kwargs):
    """
    選擇題專用評分函數
    只評估多選題（[正确答案]/[正確答案]），其他任務暫停（註解保留）。
    """
    print("[DEBUG-REWARD] ===== 進入選擇題專用評分模式 =====")
    print(f"[DEBUG-REWARD] 收到 {len(prompts)} 個提示和 {len(completions)} 個回覆")

    # 收集 user 指令 & 模型回覆
    responses, instructions = [], []
    for prompt, completion in zip(prompts, completions):
        instr = ""
        for msg in prompt:
            if msg.get("role") == "user":
                instr = msg.get("content", "")
                break
        instructions.append(instr)
        responses.append(completion[0]['content'])

    all_scores = []
    for idx, (instruction, response) in enumerate(zip(instructions, responses)):
        print(f"\n[DEBUG-REWARD] 評估第 {idx+1} 個回覆 (選擇題模式)...")
        gold = answer[idx] if idx < len(answer) else answer[0]

        # 只做選擇題抽取與比對
        pred_opt = extract_correct_option(response)
        gold_opt = extract_correct_option(gold)
        print(f"正確選項: {gold_opt}")
        print(f"提取的選項: {pred_opt}")

        if pred_opt is None or gold_opt is None:
            score = 0.0
        elif pred_opt == gold_opt:
            score = 5.0
        else:
            score = 0.0

        all_scores.append(score)
        print(f"[DEBUG-REWARD] 多選題評分: {score:.3f}")

        # ------------------ 其餘任務暫停（保留供日後啟用） ------------------
        # # 刑期任務：
        # # if ...:  # extract_sentence_number(...)
        # # 法條任務：
        # # if ...:  # extract_law_articles(...)
        # # 罪名任務：
        # # if ...:  # extract_criminal_charges(...)
        # # 金額任務：
        # # if ...:  # extract_crime_amount(...)
        # -------------------------------------------------------------------

    print(f"\n所有選擇題評分: {[round(s, 3) for s in all_scores]}")
    return all_scores


############################ info-gain bonus（保留，作用於 MCQ 分數） ############################

def calculate_answer_diversity_bonus_v2(answer_token_info, baseline_answer_info, base_score):
    """
    計算增益獎勵（保留）
    """
    if not answer_token_info or 'avg_logit' not in answer_token_info:
        print("[INFO_GAIN] answer_token_info無效，使用原始分數")
        return base_score
    if not baseline_answer_info or 'avg_logit' not in baseline_answer_info:
        print("[INFO_GAIN] baseline_answer_info無效，使用原始分數")
        return base_score
    reasoning_logit = answer_token_info['avg_logit']
    direct_logit = baseline_answer_info['avg_logit']
    info_gain = reasoning_logit - direct_logit
    print(f"[INFO_GAIN] reasoning_logit={reasoning_logit:.4f}, direct_logit={direct_logit:.4f}, info_gain={info_gain:.4f}")
    info_gain_factor = torch.sigmoid(torch.tensor(info_gain/5)).item()
    print(f"[INFO_GAIN] info_gain_factor={info_gain_factor:.3f}")
    enhanced_score = base_score * info_gain_factor
    print(f"[INFO_GAIN] base_score={base_score:.3f}, info_gain_factor={info_gain_factor:.3f}, enhanced_score={enhanced_score:.3f}")
    return enhanced_score

def enhanced_scoring_function_v2(prompts, completions, answer,
                                 answer_token_info=None, baseline_answer_info=None, **kwargs):
    """
    增強版（選擇題專用）
    先用 MCQ 專用 scoring_function，若提供 token 資訊，再乘上 info-gain 係數。
    """
    print("[DEBUG-REWARD] ===== 進入增強選擇題評分函數 =====")
    base_scores = scoring_function(prompts, completions, answer, **kwargs)
    enhanced_scores = []
    for i, base_score in enumerate(base_scores):
        current_answer_info = answer_token_info[i] if (answer_token_info and i < len(answer_token_info)) else None
        current_baseline_info = baseline_answer_info[i] if (baseline_answer_info and i < len(baseline_answer_info)) else None
        if current_answer_info and current_baseline_info:
            enhanced_score = calculate_answer_diversity_bonus_v2(
                current_answer_info,
                current_baseline_info,
                base_score
            )
            print(f"[DEBUG-REWARD] 樣本{i+1}: 使用增益，基礎分數={base_score:.3f}, 增強分數={enhanced_score:.3f}")
        else:
            enhanced_score = base_score
            print(f"[DEBUG-REWARD] 樣本{i+1}: 缺少token，使用基礎分數={base_score:.3f}")
        enhanced_scores.append(enhanced_score)

    # 可選記錄
    try:
        import swanlab
        swanlab.log({
            "reward/mcq_base_score": np.mean(base_scores),
            "reward/mcq_enhanced_score": np.mean(enhanced_scores),
        })
    except:
        pass

    return enhanced_scores
