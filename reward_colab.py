#!/usr/bin/env python3
"""
🎯 GRPO 奖励函数（Colab 专用版本）
适用于 Legal Delta 法律问题训练

包含的奖励函数：
1. xmlcount_reward_func - XML 格式检查
2. enhanced_scoring_function_v2 - 法律任务评分（刑期、法条、罪名等）
"""

import re
import itertools
import numpy as np
import math
from typing import List
import torch

# 如果需要中文数字转换，可选安装：pip install cn2an
try:
    import cn2an 
except ImportError:
    print("⚠️  cn2an 未安装，部分功能可能受限")
    cn2an = None

############################format reward##############################


def count_xml(text) -> float:
    """
    计算 XML 标签的正确性得分
    
    Args:
        text: 模型输出文本
        
    Returns:
        float: 得分 (0.0 - 1.0)
    """

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
    """
    XML 格式奖励函数
    
    Args:
        completions: 模型生成的完成列表
        
    Returns:
        List[float]: 每个完成的奖励分数
    """
    contents = [completion[0]["content"] for completion in completions]
    scores = [count_xml(c) for c in contents]
    for i, score in enumerate(scores):
        print(f"內容 {i+1} XML格式分數: {score:.3f}")

    return [count_xml(c) for c in contents]


################################reward###############################

def extract_articles(text):
    """提取法条编号"""
    matches = re.findall(r'第(\d+(?:-\d+)?)條', text)
    return matches

def extract_pairs(text):
    """提取被告-罪名对"""
    return re.findall(r'\(([^)]+)\)犯\(([^)]+)\)', text)

def extract_penalties(text):
    """提取刑期信息"""
    penalties = {}
    defendants = re.findall(r'\(([^)]+)\)犯\(([^)]+)\)', text)
    for name, _ in defendants:
        pattern = fr'\({re.escape(name)}\).*?有期徒刑時長：(\d+)\)'
        match = re.search(pattern, text)
        if match:
            fixed_term = match.group(1)
            penalties[name] = int(fixed_term)
    return penalties

def calculate_term_score(predicted: int, actual: int) -> float:
    """
    计算刑期预测得分
    
    Args:
        predicted: 预测的刑期（月）
        actual: 实际的刑期（月）
        
    Returns:
        float: 得分 (0.0 - 1.0)
    """
    if predicted == actual:
        return 1.0
    if actual == 0:
        return 0.0 if predicted > 0 else 1.0
    if predicted == 0:
        return 0.0
    
    error_ratio = abs(predicted - actual) / actual
    penalty_factor = 1.2 if predicted > actual else 1.0
    score = max(0, 1 - error_ratio * penalty_factor)
    return score


def penalty_reward(response: str, answer: str) -> float:
    """刑期预测奖励函数"""
    if not response or not answer:
        return 0.0
    
    resp_penalties = extract_penalties(response)
    ans_penalties = extract_penalties(answer)
    
    if not resp_penalties or not ans_penalties:
        return 0.0
    
    total_score = 0
    total_defendants = len(ans_penalties)
    
    for name, ans_term in ans_penalties.items():
        if name in resp_penalties:
            resp_term = resp_penalties[name]
            score = calculate_term_score(resp_term, ans_term)
            print(f"\n{name} 刑期評分詳情:")
            print(f"有期徒刑: 預測={resp_term}, 實際={ans_term}, 得分={score:.3f}")
            total_score += score
    
    return total_score / total_defendants if total_defendants > 0 else 0

def expand_brackets(text):
    """扩展方括号表达式"""
    while "[" in text and "]" in text:
        match = re.search(r"\[(.*?)\]", text)
        if not match:
            break
        options = match.group(1).split("、")
        expanded = [text[:match.start()] + opt + text[match.end():] for opt in options]
        text = expanded  
    if isinstance(text, list):
        return list(itertools.product(*[t.split("、") if "、" in t else [t] for t in text]))
    
    return [text]

def extract_law_articles(text):
    """
    提取法条编号
    
    Args:
        text: 文本内容
        
    Returns:
        List[int]: 法条编号列表
    """
    pattern = r'\[(?:法条|法條)\](?P<content>.*?)<\s*eo[as]\s*>'
    match = re.search(pattern, text, re.DOTALL)
    content = match.group('content').strip()

        
    content = match.group(1).strip()
    
    prediction_law_chunks = re.split(r'[,，、\s]+', content)
    prediction_law_index_digit_list = []
    
    for prediction_law_chunk in prediction_law_chunks:
        if not prediction_law_chunk:
            continue
        prediction_law_chunk = prediction_law_chunk.replace("萬元", "元")
        prediction_law_chunk = re.sub(r'第(.*?)款', "", prediction_law_chunk)
        prediction_law_chunk = re.sub(r'第(.*?)條', lambda m: m.group(1), prediction_law_chunk)
        
        # 使用 cn2an 转换中文数字（如果可用）
        if cn2an:
            try:
                prediction_law_chunk = cn2an.transform(prediction_law_chunk, "cn2an")
            except:
                pass 
        
        prediction_law_section_numbers = re.findall(r"\d+", prediction_law_chunk)
        if len(prediction_law_section_numbers) == 0:
            continue
        prediction_law_index_digit = int(prediction_law_section_numbers[0])
        if prediction_law_index_digit <= 490:
            prediction_law_index_digit_list.append(prediction_law_index_digit)
    
    return prediction_law_index_digit_list

def normalize_accusation(accusation):
    """规范化罪名"""
    accusation = accusation.strip()
    if accusation.endswith('罪') and len(accusation) > 1:
        special_cases = ['犯罪所得', '犯罪分子', '經濟犯']
        if not any(case in accusation for case in special_cases):
            accusation = accusation[:-1]
    return accusation

def extract_criminal_charges(text):
    """
    提取罪名
    
    Args:
        text: 文本内容
        
    Returns:
        List[str]: 罪名列表
    """
    pattern = r'\[罪名\](.*?)<\s*eo[as]\s*>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        raw_charges = [x.strip() for x in re.split(r'[;；、,\uFF0C]+', content) if x.strip()]
        normalized_charges = [normalize_accusation(charge) for charge in raw_charges]
        return normalized_charges
    return []

def extract_sentence_number(text):
    """
    提取刑期（月数）
    
    Args:
        text: 文本内容
        
    Returns:
        int or str: 刑期（月数）或 "无期"/"死刑"
    """
    pattern = r'\[刑期\](.*?)<\s*eo[as]\s*>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
        
    sentence_text = match.group(1).strip()
    if "无期" in sentence_text:
        return "无期"
    if "死刑" in sentence_text:
        return "死刑"
    
    # 使用 cn2an 转换中文数字（如果可用）
    if cn2an:
        try:
            sentence_text = cn2an.transform(sentence_text, "cn2an")
        except:
            pass
    
    month_matches = re.findall(r"\d+个月", sentence_text)
    if not month_matches:
        month_matches = re.findall(r"\d+月", sentence_text)
    
    if month_matches:
        return int(re.findall(r"\d+", month_matches[0])[0])
    year_matches = re.findall(r"\d+年", sentence_text)
    if year_matches:
        return int(re.findall(r"\d+", year_matches[0])[0]) * 12
    
    return None

def extract_crime_amount(text):
    """
    提取涉案金额
    
    Args:
        text: 文本内容
        
    Returns:
        float: 金额
    """
    pattern = r'\[(金额|金額)\](.*?)<\s*eo[as]\s*>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
        
    content = match.group(1).strip()
    content = content.replace("元", "")
    digits = re.findall(r"\d+\.?\d*", content)
    if not digits:
        return None
    return float(digits[0])

def extract_correct_option(text):
    """
    提取正确选项
    
    Args:
        text: 文本内容
        
    Returns:
        str: 选项字母 (A-E)
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



def compute_f1_score(predicted, reference):
    """
    计算 F1 分数
    
    Args:
        predicted: 预测列表
        reference: 参考列表
        
    Returns:
        float: F1 分数 (0.0 - 1.0)
    """
    predicted = [str(p) for p in predicted]
    reference = [str(r) for r in reference]
    predicted = [p.strip().lower() for p in predicted]
    reference = [r.strip().lower() for r in reference]
    true_positives = len(set(predicted) & set(reference))
    precision = true_positives / len(predicted) if predicted else 0
    recall = true_positives / len(reference) if reference else 0
    if precision + recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def scoring_function(prompts, completions, answer, **kwargs):
    """
    基础评分函数
    
    根据任务类型（刑期、法条、罪名、金额、选项）评分
    
    Args:
        prompts: 提示列表
        completions: 完成列表
        answer: 正确答案列表
        
    Returns:
        List[float]: 评分列表 (0.0 - 5.0)
    """
    print("[DEBUG-REWARD] ===== 进入评分函数 =====")
    print(f"[DEBUG-REWARD] 收到 {len(prompts)} 个提示和 {len(completions)} 个回复")
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx == 0:
            print("\n--------- 当前批次样本信息 ---------")
            content = ""
            for msg in prompt:
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    break
            id_match = re.search(r"\[QUERY_ID:(.*?)\]", content)
            sample_id = id_match.group(1) if id_match else "未找到ID"
            print(f"样本ID: {sample_id}")
            print(f"Prompt包含 {len(prompt)} 条消息")
            for i, msg in enumerate(prompt):
                role = msg.get('role', 'unknown')
                msg_content = msg.get('content', '')
                print(f"  消息{i+1} ({role}): {msg_content[:100]}...")  # 只显示前100个字符
            
            print("-------------------------------------")
            break

    ###############################################################
    responses = []
    instructions = []
    for prompt_idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        current_instruction = None
        for msg in prompt:
            if msg.get("role") == "user":
                current_instruction = msg.get("content", "")
                break
        if current_instruction:
            instructions.append(current_instruction)
        else:
            instructions.append("")  
        responses.append(completion[0]['content'])
        
    all_scores = []
    for idx, (instruction, response) in enumerate(zip(instructions, responses)):
        print(f"\n[DEBUG-REWARD] 评估第 {idx+1} 个回复...")
        current_answer = answer[idx] if idx < len(answer) else answer[0]
        
        # 刑期任务
        if "包含你对案件事实、罪名和法条的分析过程，详细说明你如何得出最终刑期。" in instruction:
            correct_sentence = extract_sentence_number(current_answer)
            extracted_sentence = extract_sentence_number(response)
            print(f"正确刑期: {correct_sentence}")
            print(f"提取的刑期: {extracted_sentence}")
            
            if extracted_sentence is None or correct_sentence is None:
                score = 0.0
            elif extracted_sentence == correct_sentence:
                score = 5.0
            elif correct_sentence in ["无期", "死刑"] or extracted_sentence in ["无期", "死刑"]:
                score = 0.0
            else:
                log_distance = abs(math.log(correct_sentence + 1) - math.log(extracted_sentence + 1))
                max_log_distance = math.log(36)  
                score = 5.0 * (max_log_distance - log_distance) / max_log_distance
                score = max(0.0, score)
            
            all_scores.append(score)
            print(f"[DEBUG-REWARD] 刑期评分: {score:.3f}")
        
        # 法条任务
        elif "包含你对案件事实和罪名的分析过程，详细说明你如何得出涉及的刑法法条。" in instruction:
            correct_laws = extract_law_articles(current_answer)
            extracted_laws = extract_law_articles(response)
            
            print(f"正确法条: {correct_laws}")
            print(f"提取的法条: {extracted_laws}")
            
            f1_score = compute_f1_score(extracted_laws, correct_laws)
            score = round(f1_score * 5, 3)
            
            all_scores.append(score)
            print(f"[DEBUG-REWARD] 法条评分: {score:.3f}")
        
        # 罪名任务
        elif "包含你对案件事实的分析过程，详细说明你如何得出最终罪名。" in instruction:
            correct_charges = extract_criminal_charges(current_answer)
            extracted_charges = extract_criminal_charges(response)
            
            print(f"正确罪名: {correct_charges}")
            print(f"提取的罪名: {extracted_charges}")
            
            f1_score = compute_f1_score(extracted_charges, correct_charges)
            score = round(f1_score * 5, 3)
            
            all_scores.append(score)
            print(f"[DEBUG-REWARD] 罪名评分: {score:.3f}")
        
        # 金额任务
        elif "包含你对案件事实的分析过程，详细说明你对案件文书中提及的所有涉案金额的计算过程。" in instruction:
            extracted_amount = extract_crime_amount(response)
            correct_amount = extract_crime_amount(current_answer)

            print(f"正确金额: {correct_amount}")
            print(f"提取的金额: {extracted_amount}")
            
            if extracted_amount is None:
                score = 0.0
            elif correct_amount is None:
                score = 0.0
            elif extracted_amount == correct_amount:
                score = 5.0
            else:
                score = 0.0
            
            all_scores.append(score)
            print(f"[DEBUG-REWARD] 金额评分: {score:.3f}")
        
        # 选择题任务
        elif "包含你对法律问题的分析过程，详细解释为什么选择特定选项作为答案。" in instruction:
            extracted_option = extract_correct_option(response)
            correct_option = extract_correct_option(current_answer)
            print(f"正确选项: {correct_option}")
            print(f"提取的选项: {extracted_option}")
            
            if extracted_option is None:
                score = 0.0
            elif correct_option is None:
                score = 0.0
            elif extracted_option == correct_option:
                score = 5.0
            else:
                score = 0.0
            
            all_scores.append(score)
            print(f"[DEBUG-REWARD] 多选题评分: {score:.3f}")
        
        # 未知任务
        else:
            score = 0.0
            all_scores.append(score)
            print(f"[DEBUG-REWARD] instruction有问题,评分: {score:.3f}")
            
    for i, response in enumerate(responses):
        score = all_scores[i] if i < len(all_scores) else 0.0
        print("\n" + "="*50)
        print(f"模型回复 #{i+1} (评分: {score:.3f}):")
        print("-"*50)
        print(response[:500])  # 只显示前500个字符
        if len(response) > 500:
            print("... (省略部分输出)")
        print("-"*50)
        print(f"参考答案:")
        print("-"*50)
        current_answer = answer[i] if i < len(answer) else answer[0]
        print(current_answer[:500])
        if len(current_answer) > 500:
            print("... (省略部分输出)")
        print("="*50)

    print(f"\n所有评分: {[round(s, 3) for s in all_scores]}")
    
    return all_scores

def calculate_answer_diversity_bonus_v2(answer_token_info, baseline_answer_info, base_score):
    """
    计算信息增益奖励
    
    Args:
        answer_token_info: 推理回答的 token 信息
        baseline_answer_info: 基准回答的 token 信息
        base_score: 基础分数
        
    Returns:
        float: 增强后的分数
    """
    if not answer_token_info or 'avg_logit' not in answer_token_info:
        print("[INFO_GAIN] answer_token_info无效，使用原始分数")
        return base_score
    
    if not baseline_answer_info or 'avg_logit' not in baseline_answer_info:
        print("[INFO_GAIN] baseline_answer_info无效，使用原始分数")
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

def enhanced_scoring_function_v2(prompts, completions, answer, answer_token_info=None, baseline_answer_info=None, **kwargs):
    """
    增强评分函数 v2
    
    在基础评分的基础上，考虑信息增益（推理过程的价值）
    
    Args:
        prompts: 提示列表
        completions: 完成列表
        answer: 正确答案列表
        answer_token_info: 推理回答的 token 信息
        baseline_answer_info: 基准回答的 token 信息
        
    Returns:
        List[float]: 增强后的评分列表
    """
    print("[DEBUG-REWARD] ===== 进入增强评分函数v2 =====")
    print(f"[DEBUG-REWARD] answer_token_info是否为None: {answer_token_info is None}")
    print(f"[DEBUG-REWARD] baseline_answer_info是否为None: {baseline_answer_info is None}")
    
    if answer_token_info:
        print(f"[DEBUG-REWARD] answer_token_info长度: {len(answer_token_info)}")
    if baseline_answer_info:
        print(f"[DEBUG-REWARD] baseline_answer_info长度: {len(baseline_answer_info)}")
    
    base_scores = scoring_function(prompts, completions, answer, **kwargs)
    enhanced_scores = []
    
    for i, base_score in enumerate(base_scores):
        current_answer_info = answer_token_info[i] if answer_token_info and i < len(answer_token_info) else None
        current_baseline_info = baseline_answer_info[i] if baseline_answer_info and i < len(baseline_answer_info) else None
        
        if current_answer_info and current_baseline_info:
            enhanced_score = calculate_answer_diversity_bonus_v2(
                current_answer_info, 
                current_baseline_info, 
                base_score
            )
            print(f"[DEBUG-REWARD] 样本{i+1}: 使用信息增益，基础分数={base_score:.3f}, 增强分数={enhanced_score:.3f}")
        else:
            enhanced_score = base_score
            print(f"[DEBUG-REWARD] 样本{i+1}: 缺少token信息，使用基础分数={base_score:.3f}")
        
        enhanced_scores.append(enhanced_score)
    
    # 可选：记录到 swanlab（如果安装）
    try:
        import swanlab
        swanlab.log({
            "reward/law_base_score": np.mean(base_scores),
            "reward/law_enhanced_score": np.mean(enhanced_scores),
        })
    except:
        pass  
    
    return enhanced_scores

