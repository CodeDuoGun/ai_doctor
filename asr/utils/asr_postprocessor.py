import threading
import queue
from typing import Optional
import pandas as pd

from dotenv import load_dotenv
from pypinyin import lazy_pinyin
from utils.log import logger
import re
from rapidfuzz import fuzz, process

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes
CHUNK_FRAMES = 320  # 320 frames * 2 bytes = 640 bytes per chunk

# Global state
asr_thread: Optional[threading.Thread] = None
asr_running = False
result_queue = queue.Queue()
current_text = ""
intermediate_text = ""

def get_domain_terms():
    file = "data/医保中草药数据_共11071条.csv"
    df = pd.read_csv(file)
    return df["名称"].values

def load_values_from_txt(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
    
# ===== 领域词 =====
DOMAIN_TERMS = load_values_from_txt("asr/hotwords.txt")

# ===== 强规则纠错 =====
CONFUSION_MAP = {
    "断信时": "锻信石",
    "断信石": "锻信石",
    "时刻": "10克"
}

CN_NUM = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


CN_UNIT = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
    "亿": 100000000,
}



def chinese_to_arabic(cn: str) -> int:
    """
    中文数字 → 阿拉伯数字
    支持：十 / 百 / 千 / 万 / 亿
    """
    total = 0      # 最终结果
    section = 0    # 万 / 亿 以内的结果
    number = 0     # 当前数字

    for char in cn:
        if char in CN_NUM:
            number = CN_NUM[char]

        elif char in CN_UNIT:
            unit = CN_UNIT[char]

            if unit < 10000:
                # 十、百、千
                if number == 0:
                    number = 1
                section += number * unit

            else:
                # 万、亿
                section = (section + number) * unit
                total += section
                section = 0

            number = 0

    return total + section + number


UNIT_NORMALIZE_MAP = {
    "毫克": "mg",
    "克": "g",
    "千克": "kg",
    "公斤": "kg",
    "袋": "袋",
    "条": "条",
    "瓶": "瓶",
    "对": "对",
    "朵": "朵",
    "个": "个"
}

UNIT_PATTERN = r"(?:mg|g|kg|袋|条|瓶|对|朵|个)"

DIGIT_HERBS = [
    herb for herb in DOMAIN_TERMS
    if re.fullmatch(r"[一二三四五六七八九十]+", herb)
]


def extract_herbs_with_dose(text: str, domain_terms: list[str]) -> list[str]:
    """
    从文本中抽取药材 + 剂量 + 单位
    - text: 分段后的 ASR 文本
    - domain_terms: 药材词库
    """
    results = []

    # ===== 1️⃣ 模糊匹配药名 =====
    # 分词或者按空格 / 中文字符切分
    tokens = re.findall(r"[\u4e00-\u9fa5]+", text)

    for token in tokens:
        # rapidfuzz.extractOne 返回 (best_match, score, index)
        match, score, _ = process.extractOne(
            token, domain_terms, scorer=fuzz.ratio
        )
        # 设置阈值，比如 80 分以上才认为是有效药材
        if score >= 80:
            herb = correct_herb_by_pinyin(match)

            # ===== 2️⃣ 尝试抽取剂量 + 单位 =====
            # 找 token 中的数字 + 单位
            dose_match = re.search(
                rf"(\d+(?:\.\d+)?)\s*({UNIT_PATTERN})?", text
            )
            if dose_match:
                num = dose_match.group(1)
                unit = dose_match.group(2) or ""
                unit = UNIT_NORMALIZE_MAP.get(unit, unit)
                results.append(f"{herb}{num}{unit}")
            else:
                # 没有剂量，也可以只保留药材名
                results.append(herb)

    return results
def correct_herb_by_pinyin(token: str) -> str:
    token_py = "".join(lazy_pinyin(token))
    # 若未匹配到，返空
    best = token
    best_score = 0

    for herb in DOMAIN_TERMS:
        herb_py = "".join(lazy_pinyin(herb))
        score = fuzz.ratio(token_py, herb_py)

        if score > best_score and score >= 85:
            best = herb
            best_score = score

    return best

def correct_herb_by_pinyin_v2(text: str) -> str:
    """
    在文本中匹配最接近的药材名称
    - text: ASR 输出片段
    - 返回最匹配的药材名称，如果匹配不到返回 ""
    
    优先规则：
    1️⃣ 文本长度与药材表名称长度一致且拼音完全匹配 → 直接返回
    2️⃣ 否则用滑动窗口 + 拼音相似度（score >= 85）匹配
    """
    if not text:
        return ""
    
    text_len = len(text)
    best_score = 0
    best_herb = ""

    # 药材表最长长度
    max_len = max(len(herb) for herb in DOMAIN_TERMS)
    # 原文包含完整药名（最高优先级）
    if text in DOMAIN_TERMS:
        return text

    # ----- 1️⃣ 长度完全一致优先匹配 -----
    for herb in DOMAIN_TERMS:
        if len(herb) == text_len:
            # 拼音匹配
            text_py = "".join(lazy_pinyin(text))
            herb_py = "".join(lazy_pinyin(herb))
            score = fuzz.ratio(text_py, herb_py)
            if score >= 85:
                return herb  # 完全长度匹配优先返回

    # ----- 2️⃣ 滑动窗口倒序匹配 -----
    for window_size in range(max_len, 0, -1):
        for i in range(text_len - window_size + 1):
            candidate = text[i:i+window_size]  # ASR片段
            candidate_py = "".join(lazy_pinyin(candidate))

            for herb in DOMAIN_TERMS:
                herb_py = "".join(lazy_pinyin(herb))
                score = fuzz.ratio(candidate_py, herb_py)

                if score > best_score and score >= 85:
                    best_score = score
                    best_herb = herb

        if best_herb:
            return best_herb

    return ""  # 没匹配到


# PUNCT_SPLIT_PATTERN = r"[，。；、,.]"
PUNCT_SPLIT_PATTERN = r"[，；、,.]"

def split_by_punctuation(text: str) -> list[str]:
    """
    按中文/英文标点切分 ASR 文本
    """
    text = text.replace("。", "").replace("\n", "，")
    segments = re.split(PUNCT_SPLIT_PATTERN, text)
    return [seg.strip() for seg in segments if seg.strip()]


def process_single_segment(segment: str) -> list[str]:
    text = segment

    # ===== 0 硬规则纠错 =====
    for wrong, right in CONFUSION_MAP.items():
        text = text.replace(wrong, right)

    # ===== 1 三七特判（必须最先）=====
    text = handle_sanqi_special(text)
    logger.debug(f"sanqi res {text}")

    # ===== 2 段内全量中文数字 → 阿拉伯数字  =====
    text = replace_zh_num_with_arabic(text)
    logger.debug(f"num res {text}")

    # ===== 3 基础清洗 =====
    for zh, en in UNIT_NORMALIZE_MAP.items():
        text = text.replace(zh, en)
    text = re.sub(r"\s+", " ", text).strip()



    # results = extract_herbs_with_dose(text, DOMAIN_TERMS)
    # ===== 4️⃣ 构造药名 Pattern =====
    HERB_PATTERN = "|".join(
        re.escape(term)
        for term in sorted(DOMAIN_TERMS, key=len, reverse=True)
    )

    # ===== 5️⃣ 抽取【药名 + 剂量 + 单位】=====
    pattern = re.compile(
        rf"({HERB_PATTERN})"
        rf"[^\d]*?"
        rf"(\d+(?:\.\d+)?)"
        rf"\s*({UNIT_PATTERN})"
    )
    pattern = re.compile(r"([\u4e00-\u9fa5]+)?(\d+(?:\.\d+)?)(\s*%s)?" % UNIT_PATTERN)

    results = []
    for match in pattern.finditer(text):
        herb = match.group(1) or ""
        num = match.group(2)
        unit = match.group(3) or ""
        logger.info(f"before correct: {herb}, {num}, {unit}")
        herb = correct_herb_by_pinyin_v2(herb)
        if not herb:
            continue
        logger.info(f"after correct_herb_by_pinyin_v2: {herb}, {num}, {unit}")
        unit = UNIT_NORMALIZE_MAP.get(unit, unit)
        results.append(f"{herb}{num}{unit}")

    return results


ZH_NUM_PATTERN = re.compile(r"[零一二两三四五六七八九十百千万亿]+")

def protect_domain_terms(text: str, terms: list[str]):
    """
    用占位符保护药名，防止内部中文数字被误替换
    """
    placeholders = {}
    for i, term in enumerate(sorted(terms, key=len, reverse=True)):
        placeholder = f"__TERM_{i}__"
        if term in text:
            text = text.replace(term, placeholder)
            placeholders[placeholder] = term
    return text, placeholders


def restore_domain_terms(text: str, placeholders: dict):
    for placeholder, term in placeholders.items():
        text = text.replace(placeholder, term)
    return text


def replace_zh_num_with_arabic(text: str) -> str:
    """
    中文数字 → 阿拉伯数字（带药名保护）
    """

    # ===== 1️⃣ 三七 + 中文数字（优先处理）=====
    text = handle_sanqi_special(text)

    # ===== 2️⃣ 保护药名（三七、五味子、醋三棱等）=====
    text, placeholders = protect_domain_terms(text, DOMAIN_TERMS)

    # ===== 3️⃣ 剩余中文数字 → 阿拉伯数字 =====
    def _repl(match):
        return str(chinese_to_arabic(match.group()))

    text = re.sub(ZH_NUM_PATTERN, _repl, text)

    # ===== 4️⃣ 还原药名 =====
    text = restore_domain_terms(text, placeholders)

    return text



SANQI_PATTERN = re.compile(
    rf"(三七)(\s*)({ZH_NUM_PATTERN})"
)

def handle_sanqi_special(text: str) -> str:
    """
    三七十五克 → 三七15克
    只转换三七后面的中文数字
    """
    def _repl(match):
        herb = match.group(1)          # 三七
        space = match.group(2) or ""
        zh_num = match.group(3)        # 十五
        print(f"zh_num: {zh_num}")
        arabic = chinese_to_arabic(zh_num)
        return f"{herb}{space}{arabic}"

    return SANQI_PATTERN.sub(_repl, text)

def postprocess_asr(asr_text: str) -> str:
    if not asr_text:
        return asr_text

    # ===== 0️⃣ 按标点切分 =====
    logger.info(f"asr_text: {asr_text}")
    segments = split_by_punctuation(asr_text)
    logger.info(f"segments: {segments}")

    all_results = []

    for seg in segments:
        seg_results = process_single_segment(seg)

        # ✅ 只保留“抽到有效药材”的段
        if seg_results:
            all_results.extend(seg_results)

    # ===== 兜底策略（可选）=====
    # if not all_results:
    #     # 你原来的兜底逻辑也可以放在这里
    #     tokens = re.findall(
    #         rf"[\u4e00-\u9fa5]+|\d+(?:\.\d+)?\s*{UNIT_PATTERN}",
    #         asr_text
    #     )
    #     corrected_tokens = []
    #     for tok in tokens:
    #         if re.search(r"\d", tok):
    #             corrected_tokens.append(tok)
    #         else:
    #             corrected_tokens.append(correct_herb_by_pinyin(tok))
    #     res = ",".join(corrected_tokens)
    #     logger.info(f"FINAL res {res}")
    #     return res 
    res = "，".join(all_results)
    
    logger.info(f"FINAL res {res}")
    return res


# if __name__ == "__main__":

    # text = postprocess_asr("片姜黄、黄芪、薄荷")
    # text = postprocess_asr("三七粉的嗯复活机会15克")
    # text = postprocess_asr("三七十五克 黄芪20克")

    # text = postprocess_asr("党参的嗯复活机会十五克")
    # text = postprocess_asr("三七一百克")
    # text = postprocess_asr("这个是大树皮6克")
    # text = postprocess_asr("疼梨根30g，就怎么那个逻辑啊，都在一个方法里面啊，黄芩片30g。这不是缺损，这是找不到了，往上往上out对，就这个out out还烦点了，点我我不是看那个啊。")
    # text = postprocess_asr("炒苦杏仁6克，黄破8克，大黄5克，火麻仁9克，陈皮10克，鱼腥草11克，浙贝母9克，叶干。")
    # text = postprocess_asr("龙胆时刻。炒栀子6克，黄芩12克，黄连6克，淡竹叶8克，通草3克，盐车前子8克，泽泻8克。鸡骨草10克，桃仁9克，红花8克，当归9克，丹参12克，生地黄32克。地龙五六克。至少八个。烫水蛭二克，瓜蒌子十克。呃，线下。薤白六克。醋延胡索9克。对。炒川楝子4克，石菖蒲9克，竹茹8克，清半夏3克，醋香附5克，白芍12克，醋柴胡10克，制远志9克。茯神9克，仙鹤草12克。炒酸枣仁18克，炒苦杏仁4克。那对。")