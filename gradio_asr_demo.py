"""
Gradio Web Demo for Real-time ASR.

This demo captures microphone audio, streams it to a realtime ASR backend
and displays recognition results in real-time on a web interface.

Current backend: DashScope fun-asr-realtime (see DASHSCOPE_API_KEY).
"""
import os
import threading
import queue
import time
from typing import Optional
import json
import pandas as pd

import gradio as gr
import pyaudio
from dotenv import load_dotenv

from asr.RealtimeFunASR import RealtimeFunASR
from pypinyin import lazy_pinyin
from rapidfuzz.distance import Levenshtein
from utils.log import logger
import re
from rapidfuzz import fuzz

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
    
# ===== 领域词 =====
DOMAIN_TERMS = get_domain_terms()

# ===== 强规则纠错 =====
CONFUSION_MAP = {
    "断信时": "锻信石",
    "断信石": "锻信石",
    "时刻": "十克"
}



CN_NUM = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10
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
    - 支持任意正整数（十 / 百 / 千 / 万 / 亿）
    - 不支持负数、小数（药方剂量场景足够）
    """
    total = 0          # 最终结果
    section = 0        # 当前小节（万 / 亿 以内）
    number = 0         # 当前数字

    for char in cn:
        if char in CN_NUM:
            number = CN_NUM[char]
        elif char in CN_UNIT:
            unit = CN_UNIT[char]
            if unit >= 10000:
                # 万、亿：直接结算一个 section
                section = (section + number) * unit
                total += section
                section = 0
            else:
                # 十、百、千
                if number == 0:
                    number = 1
                section += number * unit

            number = 0

    return total + section + number


# ===== 数字归一化 =====
def normalize_number(text: str) -> str:
    """
    将“克”前面的中文数字转换为阿拉伯数字。
    例：党参五十克 -> 党参50克
    """
    pattern = re.compile(
        r"(十|[一二两三四五六七八九]十?|十[一二三四五六七八九])"
        rf"(?=({UNIT_PATTERN}))"
    )


    def repl(match):
        cn = match.group()
        return str(chinese_to_arabic(cn))

    return pattern.sub(repl, text)


UNIT_NORMALIZE_MAP = {
    "毫克": "mg",
    "克": "g",
    "千克": "kg",
    "公斤": "kg",
    "袋": "袋",
}

UNIT_PATTERN = r"(?:mg|g|kg|袋)"

DIGIT_HERBS = [
    herb for herb in DOMAIN_TERMS
    if re.fullmatch(r"[一二三四五六七八九十]+", herb)
]


def is_valid_herb(token: str) -> tuple:
    """
    验证是否是有效药名（第2阶段精筛）
    返回: (是否是药名, 纠正后的药名)
    - 精确匹配词库 → True, 原名
    - 拼音模糊匹配成功（≥85分）→ True, 纠正后名称
    - 否则 → False, 原名
    """
    # 1. 精确匹配词库
    if token in DOMAIN_TERMS:
        return True, token

    # 2. 拼音模糊匹配
    token_py = "".join(lazy_pinyin(token))
    best = token
    best_score = 0

    for herb in DOMAIN_TERMS:
        herb_py = "".join(lazy_pinyin(herb))
        score = fuzz.ratio(token_py, herb_py)

        if score > best_score and score >= 85:
            best = herb
            best_score = score

    if best_score >= 85:
        return True, best

    return False, token


def normalize_dosage_number(num: str) -> str:
    """
    只处理剂量数字，不碰药名
    """
    if re.fullmatch(r"[零一二两三四五六七八九十]+", num):
        return str(chinese_to_arabic(num))
    return num


def postprocess_asr(asr_text: str) -> str:
    if not asr_text:
        return asr_text

    text = asr_text

    # ===== 1️⃣ 基础清洗 =====
    # text = text.replace("。", "")#.replace("，", " ")
    for zh, en in UNIT_NORMALIZE_MAP.items():
        text = text.replace(zh, en)
    text = re.sub(r"\s+", " ", text).strip()
    logger.info(f"after replace: {text}")

    # ===== 2️⃣ 硬规则纠错 =====
    for wrong, right in CONFUSION_MAP.items():
        text = text.replace(wrong, right)

    # ===== 3️⃣ 新匹配策略：先找剂量，再找前面的药名 =====

    # 定义标点符号（用于分隔药名片段）
    PUNCTUATION = r"[，。！？、；：""''（）【】《》]"

    # 定义剂量模式：数字 + 单位（可选）
    dosage_pattern = re.compile(
        rf"([零一二两三四五六七八九十百千万亿]+|\d+(?:\.\d+)?)"
        rf"\s*({UNIT_PATTERN})?"
    )

    results = []
    seen_positions = set()  # 记录已处理的位置
    logger.info(f"text: {text}")

    # 1. 找到所有剂量位置
    dosage_matches = list(dosage_pattern.finditer(text))
    logger.info(f"找到 {len(dosage_matches)} 个剂量候选")

    for match in dosage_matches:
        num = match.group(1)
        unit = match.group(2) if match.group(2) else ""
        dosage_start = match.start()
        dosage_end = match.end()

        logger.info(f"剂量候选 -> num={num}, unit={unit}, pos=({dosage_start},{dosage_end})")

        # 跳过已处理的位置
        if any(s <= dosage_start < e for s, e in seen_positions):
            logger.info(f"跳过(位置重叠)")
            continue

        # 2. 从剂量位置向前找最近的标点，以标点为界找药名
        # 从剂量位置向前扫描，找到标点或文本开头
        herb_start = dosage_start
        for i in range(dosage_start - 1, -1, -1):
            if re.match(PUNCTUATION, text[i]):
                herb_start = i + 1  # 标点后面的第一个字符开始是药名
                break
            herb_start = 0  # 没找到标点，从头开始

        herb_raw = text[herb_start:dosage_start].strip()
        logger.info(f"原始药名片段 -> '{herb_raw}' (pos={herb_start},{dosage_start})")

        # 3. 提取药名（取最后一段，忽略前面的标点噪声）
        # 从右向左找到最后一个标点，之前的是药名
        herb_match = re.search(rf"([\u4e00-\u9fa5\(\)·]{{2,10}})$", herb_raw)
        if herb_match:
            herb_raw = herb_match.group(1)
            logger.info(f"提取药名 -> '{herb_raw}'")

        # 4. 验证药名有效性
        if herb_raw and len(herb_raw) >= 2:
            is_valid, herb_corrected = is_valid_herb(herb_raw)
            logger.info(f"药名验证 -> herb_raw={herb_raw}, herb_corrected={herb_corrected}, is_valid={is_valid}")

            if is_valid:
                # 标记位置为已占用
                seen_positions.add((herb_start, dosage_end))

                # 中文数字转阿拉伯数字
                if re.fullmatch(r"[零一二两三四五六七八九十百千万亿]+", num):
                    num = str(chinese_to_arabic(num))

                unit = UNIT_NORMALIZE_MAP.get(unit, unit) if unit else unit
                results.append(f"{herb_corrected}{num}{unit}")
                logger.info(f"添加结果 -> {results[-1]}")
                continue

        # 5. 如果没有有效药名，但有单位，保留剂量+单位
        if unit:
            seen_positions.add((dosage_start, dosage_end))
            if re.fullmatch(r"[零一二两三四五六七八九十百千万亿]+", num):
                num = str(chinese_to_arabic(num))
            unit = UNIT_NORMALIZE_MAP.get(unit, unit)
            results.append(f"{num}{unit}")
            logger.info(f"无药名，保留剂量 -> {results[-1]}")

    logger.info(f"final results: {results}")

    # ===== 4️⃣ 兜底：如果没抽到结构 =====
    if not results:
        tokens = re.findall(
            rf"[\u4e00-\u9fa5]+|\d+(?:\.\d+)?\s*{UNIT_PATTERN}",
            text
        )
        corrected_tokens = []
        for tok in tokens:
            if re.search(r"\d", tok):
                corrected_tokens.append(tok)
            else:
                is_valid, corrected = is_valid_herb(tok)
                if is_valid:
                    corrected_tokens.append(corrected)
        return " ".join(corrected_tokens)

    res = " ".join(results)
    logger.info(f"res: {res}")
    return res


def microphone_chunks():
    """
    Generator yielding raw PCM chunks from default microphone.
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pa.get_format_from_width(SAMPLE_WIDTH),
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
    )
    try:
        while asr_running:
            yield stream.read(CHUNK_FRAMES, exception_on_overflow=False)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()



def on_asr_result(event: str, payload) -> None:
    """
    Callback for ASR results.
    """
    global current_text, intermediate_text
    
    if event == "intermediate":
        # 中间结果
        if isinstance(payload, str):
            try:
                payload_dict = json.loads(payload)
                text = payload_dict.get("payload", {}).get("result", "")
            except:
                text = str(payload)
        elif isinstance(payload, dict):
            text = payload.get("result", str(payload))
        else:
            text = str(payload)
        
        intermediate_text = text
        result_queue.put(("intermediate", text))
    
    elif event == "sentence_end":
        # 最终结果
        if isinstance(payload, str):
            try:
                payload_dict = json.loads(payload)
                text = payload_dict.get("payload", {}).get("result", "")
            except:
                text = str(payload)
        elif isinstance(payload, dict):
            text = payload.get("result", str(payload))
        else:
            text = str(payload)
        
        # 不在这里累加，交给 update_display 处理
        logger.info(f"before postasr： {text}")
        text = postprocess_asr(text)
        result_queue.put(("final", text))
    
    elif event == "error":
        error_msg = f"错误: {payload}"
        result_queue.put(("error", error_msg))
        logger.info(f"ASR Error: {payload}")


def start_asr():
    """
    Start ASR recognition in a background thread.
    """
    global asr_running, asr_thread
    
    if asr_running:
        return "ASR已经在运行中..."
    
    asr_running = True
    
    def worker():
        global asr_running
        try:
            asr = RealtimeFunASR()
            for _ in asr.run_stream(microphone_chunks(), on_result=on_asr_result):
                if not asr_running:
                    break
        except Exception as e:
            result_queue.put(("error", f"ASR异常: {str(e)}"))
            logger.info(f"ASR worker error: {e}")
        finally:
            asr_running = False
    
    asr_thread = threading.Thread(target=worker, daemon=True)
    asr_thread.start()
    return "ASR已启动，正在监听麦克风..."


def stop_asr():
    """
    Stop ASR recognition.
    """
    global asr_running, current_text, intermediate_text
    
    asr_running = False
    current_text = ""
    intermediate_text = ""
    
    # Clear queue
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except queue.Empty:
            break
    
    return "ASR已停止"


def get_latest_results():
    """
    Get latest recognition results from queue.
    """
    results = []
    while not result_queue.empty():
        try:
            event, text = result_queue.get_nowait()
            results.append((event, text))
        except queue.Empty:
            break
    
    return results


def update_display():
    """
    Update the display with latest results.
    中间结果展示后，最终结果覆盖中间结果，并且每次最终结果不换行，累加在后面。
    """
    global current_text, intermediate_text
    
    # Get new results
    results = get_latest_results()
    
    for event, text in results:
        if event == "final":
            # 最终结果：累加到 current_text，不换行，并清除中间结果
            sep = "，"  # 或者改成 "，"
            current_text += (text + sep) if text else ""
            intermediate_text = ""  # 最终结果覆盖中间结果
        elif event == "intermediate":
            # 中间结果：临时显示，会被最终结果覆盖
            intermediate_text = text
        elif event == "error":
            current_text += f"\n{text}\n"
    
    # Format display with HTML
    if not current_text and not intermediate_text:
        return "<p style='color: #666;'>等待开始识别...</p>"
    
    display_html = "<div style='font-size: 16px; line-height: 1.6;'>"
    
    # 先显示已确认的最终结果
    if current_text:
        display_html += f"<div style='margin-bottom: 10px;'>{current_text}</div>"
    
    # 如果有中间结果，显示在最终结果后面（灰色斜体，表示临时）
    if intermediate_text:
        display_html += f"<span style='color: #888; font-style: italic;'>{intermediate_text}</span>"
    
    display_html += "</div>"
    
    return display_html


def create_interface():
    """
    Create Gradio interface.
    """
    with gr.Blocks(title="实时语音识别 Demo") as demo:
        gr.Markdown("""
        # 🎤 实时语音识别 Demo
        
        基于阿里云实时语音识别服务的 Web Demo。
        
        **使用说明：**
        1. 点击"开始识别"按钮开始录音
        2. 对着麦克风说话，识别结果会实时显示
        3. 点击"停止识别"按钮停止录音
        
        **注意：** 需要先设置环境变量 `ALIYUN_ASR_TOKEN` 和 `ALIYUN_ASR_APPKEY`
        """)
        
        with gr.Row():
            with gr.Column():
                start_btn = gr.Button("开始识别", variant="primary", size="lg")
                stop_btn = gr.Button("停止识别", variant="stop", size="lg")
                status_text = gr.Textbox(
                    label="状态",
                    value="未启动",
                    interactive=False
                )
        
        with gr.Row():
            output_text = gr.HTML(
                label="识别结果",
                value="<p style='color: #666;'>等待开始识别...</p>"
            )

        # Button events (use generator streaming instead of `every=`,
        # because older gradio versions don't support it).
        def start_and_stream():
            status = start_asr()
            # first paint
            yield status, update_display()
            # stream updates
            while asr_running:
                time.sleep(0.2)
                yield status, update_display()
            # final paint after stop
            yield "ASR已停止", update_display()

        def stop_and_refresh():
            status = stop_asr()
            return status, update_display()

        start_btn.click(
            fn=start_and_stream,
            inputs=None,
            outputs=[status_text, output_text],
        )

        stop_btn.click(
            fn=stop_and_refresh,
            inputs=None,
            outputs=[status_text, output_text],
        )
    
    return demo


def main():
    """
    Launch Gradio demo.
    """
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.info("错误: 请先设置环境变量 DASHSCOPE_API_KEY")
        logger.info("示例:")
        logger.info("  export DASHSCOPE_API_KEY='your_dashscope_api_key'")
        return

    demo = create_interface()
    
    # Launch with sharing disabled by default
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

# if __name__ == "__main__":
#     main()


if __name__ == "__main__":

    text = postprocess_asr("三七粉的嗯复活机会15克")
    text = postprocess_asr("三七一百克")
    text = postprocess_asr("糖梨根30g，就怎么那个逻辑啊，都在一个方法里面啊，黄芩片30g。这不是缺损，这是找不到了，往上往上out对，就这个out out还烦点了，点我我不是看那个啊。")