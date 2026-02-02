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


def correct_herb_by_pinyin(token: str) -> str:
    token_py = "".join(lazy_pinyin(token))
    best = token
    best_score = 0

    for herb in DOMAIN_TERMS:
        herb_py = "".join(lazy_pinyin(herb))
        score = fuzz.ratio(token_py, herb_py)

        if score > best_score and score >= 85:
            best = herb
            best_score = score

    return best


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

    # ===== 3️⃣ 构造药名 Pattern（按长度倒序，防止短词抢匹配）=====
    HERB_PATTERN = "|".join(
        re.escape(term)
        for term in sorted(DOMAIN_TERMS, key=len, reverse=True)
    )
    # print(HERB_PATTERN)

    """
    解决 "三七一百克"
    pattern = re.compile( 
        rf"({HERB_PATTERN})" # 药名（必须在词库） 
        rf"[^\d零一二两三四五六七八九十百千万亿]*?" # 药名与剂量之间噪声 
        rf"([零一二两三四五六七八九十百千万亿]+|\d+(?:\.\d+)?)" # 剂量 
        rf"\s*({UNIT_PATTERN})" # 单位 
    )
    
    """
    # ===== 4️⃣ 抽取【药名 + 剂量 + 单位】=====
    pattern = re.compile(
        rf"({HERB_PATTERN})"      # 只允许药名词库
        rf"[^\d]*?"               # 药名与剂量之间的噪声
        rf"(\d+(?:\.\d+)?)"       # 数字
        rf"\s*({UNIT_PATTERN})"   # 单位
    )
    pattern = re.compile( 
        rf"({HERB_PATTERN})" # 药名（必须在词库） 
        rf"[^\d零一二两三四五六七八九十百千万亿]*?" # 药名与剂量之间噪声 
        rf"([零一二两三四五六七八九十百千万亿]+|\d+(?:\.\d+)?)" # 剂量 
        rf"\s*({UNIT_PATTERN})" # 单位 
    )

    results = []
    logger.info(f"text: {text}")
    for match in pattern.finditer(text):
        herb, num, unit = match.groups()
        logger.info(f"raw match -> herb={herb}, num={num}, unit={unit}")

        herb = correct_herb_by_pinyin(herb)

        # —— 只在【剂量位】做中文数字转换 ——
        if re.fullmatch(r"[零一二两三四五六七八九十百千万亿]+", num):
            num = str(chinese_to_arabic(num))

        unit = UNIT_NORMALIZE_MAP.get(unit, unit)

        results.append(f"{herb}{num}{unit}")

    logger.info(f"final results: {results}")

    # ===== 5️⃣ 兜底：如果完全没抽到结构 =====
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
                corrected_tokens.append(correct_herb_by_pinyin(tok))
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
    # text = postprocess_asr("糖梨根30g，就怎么那个逻辑啊，都在一个方法里面啊，黄芩片30g。这不是缺损，这是找不到了，往上往上out对，就这个out out还烦点了，点我我不是看那个啊。")