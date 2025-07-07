from app.utils.log import logger
from app.config import config
import subprocess
import os
import whisper

model = whisper.load_model("turbo")


def main():
    # 获取音频，特征
    audio = extract_audio(file_path="app/data/report/uuid001/297_1749534339.mp4")
    # 处理音频，转文本
    custom_asr(audio, use_funasr=True)

def process_audio():
    pass

def extract_audio(file_path: str):
    file_type = os.path.basename(file_path).split('.')[-1]
    if file_type in ("mp4", "mkv", "avi"):
        # 处理视频文件，提取音频
        local_audio_path = os.path.join("temp", f"{os.path.basename(file_path).split('.')[0]}.wav")
        command = [
            'ffmpeg',
            '-i', file_path,
            '-c:v', 'libxvid',
            '-c:a', 'libmp3lame',
            local_audio_path
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print(f"转换成功，输出文件: {local_audio_path}")
        else:
            print(f"转换失败，错误信息: {stderr}")
        command = f"ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 output.wav"

        logger.info(f"Extracting audio from {file_path}")
        # 这里可以使用ffmpeg或其他库来提取音频
        return local_audio_path
    elif file_type in ("wav", "mp3"):
        # 直接返回音频文件路径
        logger.info(f"Using audio file {file_path}")
        return file_path
    else:
        logger.error(f"Unsupported file type: {file_type}")
        raise ValueError("Unsupported file type")



def gen_advice(text: str, model=None):
    pass

def custom_asr(audio_path:str, use_funasr:bool=False):
    if use_funasr:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        model_dir = "iic/SenseVoiceSmall"

        model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="auto",
        )
        print(1111)

        # en
        res = model.generate(
            input=audio_path,
            cache={},
            language="zh",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])

        print(f"asr res: {text}")
    else:
        result = model.transcribe(audio_path)
        print(result["text"])

main()