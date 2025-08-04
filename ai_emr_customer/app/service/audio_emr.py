import os
import whisper
import subprocess
from utils.log import logger

class AudioEmrService:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
    
    def genasr(self,  model=None):
        if not model:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            model_dir = "iic/SenseVoiceSmall"

            model = AutoModel(
                model=model_dir,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="auto",
            )

            # en
            res = model.generate(
                input=self.audio_path,
                cache={},
                language="zh",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,  #
                merge_length_s=15,
            )
            text = rich_transcription_postprocess(res[0]["text"])

            print(f"asr res: {text}")
            return text
        else:
            model = whisper.load_model("turbo")
            result = model.transcribe(audio_path)
            print(result["text"])
            return result["text"]



    def gen_emr(self, asr_result:str):
        """
        基于音视频的解读结果，自动生成病历
        """
        # TODO: 使用自己微调后ai病历模型
        emr_prompt = ""
        emr_data = {
            "patient_name": "张三",
            "diagnosis": "感冒",
            "treatment": "多喝水，休息"
        }
        return emr_data

    def process_audio(self,file_path: str):
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

if __name__ == "__main__":
    audio_path = "app/data/audio/xlys_hx1754126556959.mp3"
    audio_servicec= AudioEmrService(audio_path=audio_path)
