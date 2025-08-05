import wave
import base64

def pcm_to_wav(pcm_data, wav_file, channels=1, sample_rate=16000, bits_per_sample=16):
    with wave.open(wav_file, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)



def convert_image_file_to_base64(image_path):
    """将图片文件转换为 Base64 编码字符串"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string