from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from typing import Any

class AliFunASR():
    def __init__(self):
        self.model_dir = "iic/SenseVoiceSmall"
        self.device = "cpu"

    async def recognize(self, audio_bytes: Any):
        model = AutoModel(
            model=self.model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )

        # en
        print(model.model_path)
        res = model.generate(
            input=audio_bytes,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        print(f"******ASR result: {text}")
        return text

def read_audio_file_as_bytes(file_path):
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

# if __name__=="__main__":
#     asr = AliFunASR()
#     from scipy.io import wavfile
#     data = read_audio_file_as_bytes("downloaded_audio.wav")
#     # samplerate, data = wavfile.read('downloaded_audio.wav')
#     print(data[:100])
#     asr.recognize(data)
