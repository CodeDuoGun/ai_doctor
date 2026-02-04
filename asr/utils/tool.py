from pathlib import Path
import wave


def save_pcm16le_to_wav(
    pcm_bytes: bytes,
    wav_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,  # 16bit
) -> None:
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
