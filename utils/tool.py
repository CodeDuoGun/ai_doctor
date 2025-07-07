import wave

def pcm_to_wav(pcm_data, wav_file, channels=1, sample_rate=16000, bits_per_sample=16):
    with wave.open(wav_file, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)