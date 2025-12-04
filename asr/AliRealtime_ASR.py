"""
Realtime ASR helper built on Aliyun NLS SDK.

This rewrites the former test script into a reusable class that can
consume a stream of PCM audio bytes and yield/callback intermediate
results.
"""
import queue
import time
from typing import Callable, Generator, Iterable, Optional, Tuple

import json

import nls

DEFAULT_URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"


ResultEvent = Tuple[str, dict]


class AliRealtimeASR:
    """
    Wraps NlsSpeechTranscriber for streaming speech-to-text.

    Example:
        asr = AliRealtimeASR(token, appkey)
        def on_result(event, payload):
            print(event, payload)
        asr.run_stream(pcm_iterable, on_result)
    """

    def __init__(
        self,
        token: str,
        appkey: str,
        url: str = DEFAULT_URL,
    ) -> None:
        self.token = token
        self.appkey = appkey
        self.url = url

    def _put_event(self, q: queue.Queue, event: str, message: dict) -> None:
        q.put((event, message))

    def _build_transcriber(self, result_queue: queue.Queue) -> "nls.NlsSpeechTranscriber":
        """
        Instantiate transcriber with callbacks that push events into a queue.
        """
        return nls.NlsSpeechTranscriber(
            url=self.url,
            token=self.token,
            appkey=self.appkey,
            on_sentence_begin=lambda message, *args: self._put_event(
                result_queue, "sentence_begin", message
            ),
            on_sentence_end=lambda message, *args: self._put_event(
                result_queue, "sentence_end", json.loads(message).get("payload", {}).get("result", "")
            ),
            on_start=lambda message, *args: self._put_event(
                result_queue, "start", message
            ),
            on_result_changed=lambda message, *args: self._put_event(
                result_queue, "intermediate", json.loads(message).get("payload", {}).get("result", "")
            ),
            on_completed=lambda message, *args: self._put_event(
                result_queue, "completed", message
            ),
            on_error=lambda message, *args: self._put_event(
                result_queue, "error", {"message": message, "args": args}
            ),
            on_close=lambda *args: self._put_event(
                result_queue, "close", {"args": args}
            ),
        )

    def _drain_queue(
        self, result_queue: queue.Queue, callback: Optional[Callable[[str, dict], None]]
    ) -> Generator[ResultEvent, None, None]:
        """
        Yield all pending events and optionally invoke callback.
        """
        while True:
            try:
                event, payload = result_queue.get_nowait()
            except queue.Empty:
                break
            if callback:
                callback(event, payload)
            yield event, payload

    def run_stream(
        self,
        audio_iterable: Iterable[bytes],
        on_result: Optional[Callable[[str, dict], None]] = None,
        chunk_interval: float = 0.01,
        enable_intermediate_result: bool = True,
        enable_punctuation_prediction: bool = True,
        enable_inverse_text_normalization: bool = True,
        aformat: str = "pcm",
    ) -> Generator[ResultEvent, None, None]:
        """
        Stream audio to Aliyun and yield events in near real-time.

        Args:
            audio_iterable: Iterable of PCM byte chunks.
            on_result: Optional callback invoked for each event.
            chunk_interval: Sleep between sends to avoid flooding.
        """
        result_queue: queue.Queue = queue.Queue()
        sr = self._build_transcriber(result_queue)
        print("ASR session starting")

        sr.start(
            aformat=aformat,
            enable_intermediate_result=enable_intermediate_result,
            enable_punctuation_prediction=enable_punctuation_prediction,
            enable_inverse_text_normalization=enable_inverse_text_normalization,
        )

        for chunk in audio_iterable:
            sr.send_audio(chunk)
            time.sleep(chunk_interval)
            yield from self._drain_queue(result_queue, on_result)

        sr.ctrl(ex={"stream_end": True})
        time.sleep(0.2)
        sr.stop()
        print("ASR session stopped")
        yield from self._drain_queue(result_queue, on_result)
