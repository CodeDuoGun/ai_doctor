"""
Realtime ASR helper built on DashScope fun-asr-realtime.

This wraps the DashScope streaming ASR so it can be used in the same
style as AliRealtimeASR: feed it an iterable of PCM audio chunks and
receive intermediate/final results via callback and generator.

Environment:
    DASHSCOPE_API_KEY  - required, see DashScope console
"""
from __future__ import annotations

import os
import queue
import time
from typing import Callable, Generator, Iterable, Optional, Tuple

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult, vocabulary


ResultEvent = Tuple[str, str]


class _QueueCallback(RecognitionCallback):
    """
    DashScope recognition callback that pushes events into a queue.
    """

    def __init__(self, q: "queue.Queue[ResultEvent]") -> None:
        super().__init__()
        self._q = q

    def on_open(self) -> None:  # type: ignore[override]
        # Connection established
        pass

    def on_close(self) -> None:  # type: ignore[override]
        # Connection closed
        self._q.put(("close", ""))

    def on_complete(self) -> None:  # type: ignore[override]
        # Task completed
        self._q.put(("completed", ""))

    def on_error(self, message) -> None:  # type: ignore[override]
        # message usually has request_id and message fields
        text = getattr(message, "message", str(message))
        self._q.put(("error", text))

    def on_event(self, result: RecognitionResult) -> None:  # type: ignore[override]
        sentence = result.get_sentence()
        if "text" not in sentence:
            return
        text = sentence["text"]
        if RecognitionResult.is_sentence_end(sentence):
            self._q.put(("sentence_end", text))
        else:
            self._q.put(("intermediate", text))


class RealtimeFunASR:
    """
    Wraps DashScope fun-asr-realtime for streaming speech-to-text.

    Example:
        asr = RealtimeFunASR()

        def on_result(event, text):
            print(event, text)

        asr.run_stream(pcm_iterable, on_result)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "fun-asr-realtime",
        base_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY must be set in environment or passed explicitly")

        # Configure DashScope globals
        dashscope.api_key = self.api_key
        dashscope.base_websocket_api_url = base_url

        self.model = model

    def _drain_queue(
        self,
        result_queue: "queue.Queue[ResultEvent]",
        on_result: Optional[Callable[[str, str], None]],
    ) -> Generator[ResultEvent, None, None]:
        """
        Yield all pending events and optionally invoke callback.
        """
        while True:
            try:
                event, text = result_queue.get_nowait()
            except queue.Empty:
                break
            if on_result:
                on_result(event, text)
            yield event, text

    def run_stream(
        self,
        audio_iterable: Iterable[bytes],
        on_result: Optional[Callable[[str, str], None]] = None,
        sample_rate: int = 16000,
        audio_format: str = "pcm",
        semantic_punctuation_enabled: bool = False,
        chunk_interval: float = 0.0,
    ) -> Generator[ResultEvent, None, None]:
        """
        Stream audio to DashScope fun-asr-realtime and yield events.

        Args:
            audio_iterable: Iterable of PCM byte chunks (16kHz, mono).
            on_result: Optional callback invoked for each event.
            sample_rate: Audio sample rate, typically 16000.
            audio_format: One of 'pcm', 'wav', 'opus', etc. (we use 'pcm').
            semantic_punctuation_enabled: Whether to enable semantic punctuation.
            chunk_interval: Optional sleep between sends.
        """
        result_queue: "queue.Queue[ResultEvent]" = queue.Queue()
        callback = _QueueCallback(result_queue)

        recognition = Recognition(
            model=self.model,
            # model = "c128274d818844f6a8c58c5c897834ef",
            vocabulary_id="vocab-shylasr0-04d02cc8f3714a5d805f0a45955776c0",
            format=audio_format,
            sample_rate=sample_rate,
            semantic_punctuation_enabled=semantic_punctuation_enabled,
            heartbeat=True,
            language_hints=["zh"],
            callback=callback,
        )

        recognition.start()

        try:
            for chunk in audio_iterable:
                recognition.send_audio_frame(chunk)
                if chunk_interval > 0:
                    time.sleep(chunk_interval)
                # Drain any events that arrived
                yield from self._drain_queue(result_queue, on_result)

            # Stop recognition and drain remaining events
            recognition.stop()
            time.sleep(0.2)
            yield from self._drain_queue(result_queue, on_result)
        finally:
            # Best-effort stop; Recognition handles internal cleanup
            try:
                recognition.stop()
            except Exception:
                pass

