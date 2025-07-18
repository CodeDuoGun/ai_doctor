# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ctypes
from typing import Union


class AudioFrame:
    """
    A class that represents a frame of audio data with specific properties such as sample rate,
    number of channels, and samples per channel.
    """

    def __init__(
        self,
        data: Union[bytes, bytearray, memoryview],
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
    ) -> None:
        """
        Initialize an AudioFrame instance.

        Args:
            data (Union[bytes, bytearray, memoryview]): The raw audio data, which must be at least
                `num_channels * samples_per_channel * sizeof(int16)` bytes long.
            sample_rate (int): The sample rate of the audio in Hz.
            num_channels (int): The number of audio channels (e.g., 1 for mono, 2 for stereo).
            samples_per_channel (int): The number of samples per channel.

        Raises:
            ValueError: If the length of `data` is smaller than the required size.
        """
        if len(data) < num_channels * samples_per_channel * ctypes.sizeof(
            ctypes.c_int16
        ):
            raise ValueError(
                "data length must be >= num_channels * samples_per_channel * sizeof(int16)"
            )

        self._data = bytearray(data)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._samples_per_channel = samples_per_channel

    @staticmethod
    def create(
        sample_rate: int, num_channels: int, samples_per_channel: int
    ) -> "AudioFrame":
        """
        Create a new empty AudioFrame instance with specified sample rate, number of channels,
        and samples per channel.

        Args:
            sample_rate (int): The sample rate of the audio in Hz.
            num_channels (int): The number of audio channels (e.g., 1 for mono, 2 for stereo).
            samples_per_channel (int): The number of samples per channel.

        Returns:
            AudioFrame: A new AudioFrame instance with uninitialized (zeroed) data.
        """
        size = num_channels * samples_per_channel * ctypes.sizeof(ctypes.c_int16)
        data = bytearray(size)
        return AudioFrame(data, sample_rate, num_channels, samples_per_channel)

    

    @property
    def data(self) -> memoryview:
        """
        Returns a memory view of the audio data as 16-bit signed integers.

        Returns:
            memoryview: A memory view of the audio data.
        """
        return memoryview(self._data).cast("h")

    @property
    def sample_rate(self) -> int:
        """
        Returns the sample rate of the audio frame.

        Returns:
            int: The sample rate in Hz.
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels in the audio frame.

        Returns:
            int: The number of audio channels (e.g., 1 for mono, 2 for stereo).
        """
        return self._num_channels

    @property
    def samples_per_channel(self) -> int:
        """
        Returns the number of samples per channel.

        Returns:
            int: The number of samples per channel.
        """
        return self._samples_per_channel

    @property
    def duration(self) -> float:
        """
        Returns the duration of the audio frame in seconds.

        Returns:
            float: The duration in seconds.
        """
        return self.samples_per_channel / self.sample_rate

    def to_wav_bytes(self) -> bytes:
        """
        Convert the audio frame data to a WAV-formatted byte stream.

        Returns:
            bytes: The audio data encoded in WAV format.
        """
        import wave
        import io

        with io.BytesIO() as wav_file:
            with wave.open(wav_file, "wb") as wav:
                wav.setnchannels(self.num_channels)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes(self._data)

            return wav_file.getvalue()

    def __repr__(self) -> str:
        return (
            f"rtc.AudioFrame(sample_rate={self.sample_rate}, "
            f"num_channels={self.num_channels}, "
            f"samples_per_channel={self.samples_per_channel}, "
            f"duration={self.duration:.3f})"
        )
