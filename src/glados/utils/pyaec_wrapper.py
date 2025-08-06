from pyaec import Aec
import numpy as np

class PyaecEchoCanceller:
    def __init__(self, frame_size=160, filter_length=1600, sample_rate=16000):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.echo_ref_buffer = np.zeros(frame_size, dtype=np.int16)
        self.aec = Aec(frame_size, filter_length, sample_rate, True)

    def update_echo_reference(self, float_audio: np.ndarray) -> None:
        # Flatten and clip to int16 range
        float_audio = float_audio.flatten()
        int_audio = np.clip(float_audio * 32768, -32768, 32767).astype(np.int16)

        # Truncate or pad to match frame_size
        if len(int_audio) < self.frame_size:
            padded = np.zeros(self.frame_size, dtype=np.int16)
            padded[:len(int_audio)] = int_audio
            self.echo_ref_buffer[:] = padded
        else:
            self.echo_ref_buffer[:] = int_audio[:self.frame_size]

    def cancel(self, mic_audio: np.ndarray) -> np.ndarray:
        # Flatten and clip mic audio
        mic_audio = mic_audio.flatten()
        mic_int16 = np.clip(mic_audio * 32768, -32768, 32767).astype(np.int16)

        # Truncate if needed
        min_len = min(len(mic_int16), len(self.echo_ref_buffer))
        mic_int16 = mic_int16[:min_len]
        echo_ref = self.echo_ref_buffer[:min_len]

        clean = np.array(self.aec.cancel_echo(mic_int16, echo_ref), dtype=np.int16)
        return clean.astype(np.float32) / 32768.0