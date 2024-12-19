from typing import List
from utils import *

class TtsMetrics:
    def __init__(self, chunk_size: int = 20, first_chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.first_chunk_size = first_chunk_size
        self.time_start: float = 0
        self.time_end: float = 0
        self.token_times: List[float] = []
        self.audio_chunk_times: List[float] = []
        
    def calc_non_streaming(self):
        total_time = self.time_end - self.time_start
        audio_time = len(self.token_times) * 50 / 1000
        rtf = total_time / audio_time
        latent_time = self.token_times[-1] - self.time_start
        first_byte_time = self.audio_chunk_times[0] - self.time_start
        print(f'latent time: {latent_time}, first byte time: {first_byte_time}, total time: {total_time}, audio time: {audio_time}, rtf: {rtf}')

    def calc_streaming(self):
        total_time = self.time_end - self.time_start
        audio_time = len(self.token_times) * 50 / 1000
        rtf = total_time / audio_time
        first_chunk_time = self.token_times[self.first_chunk_size - 1] - self.time_start
        first_byte_time = self.audio_chunk_times[0] - self.time_start
        print(f'first chunk time: {first_chunk_time}, first byte time: {first_byte_time}, total time: {total_time}, audio time: {audio_time}, rtf: {rtf}')