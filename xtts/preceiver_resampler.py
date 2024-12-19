from typing import List
from numpy import ndarray
import numpy
from onnxruntime import InferenceSession
import os

from utils import *
from model_setting import ModelSetting

class PreceiverResampler:
    def __init__(self, model_setting: ModelSetting):
        self.onnx_session: InferenceSession = None
        self.ref_mel: ndarray = None
        self.model_setting = model_setting
        self.magic_number_1 = 0.4
        self.post_init()
    
    def post_init(self):
        self.onnx_session = InferenceSession(os.path.join(self.model_setting.model_dir, 'preceiver_resampler.onnx'), providers=['CUDAExecutionProvider'])
        self.ref_mel = np.load(os.path.join(self.model_setting.model_dir, 'ref_mel.npy'))
        
    def get_reference_audio(self, text_token_count: int) -> torch.Tensor:
        ref_length = max(round(text_token_count * self.model_setting.scale_rate * self.magic_number_1), 2)
        conds: ndarray = self.ref_mel.copy()
        cut_len = conds.shape[1] % ref_length if conds.shape[1] % ref_length != 0 else 0
        if cut_len != 0:
            conds = conds[:, :-cut_len]
        if conds.shape[1] // ref_length > 0:
            conds = numpy.split(conds, conds.shape[1] // ref_length, axis=1)
            conds = numpy.concatenate(conds, axis=0)
        onnxruntime_input = {k.name: v for k, v in zip(self.onnx_session.get_inputs(), (conds,))}
        onnxruntime_outputs = self.onnx_session.run(None, onnxruntime_input)
        conds_output = torch.Tensor(onnxruntime_outputs[0]).mean(0, keepdim=True).to('cuda')
        return conds_output[0]

