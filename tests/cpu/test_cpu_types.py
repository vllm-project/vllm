import numpy as np
import pytest
import torch
from scipy.special import erf
from cputypes import (bf16vec8, bf16vec16, bf16vec32, fp32vec4,
                      fp32vec8, int32vec16, fp32vec16, int8vec16)


@pytest.fixture()
def ctor_f(meta, arr_s):
    return {'in': np.arange(arr_s, arr_s + meta['size'], 1, dtype=meta['type']),
            'out': np.zeros((meta['size'],), dtype=meta['type']),
            'msg': f'{meta["vect"]} ctor '}


@pytest.fixture()
def ctor_bf16_f(meta, arr_s):
    arr_fp32 = np.arange(arr_s, arr_s + meta['size'], 1, dtype=np.float32)
    tensor_bf16 = torch.from_numpy(arr_fp32).to(torch.bfloat16)
    tensor_bf16_uint16 = tensor_bf16.view(torch.uint16)
    # for e in tensor_bf16_uint16:
    #     print(f"{e.item(): #X}")

    return {'in': tensor_bf16_uint16.numpy(),
            'in2': arr_fp32,
            'gt': tensor_bf16.to(torch.torch.float32).numpy().astype(np.float32),
            'out': np.zeros((meta['size'],), dtype=meta['type']),
            'msg': f'{meta["vect"]} ctor '}


class TestBF16Vec8:
    meta = dict(vect=bf16vec8.BF16Vec8, type=np.uint16, size=8)

    @pytest.mark.parametrize('meta, arr_s', [(meta, 1), (meta, 100), (meta, 1000)])
    def test_ctor(self, ctor_f):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save(ctor_f['out'])
        assert np.array_equal(ctor_f['in'], ctor_f['out']), ctor_f['msg']

    @pytest.mark.parametrize('vect2, size2', [(fp32vec8.FP32Vec8, 8)])
    @pytest.mark.parametrize('arr_s', [1, 100, 200, 2200.5, 10000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_vect(self, ctor_bf16_f, arr_s, vect2, size2):
        vec2 = vect2(ctor_bf16_f['in2'][:size2])
        vec = self.meta['vect'](vec2)
        vec.save(ctor_bf16_f['out'])
        # no more than 1 in uint16 format difference from torch due to rounding
        assert np.allclose(ctor_bf16_f['in'], ctor_bf16_f['out'], atol=1e0), f"{ctor_bf16_f['msg']}FP32Vec8 {arr_s}"


class TestBF16Vec16:
    meta = dict(vect=bf16vec16.BF16Vec16, type=np.uint16, size=16)

    @pytest.mark.parametrize('arr_s', [1, 100, 1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor(self, ctor_f):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save(ctor_f['out'])
        assert np.array_equal(ctor_f['in'], ctor_f['out']), ctor_f['msg']

    @pytest.mark.parametrize('vect2, size2', [(fp32vec16.FP32Vec16, 16)])
    @pytest.mark.parametrize('arr_s', [1, 100, 2000.5, 3000.5, 4000.5, 5000.5, 6000.5, 7000.5])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_vect(self, ctor_bf16_f, arr_s, vect2, size2):
        vec2 = vect2(ctor_bf16_f['in2'][:size2])
        vec = self.meta['vect'](vec2)
        vec.save(ctor_bf16_f['out'])
        # no more than 1 in uint16 format difference from torch due to rounding
        assert np.allclose(ctor_bf16_f['in'], ctor_bf16_f['out'], atol=1e0), f"{ctor_bf16_f['msg']}FP32Vec16 {arr_s}"

    @pytest.mark.parametrize('n', [3, 8])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_save_n(self, ctor_f, arr_s, n):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save_n(ctor_f['out'], n)
        gt = np.concatenate((np.arange(arr_s, arr_s + n, 1, dtype=self.meta['type']),
                             np.zeros((self.meta['size'] - n,), dtype=self.meta['type'])))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{n}"


class TestBF16Vec32:
    meta = dict(vect=bf16vec32.BF16Vec32, type=np.uint16, size=32)

    @pytest.mark.parametrize('arr_s', [1, 100, 1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor(self, ctor_f):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save(ctor_f['out'])
        assert np.array_equal(ctor_f['in'], ctor_f['out']), ctor_f['msg']

    @pytest.mark.parametrize('vect2, size2', [(bf16vec8.BF16Vec8, 8)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_vect(self, ctor_f, arr_s, vect2, size2):
        vec2 = vect2(ctor_f['in'][:size2])
        vec = self.meta['vect'](vec2)
        vec.save(ctor_f['out'])
        gt = np.tile(np.arange(arr_s, arr_s + size2, 1, dtype=self.meta['type']),
                     int(self.meta['size'] / size2))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{arr_s}"


class TestFP32Vec4:
    meta = dict(vect=fp32vec4.FP32Vec4, type=np.float32, size=4)

    @pytest.mark.parametrize('arr_s', [1, 100, 1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor(self, ctor_f):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save(ctor_f['out'])
        assert np.array_equal(ctor_f['in'], ctor_f['out']), ctor_f['msg']

    @pytest.mark.parametrize('vect2, size2', [(fp32vec4.FP32Vec4, 4)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_vect(self, ctor_f, arr_s, vect2, size2):
        vec2 = vect2(ctor_f['in'][:size2])
        vec = self.meta['vect'](vec2)
        vec.save(ctor_f['out'])
        gt = np.tile(np.arange(arr_s, arr_s + size2, 1, dtype=self.meta['type']),
                     int(self.meta['size'] / size2))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{arr_s}"

    @pytest.mark.parametrize('scalar, value', [((4.0,), 4.0), ((), 0.0)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_scalar(self, ctor_f, scalar, value):
        vec = self.meta['vect'](*scalar)
        vec.save(ctor_f['out'])
        gt = np.array([value]*self.meta['size'], dtype=self.meta['type'])
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{value}"


class TestFP32Vec8:
    meta = dict(vect=fp32vec8.FP32Vec8, type=np.float32, size=8)

    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -1, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor(self, ctor_f):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save(ctor_f['out'])
        assert np.array_equal(ctor_f['in'], ctor_f['out']), ctor_f['msg']

    @pytest.mark.parametrize('vect2, size2', [(fp32vec8.FP32Vec8, 8)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -1, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_vect(self, ctor_f, arr_s, vect2, size2):
        vec2 = vect2(ctor_f['in'][:size2])
        vec = self.meta['vect'](vec2)
        vec.save(ctor_f['out'])
        gt = np.tile(np.arange(arr_s, arr_s + size2, 1, dtype=self.meta['type']),
                     int(self.meta['size'] / size2))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{arr_s}"

    @pytest.mark.parametrize('arr_s', [1, 100.8, 200.5, -2000.5, 3000.5, -3000.5, -5000.5])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_bf16vec8(self, ctor_bf16_f, arr_s):
        vec2 = bf16vec8.BF16Vec8(ctor_bf16_f['in'])
        vec = self.meta['vect'](vec2)
        vec.save(ctor_bf16_f['out'])
        assert np.array_equal(ctor_bf16_f['out'], ctor_bf16_f['gt']), f"{ctor_bf16_f['msg']}BF16Vec8 {arr_s}"

    @pytest.mark.parametrize('scalar, value', [((4.0,), 4.0), ((), 0.0)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -1, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_scalar(self, ctor_f, scalar, value):
        vec = self.meta['vect'](*scalar)
        vec.save(ctor_f['out'])
        gt = np.array([value]*self.meta['size'], dtype=self.meta['type'])
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{value}"

    @pytest.mark.parametrize('f_name, f_vec, f_gt',
                             [('sum', lambda x: x.reduce_sum(), np.sum),
                              ('exp', lambda x: x.exp(), np.exp),
                              ('tanh', lambda x: x.tanh(), np.tanh),
                              ('er', lambda x: x.er(), erf),
                              ('mul', lambda x: x*x, lambda x: x*x),
                              ('add', lambda x: x+x, lambda x: x+x),
                              ('sub', lambda x: x-x, lambda x: x-x),
                              ('div', lambda x: x/x, lambda x: x/x)])
    @pytest.mark.parametrize('arr_s', [1, 10, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_func(self, ctor_f, f_name, f_vec, f_gt):
        vec = self.meta['vect'](ctor_f['in'])
        vec2 = f_vec(vec)
        if not isinstance(vec2, float):
            vec2.save(ctor_f['out'])
            assert np.array_equal(f_gt(ctor_f['in']), ctor_f['out']), ctor_f['msg'] + f_name
        else:
            assert np.array_equal(f_gt(ctor_f['in']), vec2), ctor_f['msg'] + f_name

    @pytest.mark.skip('Not implemented')
    @pytest.mark.parametrize('vect2, size2', [(fp32vec4.FP32Vec4, 4)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -1, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_vects(self, ctor_f, arr_s, vect2, size2):
        vec2 = vect2(ctor_f['in'][:size2])
        vec = self.meta['vect'](*[vec2 for _ in range(int(self.meta['size']/size2))])
        vec.save(ctor_f['out'])
        gt = np.tile(np.arange(arr_s, arr_s + size2, 1, dtype=self.meta['type']),
                     int(self.meta['size'] / size2))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{arr_s}"


class TestINT32Vec16:
    meta = dict(vect=int32vec16.INT32Vec16, type=np.int32, size=16)

    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -1, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor(self, ctor_f):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save(ctor_f['out'])
        assert np.array_equal(ctor_f['in'], ctor_f['out']), ctor_f['msg']

    @pytest.mark.parametrize('n', [3, 7, 8, 11, 13, 15])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -1, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_save_n(self, ctor_f, arr_s, n):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save_n(ctor_f['out'], n)
        gt = np.concatenate((np.arange(arr_s, arr_s + n, 1, dtype=self.meta['type']),
                             np.zeros((self.meta['size'] - n,), dtype=self.meta['type'])))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{n}"


class TestFP32Vec16:
    meta = dict(vect=fp32vec16.FP32Vec16, type=np.float32, size=16)

    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -10, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor(self, ctor_f):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save(ctor_f['out'])
        assert np.array_equal(ctor_f['in'], ctor_f['out']), ctor_f['msg']

    @pytest.mark.parametrize('vect2, size2', [(fp32vec4.FP32Vec4, 4),
                                              (fp32vec8.FP32Vec8, 8),
                                              (fp32vec16.FP32Vec16, 16)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -10, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_vect(self, ctor_f, arr_s, vect2, size2):
        vec2 = vect2(ctor_f['in'][:size2])
        vec = self.meta['vect'](vec2)
        vec.save(ctor_f['out'])
        gt = np.tile(np.arange(arr_s, arr_s + size2, 1, dtype=self.meta['type']),
                     int(self.meta['size'] / size2))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{arr_s}"

    @pytest.mark.parametrize('scalar, value', [((4.0,), 4.0), ((), 0.0)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -10, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_scalar(self, ctor_f, scalar, value):
        vec = self.meta['vect'](*scalar)
        vec.save(ctor_f['out'])
        gt = np.array([value]*self.meta['size'], dtype=self.meta['type'])
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{value}"

    @pytest.mark.parametrize('f_name, f_vec, f_gt',
                             [('sum', lambda x: x.reduce_sum(), np.sum),
                              ('mul', lambda x: x*x, lambda x: x*x),
                              ('add', lambda x: x+x, lambda x: x+x),
                              ('sub', lambda x: x-x, lambda x: x-x),
                              ('div', lambda x: x/x, lambda x: x/x)])
    @pytest.mark.parametrize('arr_s', [1, 10, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_func(self, ctor_f, f_name, f_vec, f_gt):
        vec = self.meta['vect'](ctor_f['in'])
        vec2 = f_vec(vec)
        if not isinstance(vec2, float):
            vec2.save(ctor_f['out'])
            assert np.array_equal(f_gt(ctor_f['in']), ctor_f['out']), ctor_f['msg'] + f_name
        else:
            assert np.array_equal(f_gt(ctor_f['in']), vec2), ctor_f['msg'] + f_name

    @pytest.mark.skip('Not implemented')
    @pytest.mark.parametrize('vect2, size2', [(fp32vec4.FP32Vec4, 4)])
    @pytest.mark.parametrize('arr_s', [1, 100, 1000, -10, -100, -1000])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor_vects(self, ctor_f, arr_s, vect2, size2):
        vec2 = vect2(ctor_f['in'][:size2])
        vec = self.meta['vect'](*[vec2 for _ in range(int(self.meta['size']/size2))])
        vec.save(ctor_f['out'])
        gt = np.tile(np.arange(arr_s, arr_s + size2, 1, dtype=self.meta['type']),
                     int(self.meta['size'] / size2))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{arr_s}"


class TestINT8Vec16:
    meta = dict(vect=int8vec16.INT8Vec16, type=np.int8, size=16)

    @pytest.mark.parametrize('arr_s', [1, 100])
    @pytest.mark.parametrize('meta', [meta])
    def test_ctor(self, ctor_f):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save(ctor_f['out'])
        assert np.array_equal(ctor_f['in'], ctor_f['out']), ctor_f['msg']

    @pytest.mark.parametrize('n', [3, 7, 8, 11, 13, 15])
    @pytest.mark.parametrize('arr_s', [1, 100])
    @pytest.mark.parametrize('meta', [meta])
    def test_save_n(self, ctor_f, arr_s, n):
        vec = self.meta['vect'](ctor_f['in'])
        vec.save_n(ctor_f['out'], n)
        gt = np.concatenate((np.arange(arr_s, arr_s + n, 1, dtype=self.meta['type']),
                             np.zeros((self.meta['size'] - n,), dtype=self.meta['type'])))
        assert np.array_equal(ctor_f['out'], gt), f"{ctor_f['msg']}{n}"

    @pytest.fixture()
    def ctor_f2(self, arr_s):
        return {'in': np.arange(arr_s, arr_s + self.meta['size'], 1, dtype=np.int32),
                'in2': np.arange(arr_s, arr_s + self.meta['size'], 1, dtype=np.float32),
                'out': np.zeros((self.meta['size'],), dtype=self.meta['type']),
                'msg': f'{self.meta["vect"]} ctor '}

    @pytest.mark.parametrize('arr_s', [1, 111, 121, 131, -11, -128, -138, -158])
    def test_ctor_narrow(self, ctor_f2, arr_s):
        def _saturate(x):
            if x > 127:
                return 127
            if x < -128:
                return -128
            return x

        vec2 = fp32vec16.FP32Vec16(ctor_f2['in2'])
        vec = self.meta['vect'](vec2)
        vec.save(ctor_f2['out'])
        gt = np.array(list(map(_saturate, ctor_f2['in']))).astype(np.int8)
        assert np.array_equal(ctor_f2['out'], gt), ctor_f2['msg'] + f'{arr_s}'
