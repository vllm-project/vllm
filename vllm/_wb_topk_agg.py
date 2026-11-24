# -*- coding: utf-8 -*-
"""引擎侧 TOPK 热点专家聚合(图模式安全,响应零负载)。

在 vllm/v1/engine/output_processor.py 拼接完请求的 routed_experts
(T, L, K) 后调用 aggregate():按层 np.bincount 累加进常驻计数器。
该数据来自引擎设备侧 buffer(ALLTOALL/MC2 已适配图捕获/回放),因此
FULL_DECODE_ONLY 图模式下 prefill+decode 全量有效;聚合后默认把
CompletionOutput.routed_experts 置 None,不再向客户端下发大数组。

输出(TOPK_AGG_PATH 目录,数据量恒定):
  top{K}_experts.json —— 与 stat_moe.py / expert_pruned.py 同构
  topk_counts.npz / topk_agg_meta.json
ARM 文件出现后以当时计数为基线(扣除 warmup)。

环境变量: TOPK_AGG_PATH(空=关闭)、TOPK_AGG_FLUSH_SEC(默认15)、
          TOPK_AGG_TOPK(默认64)、TOPK_AGG_NUM_EXPERTS(默认896)、
          TOPK_AGG_KEEP_RESPONSE(默认0=不下发响应)。
"""
import atexit
import json
import os
import threading
import time

import numpy as np

_S = {
    'tried': False, 'enabled': False,
    'dir': None, 'flush_sec': 15.0, 'topk': 64, 'E': 896, 'keep_resp': False,
    'counts': None,          # np.ndarray [L, E]
    'n_req': 0, 'n_tokens': 0,
    'base': None, 'armed': False,
    'lock': threading.Lock(),
    'warned': set(),
}


def _log(msg):
    k = msg[:60]
    if k not in _S['warned']:
        _S['warned'].add(k)
        try:
            print(f'[wb_topk_agg] {msg}', flush=True)
        except Exception:
            pass


def _init():
    if _S['tried']:
        return _S['enabled']
    _S['tried'] = True
    d = os.environ.get('TOPK_AGG_PATH', '')
    if not d:
        return False
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:
        _log(f'cannot create {d}: {e}')
        return False
    _S.update(
        enabled=True, dir=d,
        flush_sec=float(os.environ.get('TOPK_AGG_FLUSH_SEC', '15')),
        topk=int(os.environ.get('TOPK_AGG_TOPK', '64')),
        E=int(os.environ.get('TOPK_AGG_NUM_EXPERTS', '896')),
        keep_resp=os.environ.get('TOPK_AGG_KEEP_RESPONSE', '0') == '1',
    )
    t = threading.Thread(target=_loop, daemon=True, name='wb-topk-agg')
    t.start()
    atexit.register(dump_final)
    _log(f'enabled dir={d} E={_S["E"]} keep_resp={_S["keep_resp"]}')
    return True


def aggregate(arr) -> None:
    """arr: np.ndarray (T, L, K) 每请求调用一次。"""
    try:
        if not _init():
            return
        if arr is None or arr.ndim != 3:
            return
        T, L, K = arr.shape
        with _S['lock']:
            if _S['counts'] is None or _S['counts'].shape[0] < L:
                E = _S['E']
                new = np.zeros((max(L, _S['counts'].shape[0] if _S['counts'] is not None else 0), E), dtype=np.int64)
                if _S['counts'] is not None:
                    new[: _S['counts'].shape[0]] = _S['counts']
                _S['counts'] = new
            for l in range(L):
                c = np.bincount(arr[:, l, :].reshape(-1), minlength=_S['E'])
                _S['counts'][l] += c[: _S['E']]
            _S['n_req'] += 1
            _S['n_tokens'] += T
    except Exception as e:
        _log(f'aggregate failed: {e}')


def drop_response() -> bool:
    return _S['enabled'] and not _S['keep_resp']


def _arm_check():
    if not _S['armed'] and os.path.exists(os.path.join(_S['dir'], 'ARM')):
        with _S['lock']:
            if not _S['armed'] and _S['counts'] is not None:
                _S['base'] = _S['counts'].copy()
                _S['armed'] = True
                _log('armed: baseline snapshot taken')


def _eff_locked():
    if _S['counts'] is None:
        return None
    return (_S['counts'] - _S['base']) if (_S['armed'] and _S['base'] is not None) else _S['counts'].copy()


def _write_locked():
    eff = _eff_locked()
    if eff is None:
        return
    k = _S['topk']
    result = {}
    npz = {}
    totals = {}
    for l in range(eff.shape[0]):
        name = f'model.layers.{l}.block_sparse_moe'
        order = np.argsort(-eff[l], kind='stable')
        top = [int(x) for x in order[:k]]
        cnt = [int(x) for x in eff[l][top]]
        result[name] = {'topk_ids': top, 'topk_counts': cnt}
        npz[name] = eff[l]
        totals[name] = int(eff[l].sum())
    d = _S['dir']
    with open(os.path.join(d, f'top{k}_experts.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(d, 'topk_counts.npz'), **npz)
    with open(os.path.join(d, 'topk_agg_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'armed': _S['armed'], 'requests': _S['n_req'], 'routed_tokens': _S['n_tokens'],
            'totals': totals, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, ensure_ascii=False, indent=2)


def _loop():
    while True:
        time.sleep(_S['flush_sec'])
        try:
            _arm_check()
            with _S['lock']:
                _write_locked()
        except Exception as e:
            _log(f'flush failed: {e}')


def dump_final():
    try:
        if not _S['enabled']:
            return
        with _S['lock']:
            _write_locked()
        _log('final dump done')
    except Exception as e:
        _log(f'final dump failed: {e}')
