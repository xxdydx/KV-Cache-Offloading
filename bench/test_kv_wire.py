"""CPU tests for byte-exact raw KV framing and interrupted connections."""

import socket
import struct
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from bench.kv_tcp_handoff import receive_object,read_snapshot_payload,send_snapshot
from bench import kv_tcp_handoff


@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
def test_raw_snapshot_preserves_every_bit_and_metadata(dtype):
    # Include signed zero and NaN patterns: numerical equality is insufficient.
    bits=torch.arange(2*2*257*2*4,dtype=torch.int16).reshape(2,2,257,2,4)
    bits.flatten()[0]=-32768
    bits.flatten()[1]=32767
    kv=bits.view(dtype)
    snapshot=SimpleNamespace(token_ids=[5]*258,num_prompt_tokens=257,num_cached_tokens=257,
                             block_size=256,temperature=0,max_tokens=64,ignore_eos=True,
                             kv=kv,payload_bytes=kv.numel()*kv.element_size())
    left,right=socket.socketpair()
    with left,right,ThreadPoolExecutor(max_workers=1) as executor:
        left.settimeout(5);right.settimeout(5)
        sent=executor.submit(send_snapshot,left,snapshot,"raw")
        message,timing=receive_object(right)
        assert "snapshot" not in message  # The tensor was not pickled in the header.
        restored,timing=read_snapshot_payload(right,message,timing)
        stats=sent.result(timeout=5)
    assert torch.equal(restored["kv"].view(torch.uint8),kv.view(torch.uint8))
    assert restored["token_ids"]==snapshot.token_ids
    assert restored["num_cached_tokens"]==257
    assert stats["wire_payload_bytes"]>=snapshot.payload_bytes
    assert stats["wire_payload_bytes"]<snapshot.payload_bytes+4096


def test_oversized_control_frame_is_rejected_before_allocation():
    left,right=socket.socketpair()
    with left,right:
        left.sendall(struct.pack("!Q",8*1024**3+1))
        with pytest.raises(ValueError,match="8 GiB"):
            receive_object(right)


def test_truncated_control_frame_raises_eof():
    left,right=socket.socketpair()
    with right:
        left.sendall(struct.pack("!Q",10)+b"short")
        left.close()
        with pytest.raises(EOFError):
            receive_object(right)


def test_full_cache_pickle_is_rejected_before_workers_start_in_8gib_container(tmp_path,monkeypatch):
    model=tmp_path/"model"
    model.mkdir()
    (model/"config.json").write_text(json.dumps(dict(num_hidden_layers=36,num_key_value_heads=8,head_dim=128)))
    args=SimpleNamespace(model=str(model),out=str(tmp_path/"results"),lengths=[8192],
                         handoff_after=[1,17],wire_format="pickle")
    monkeypatch.setattr(kv_tcp_handoff,"memory_info",lambda:{"memory.max":8*1024**3})
    monkeypatch.setattr(kv_tcp_handoff.importlib.metadata,"version",lambda _:"test")
    monkeypatch.setattr(kv_tcp_handoff.subprocess,"check_output",lambda *a,**kw:"two test GPUs")
    def forbidden_spawn(*a,**kw):
        raise AssertionError("Workers must not start when preflight rejects the memory budget")
    monkeypatch.setattr(kv_tcp_handoff.subprocess,"Popen",forbidden_spawn)
    with pytest.raises(RuntimeError,match="CPU memory"):
        kv_tcp_handoff.parent(args)
