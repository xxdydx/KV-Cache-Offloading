"""Versioned JSON control frames and raw BF16 payloads for the M1 probe.

No retries: each connection carries OFFER/PERMIT/TRANSFER/READY exchanges,
followed by STOP. Importing this module does not import torch or the engine.
"""

import json
import math
import struct
from dataclasses import dataclass
from types import SimpleNamespace


VERSION = 1
MAX_HEADER_BYTES = 256 * 1024
KINDS = {"OFFER", "PERMIT", "TRANSFER", "READY", "STOP"}


@dataclass(frozen=True)
class Limits:
    layers: int
    heads: int
    head_dim: int
    block_size: int
    vocab_size: int
    max_prompt_tokens: int
    output_tokens: int
    max_payload_bytes: int

    def payload_bytes(self, cached_tokens):
        return 2 * self.layers * cached_tokens * self.heads * self.head_dim * 2


def positive_int(value):
    return type(value) is int and value > 0


def recv_into_exact(connection, view):
    offset = 0
    while offset < len(view):
        count = connection.recv_into(view[offset:])
        if not count:
            raise EOFError("Peer closed during handoff")
        offset += count


def send_message(connection, kind, **fields):
    message = dict(fields, version=VERSION, kind=kind)
    validate_message(message)
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_HEADER_BYTES:
        raise ValueError("Control frame too large")
    connection.sendall(struct.pack("!I", len(payload)))
    connection.sendall(payload)


def validate_message(message):
    if (not isinstance(message, dict) or type(message.get("version")) is not int
            or message["version"] != VERSION or not isinstance(message.get("kind"), str)
            or message["kind"] not in KINDS):
        raise ValueError("Invalid protocol version or message kind")
    if message["kind"] != "STOP":
        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
            raise ValueError("Invalid request ID")


def receive_message(connection, kind=None, request_id=None):
    prefix = bytearray(4)
    recv_into_exact(connection, memoryview(prefix))
    size = struct.unpack("!I", prefix)[0]
    if not 0 < size <= MAX_HEADER_BYTES:
        raise ValueError("Invalid control frame length")
    payload = bytearray(size)
    recv_into_exact(connection, memoryview(payload))
    message = json.loads(payload)
    validate_message(message)
    if kind is not None and message["kind"] != kind:
        raise ValueError(f"Expected {kind}, got {message['kind']}")
    if request_id is not None and message.get("request_id") != request_id:
        raise ValueError("Mismatched request ID")
    return message


def validate_offer(message, limits):
    prompt = message.get("num_prompt_tokens")
    outputs = message.get("max_tokens")
    if (message.get("kind") != "OFFER" or not positive_int(prompt)
            or prompt > limits.max_prompt_tokens or type(outputs) is not int
            or outputs != limits.output_tokens or outputs < 2):
        raise ValueError("Invalid request budget")
    if limits.payload_bytes(prompt) > limits.max_payload_bytes:
        raise MemoryError("Snapshot exceeds host payload budget")
    return prompt, outputs


def snapshot_header(request_id, snapshot):
    return dict(request_id=request_id, token_ids=snapshot.token_ids,
                num_prompt_tokens=snapshot.num_prompt_tokens,
                num_cached_tokens=snapshot.num_cached_tokens,
                block_size=snapshot.block_size, temperature=snapshot.temperature,
                max_tokens=snapshot.max_tokens, ignore_eos=snapshot.ignore_eos,
                kv_shape=list(snapshot.kv.shape), kv_dtype=str(snapshot.kv.dtype),
                payload_bytes=snapshot.kv.numel() * snapshot.kv.element_size())


def validate_snapshot(header, offer, limits):
    """Validate everything before allocating a receiver tensor."""
    prompt, outputs = validate_offer(offer, limits)
    tokens = header.get("token_ids")
    expected = [2, limits.layers, prompt, limits.heads, limits.head_dim]
    if (header.get("kind") != "TRANSFER" or header.get("request_id") != offer["request_id"]
            or type(header.get("num_prompt_tokens")) is not int
            or header["num_prompt_tokens"] != prompt
            or type(header.get("num_cached_tokens")) is not int
            or header["num_cached_tokens"] != prompt
            or type(header.get("max_tokens")) is not int or header["max_tokens"] != outputs
            or type(header.get("block_size")) is not int or header["block_size"] != limits.block_size
            or header.get("kv_dtype") != "torch.bfloat16"
            or not isinstance(header.get("kv_shape"), list)
            or any(type(n) is not int for n in header["kv_shape"])
            or header["kv_shape"] != expected
            or type(header.get("payload_bytes")) is not int
            or header["payload_bytes"] != limits.payload_bytes(prompt)
            or not isinstance(tokens, list) or len(tokens) != prompt + 1
            or any(type(t) is not int or not 0 <= t < limits.vocab_size for t in tokens)
            or type(header.get("temperature")) not in (int, float)
            or not math.isfinite(header["temperature"]) or header["temperature"] != 0
            or header.get("ignore_eos") is not True):
        raise ValueError("Snapshot does not match reserved request/layout")


def send_snapshot(connection, request_id, snapshot):
    import torch

    if snapshot.kv.device.type != "cpu" or not snapshot.kv.is_contiguous():
        raise ValueError("Snapshot must use contiguous CPU storage")
    send_message(connection, "TRANSFER", **snapshot_header(request_id, snapshot))
    # The byte view shares snapshot storage; sendall completes before returning.
    connection.sendall(memoryview(snapshot.kv.view(torch.uint8).numpy()).cast("B"))


def receive_snapshot(connection, offer, limits):
    header = receive_message(connection, "TRANSFER", offer["request_id"])
    validate_snapshot(header, offer, limits)
    import torch

    payload = torch.empty(header["kv_shape"], dtype=torch.bfloat16, device="cpu")
    recv_into_exact(connection, memoryview(payload.view(torch.uint8).numpy()).cast("B"))
    names = ("token_ids", "num_prompt_tokens", "num_cached_tokens", "block_size",
             "temperature", "max_tokens", "ignore_eos")
    return SimpleNamespace(**{name: header[name] for name in names}, kv=payload)
