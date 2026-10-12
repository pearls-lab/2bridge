"""OpenPI-compatible msgpack ndarray wire format, without importing OpenPI."""

import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"Unsupported array dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str,
                    b"shape": value.shape}
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _decode(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Only real numeric arrays are accepted")
        if b"__ndarray__" in value:
            return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def pack(value):
    return msgpack.packb(value, default=_encode)


def unpack(value):
    return msgpack.unpackb(value, object_hook=_decode)
