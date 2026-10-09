"""The controller <-> trainer worker protocol: length-prefixed pickle frames on a unix socket, in order.

    controller -> worker   ("hello", cfg dict) ("batch", TrainBatch) ("grant", v) ("snapshot", tag, dir) ("stop",)
    worker -> controller   ("ready", v0) ("weights_ready", WeightsReady) ("pushed", Pushed)
                           ("snapshot_done", tag, path, seconds) ("bye", summary) ("error", text)

Both ends are ours and run on one host, so pickle is fine. The worker sends from two threads (the training loop and
the sender), so its sends hold a lock; the controller's reader dispatches by message kind.
"""
import asyncio
import pickle
import socket
import struct
import threading

_HEADER = struct.Struct("!Q")


def encode(msg) -> bytes:
    body = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    return _HEADER.pack(len(body)) + body


class Channel:
    """The worker's (blocking) end."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._lock = threading.Lock()

    def send(self, *msg) -> None:
        data = encode(msg)
        with self._lock:
            self.sock.sendall(data)

    def recv(self):
        (n,) = _HEADER.unpack(self._exactly(_HEADER.size))
        return pickle.loads(self._exactly(n))

    def _exactly(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise EOFError("peer closed")
            buf += chunk
        return bytes(buf)


async def asend(writer: asyncio.StreamWriter, *msg) -> None:
    writer.write(encode(msg))
    await writer.drain()


async def arecv(reader: asyncio.StreamReader):
    (n,) = _HEADER.unpack(await reader.readexactly(_HEADER.size))
    return pickle.loads(await reader.readexactly(n))
