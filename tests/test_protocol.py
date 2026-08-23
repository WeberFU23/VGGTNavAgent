"""mapping TCP framing and client-side payload validation."""

import socket
import struct

import numpy as np
import pytest

from mapping.client import MappingClient
from mapping.protocol import MAX_HEADER_BYTES, recv_msg, send_msg


def test_protocol_roundtrip_with_binary_payload():
    left, right = socket.socketpair()
    try:
        send_msg(left, {"cmd": "ping"}, b"rgb")
        header, payload = recv_msg(right)
        assert header["cmd"] == "ping"
        assert header["payload_len"] == 3
        assert payload == b"rgb"
    finally:
        left.close()
        right.close()


def test_protocol_rejects_oversized_header_before_reading_body():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack(">Q", MAX_HEADER_BYTES + 1))
        with pytest.raises(ValueError, match="header length"):
            recv_msg(right)
    finally:
        left.close()
        right.close()


def test_client_rejects_non_rgb_frame_before_network_io():
    client = MappingClient(port=1)
    with pytest.raises(ValueError, match="HxWx3"):
        client.feed_frame(np.zeros((8, 8), dtype=np.uint8))


def test_frame_points_keeps_atomic_snapshot_revision():
    client = MappingClient(port=1)
    payload = np.array([[1.0, 2.0, 3.0]], dtype=np.float32).tobytes()
    response = {
        "frames": [{"frame_id": 7, "h": 1, "w": 1, "stride": 6,
                    "pose": np.eye(4).tolist()}],
        "snapshot_revision": {"num_frames": 8, "num_submaps": 2,
                              "num_loop_closures": 1},
    }
    client._request = lambda _header: (response, payload)
    frames = client.get_frame_points()
    assert frames[0]["frame_id"] == 7
    assert client.last_frame_snapshot_revision == response["snapshot_revision"]


def test_client_batches_candidate_resolution():
    client = MappingClient(port=1)
    seen = {}

    def request(header):
        seen.update(header)
        return ({"candidates": {"c1": {"found": True,
                                        "point": [1, 2, 3]}}}, b"")

    client._request = request
    rows = client.resolve_candidates(["c1"])
    assert seen == {"cmd": "resolve_candidates", "candidate_ids": ["c1"]}
    assert rows["c1"]["found"] is True
