"""Shared image media-type sniffing for the vision providers."""

from __future__ import annotations

_MEDIA_SNIFF = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP
]


def media_type(image_bytes: bytes) -> str:
    for sig, mt in _MEDIA_SNIFF:
        if image_bytes.startswith(sig):
            return mt
    return "image/jpeg"
