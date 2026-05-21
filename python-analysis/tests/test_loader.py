import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.loader import is_supported_format


class TestLoader:
    def test_supported_formats(self):
        assert is_supported_format("test.wav") is True
        assert is_supported_format("test.mp3") is True
        assert is_supported_format("test.flac") is True
        assert is_supported_format("test.ogg") is True

    def test_unsupported_format(self):
        assert is_supported_format("test.txt") is False
        assert is_supported_format("test.pdf") is False
