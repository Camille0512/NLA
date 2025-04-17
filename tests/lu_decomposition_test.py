import pytest

def demo():
    return 1+1

def test_demo():
    assert demo() == 2