import pytest

def demo():
    return 1+1

def fibonacci(n: int) -> int:
    a, b = 1, 1
    rounds = int((n - 1) / 2)
    for r in range(rounds):
        a = a + b
        b = a + b
    return b if (n + 1) % 2 else a

def test_demo():
    assert demo() == 2

def test_fibonacci():
    assert fibonacci(5) == 5

if __name__ == "__main__":
    test_demo()
    test_fibonacci()