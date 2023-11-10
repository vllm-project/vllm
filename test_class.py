class A:
    def __init__(self) -> None:
        self.hello()

    def hello(self):
        print("Hello, I'm A")


class B(A):
    def __init__(self) -> None:
        super().__init__()
        self.hello()

    def hello(self):
        print("Hello, I'm B")

b = B()