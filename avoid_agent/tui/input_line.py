class InputLine:
    def __init__(self):
        self.text = ""
        # Where to place the next char in self.text
        self.cursor = 0

    # Movement

    def insert(self, char: str) -> None:
        self.text = self.text[: self.cursor] + char + self.text[self.cursor :]
        self.cursor += len(char)

    def move_left(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def move_right(self) -> None:
        if self.cursor < len(self.text):
            self.cursor += 1

    def move_home(self) -> None:
        self.cursor = 0

    def move_end(self) -> None:
        self.cursor = len(self.text)

    # Deletion

    def backspace(self) -> None:
        if self.cursor > 0:
            self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
            self.cursor -= 1

    def delete_forward(self) -> None:
        if self.cursor < len(self.text):
            self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]

    def kill_to_end(self) -> None:
        self.text = self.text[: self.cursor]

    def clear(self) -> str:
        value = self.text
        self.text = ""
        self.cursor = 0
        return value
