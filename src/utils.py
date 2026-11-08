from enum import Enum


class Color(Enum):
    """Represent ANSI color escape codes for terminal messages."""

    BLUE = "\033[1;94m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[1;91m"
    WHITE = "\033[1;97m"
    YELLOW = "\033[1;93m"
    GREEN = "\033[1;92m"
    MAGENTA = "\033[1;95m"
    CYAN = "\033[1;96m"
    RST = "\033[0m"
