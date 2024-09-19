# ANSI code colors python

RED = "\033[0;31m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
YELLOW = "\033[1;33m"

""" ANOTHER COLORS NOT IMPLEMENTED. """
# BLACK = "\033[0;30m"
# BROWN = "\033[0;33m"
# PURPLE = "\033[0;35m"
# CYAN = "\033[0;36m"
# LIGHT_GRAY = "\033[0;37m"
# DARK_GRAY = "\033[1;30m"
# LIGHT_RED = "\033[1;31m"
# LIGHT_GREEN = "\033[1;32m"
# LIGHT_BLUE = "\033[1;34m"
# LIGHT_PURPLE = "\033[1;35m"
# LIGHT_CYAN = "\033[1;36m"
# LIGHT_WHITE = "\033[1;37m"
# BOLD = "\033[1m"
# FAINT = "\033[2m"
# ITALIC = "\033[3m"
# UNDERLINE = "\033[4m"
# BLINK = "\033[5m"
# NEGATIVE = "\033[7m"
# CROSSED = "\033[9m"


def __reset_color(text: str):
    return text + "\u001b[0m"


def __convert_to_str(text) -> str:
    return str(text)


def red(text: str) -> str:
    text = __convert_to_str(text)
    return RED + __reset_color(text)


def green(text: str) -> str:
    text = __convert_to_str(text)
    return GREEN + __reset_color(text)


def yellow(text: str) -> str:
    text = __convert_to_str(text)
    return YELLOW + __reset_color(text)


def blue(text: str) -> str:
    text = __convert_to_str(text)
    return BLUE + __reset_color(text)