from typing import Union

LIST_STR = Union[list[str], str]


def parse_list_str(value: LIST_STR) -> list[str]:
    if isinstance(value, str):
        return [value]
    elif isinstance(value, list):
        return value
    else:
        raise TypeError(f"{value=} is not str or list")
