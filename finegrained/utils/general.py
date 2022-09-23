from finegrained.utils import types


def parse_list_str(value: types.LIST_STR_STR) -> types.LIST_STR:
    """Parses str ot list[str] into a list[str]

    Args:
        value: a str or a list[str]

    Returns:
        a list[str]
    """
    if isinstance(value, str):
        return [value]
    elif isinstance(value, (list, tuple)):
        return value
    else:
        raise TypeError(f"{value=} is not str or list")


def find_diff(left: list, right: list) -> list:
    return list(set(left).difference(right))
