import json
from configparser import ConfigParser
from pathlib import Path
from typing import Optional, Union

import yaml

# TODO rename to io.py


def read_txt(filepath: str):
    return Path(filepath).read_text().strip().split("\n")


def parse_keyval_line(line: str) -> tuple:
    assert line.count("=") == 1, f"{line=} has to contain exactly on '='"
    key, val = line.split("=")
    return key, val


def read_file_config(file: str, section=None):
    if file.endswith((".txt", ".text")):
        config = read_txt_credentials(file)
    elif file.endswith(".ini"):
        config = read_ini_credentials(file, section=section)
    else:
        raise NotImplementedError(f"{file=} file extension not supported yet.")
    return config


def read_txt_credentials(path: str) -> dict:
    lines = read_txt(path)
    config = list(map(parse_keyval_line, lines))
    config = dict(config)
    return config


def read_ini_credentials(
    path: str, section: Optional[str] = None
) -> Union[dict, ConfigParser]:
    parser = ConfigParser()
    parser.read(path)
    if section:
        assert section in parser.sections()
        config = {k: v for k, v in parser[section].items()}
        return config
    else:
        return parser


def read_yaml(path: str | Path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def write_yaml(data: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def read_json(file: str) -> list | dict:
    with open(file) as f:
        data = json.load(f)
    return data


def write_json(data: list | dict, file: str):
    with open(file, "w") as f:
        json.dump(data, f)
