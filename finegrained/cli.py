"""Main module."""
import fire

from finegrained import data, models


def main():
    fire.Fire({"data": data, "models": models})


if __name__ == "__main__":
    main()
