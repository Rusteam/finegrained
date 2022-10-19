"""Main module."""
import fire

from finegrained import data, models, services


def main():
    fire.Fire({"data": data, "models": models, "services": services})


if __name__ == "__main__":
    main()
