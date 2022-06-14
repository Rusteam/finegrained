"""Main module."""
import fire

from finegrained import data, models, frontend


def main():
    fire.Fire({"data": data, "models": models, "frontend": frontend})


if __name__ == "__main__":
    main()
