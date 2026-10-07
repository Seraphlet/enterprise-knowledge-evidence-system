"""Allow ``python -m knowledge_system`` to start the CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

