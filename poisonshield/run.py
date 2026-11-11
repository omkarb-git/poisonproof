"""Convenience launcher: python run.py."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP = PROJECT_ROOT / "app.py"


def main() -> None:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print("Streamlit is not installed. Install dependencies with: pip install -r requirements.txt")
        raise SystemExit(1)

    sys.argv = [
        "streamlit",
        "run",
        str(APP),
        "--server.headless=true",
        f"--server.maxUploadSize=2048",
    ]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
