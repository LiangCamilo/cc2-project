"""Legacy entry point: validate the definitive ETL outputs without writing data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zta.data.__main__ import main


if __name__ == "__main__":
    main()