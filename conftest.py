"""Put the repo root on sys.path so tests can `import config` and `from src import ...`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
