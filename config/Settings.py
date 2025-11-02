# Config/Settings.py
from pathlib import Path

class Settings:
    ROOT = Path(__file__).resolve().parents[1]  # AutoGluonProje/
    PAGE_TITLE = "Code Cosmos - ML"
    PAGE_ICON = ROOT / "Assets" / "CodeCosmosIcon.png"  # Path objesi
    LAYOUT = "wide"
