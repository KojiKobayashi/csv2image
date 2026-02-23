import sys
from pathlib import Path
from streamlit.web import cli as stcli

def main() -> int:
    # PyInstaller展開時(_MEIPASS)と通常実行の両対応
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    app_file = base_dir / "app" / "app.py"

    sys.argv = [
        "streamlit",
        "run",
        str(app_file),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    return stcli.main()

if __name__ == "__main__":
    raise SystemExit(main())