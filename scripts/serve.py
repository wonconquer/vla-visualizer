"""Launch the local Streamlit application on one configured port."""

import os
import sys


def main():
    from streamlit.web import cli
    from vla_visualizer.config import ROOT
    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), "--server.address", os.environ.get("VLA_HOST", "127.0.0.1"),
                "--server.port", os.environ.get("VLA_PORT", "7860"),
                "--server.headless", "true", "--browser.gatherUsageStats", "false",
                "--server.fileWatcherType", "none"]
    cli.main()


if __name__ == "__main__":
    main()
