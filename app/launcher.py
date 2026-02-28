import os
import sys
import time
import threading
from pathlib import Path

import psutil

PORT = 8501
IDLE_SHUTDOWN_SEC = 15

# streamlit import前に固定
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
os.environ["STREAMLIT_DEVELOPMENT_MODE"] = "false"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
os.environ["STREAMLIT_SERVER_PORT"] = str(PORT)
os.environ["STREAMLIT_SERVER_HEADLESS"] = "false"
os.environ["STREAMLIT_SERVER_BASE_URL_PATH"] = ""
os.environ["STREAMLIT_BROWSER_SERVER_ADDRESS"] = "127.0.0.1"
os.environ["STREAMLIT_BROWSER_SERVER_PORT"] = str(PORT)
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

def _auto_shutdown_when_no_client(log_path: Path):
    had_client = False
    no_client_since = None
    my_pid = os.getpid()

    while True:
        try:
            conns = []
            for c in psutil.net_connections(kind="tcp"):
                if c.pid != my_pid:
                    continue
                if not c.laddr or c.laddr.port != PORT:
                    continue
                if not c.raddr:
                    continue
                # ブラウザ接続をクライアントとして扱う
                if c.status in ("ESTABLISHED", "CLOSE_WAIT"):
                    conns.append(c)

            if conns:
                had_client = True
                no_client_since = None
            else:
                if had_client:
                    if no_client_since is None:
                        no_client_since = time.time()
                    elif time.time() - no_client_since >= IDLE_SHUTDOWN_SEC:
                        log_path.write_text("auto shutdown\n", encoding="utf-8")
                        os._exit(0)

        except Exception as e:
            # --windowed でも原因が追えるようにログ
            log_path.write_text(f"watcher error: {e}\n", encoding="utf-8")

        time.sleep(1)

def main() -> int:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    app_file = base_dir / "app" / "app.py"
    if not app_file.exists():
        raise FileNotFoundError(f"app.py not found: {app_file}")

    log_path = base_dir / "launcher_diag.txt"

    t = threading.Thread(target=_auto_shutdown_when_no_client, args=(log_path,), daemon=True)
    t.start()

    import streamlit.config as st_config
    st_config.set_option("global.developmentMode", False)
    st_config.set_option("server.address", "127.0.0.1")
    st_config.set_option("server.port", PORT)
    st_config.set_option("server.baseUrlPath", "")
    st_config.set_option("browser.serverAddress", "127.0.0.1")
    st_config.set_option("browser.serverPort", PORT)

    # 診断ログ
    (base_dir / "launcher_diag.txt").write_text(
        f"developmentMode={st_config.get_option('global.developmentMode')}\n"
        f"server.port={st_config.get_option('server.port')}\n"
        f"browser.serverPort={st_config.get_option('browser.serverPort')}\n"
        f"app_file={app_file}\n",
        encoding="utf-8",
    )

    from streamlit.web import bootstrap
    bootstrap.run(str(app_file), False, [], {})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())