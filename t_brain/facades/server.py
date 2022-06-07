from typing import NoReturn
from ..server import app
from ..config import get_setting

setting = get_setting()


def run() -> None:
    port = 8080
    debug = False
    app.run(
        host=str(setting.server_ip),
        port=setting.server_port,
        debug=setting.server_debug,
    )
