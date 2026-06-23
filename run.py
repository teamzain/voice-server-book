"""Dev entrypoint:  python run.py  (or:  flask --app app run)"""

from app import create_app
from app.config import get_config

app = create_app()

if __name__ == "__main__":
    cfg = get_config()
    app.run(host="0.0.0.0", port=cfg.port, debug=True)
