"""Serve the contract-rag API with uvicorn.

The API key comes from the APP_API_KEY environment variable. Without one the server refuses to start,
unless `--no-auth` is given for local development.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

from contract_rag.api.app import LOGGER_NAME, create_app
from contract_rag.api.service import QAService, load_config


def configure_logging(log_path: Path | None) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/api.yaml"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-auth", action="store_true", help="serve without an API key (local development)")
    parser.add_argument("--root-path", default="", help="URL prefix a proxy serves the API under, e.g. /api")
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_key = os.environ.get("APP_API_KEY") or None
    if api_key is None and not args.no_auth:
        raise SystemExit("set APP_API_KEY, or pass --no-auth to serve without authentication locally")
    configure_logging(cfg.log_path)
    service = QAService(cfg, Path.cwd().resolve())
    try:
        uvicorn.run(
            create_app(service, api_key),
            host=args.host or cfg.host,
            port=args.port or cfg.port,
            root_path=args.root_path,
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
