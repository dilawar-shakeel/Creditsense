"""Entry point: `python -m creditsense.mcp_server [--transport stdio|streamable-http]`.

Defaults to stdio -- what a local MCP client (Claude Desktop, Claude Code) connects
to by launching this as a subprocess. `--transport streamable-http` runs it as a
long-lived HTTP service instead, e.g. for the docker-compose `mcp` service or the
future frontend.
"""

from __future__ import annotations

import argparse

from creditsense.config import get_settings
from creditsense.mcp_server.server import build_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport", choices=["stdio", "streamable-http"], default="stdio",
        help="stdio for a local MCP client subprocess (default); streamable-http to "
             "run as a long-lived HTTP service.",
    )
    args = parser.parse_args()

    settings = get_settings()
    server = build_server()

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    main()
