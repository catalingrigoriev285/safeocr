"""Entrypoint referenced by fastmcp.json (`fastmcp run fastmcp.json`).

FastMCP puts this file's folder (server/) on sys.path, so the `safeocr` package imports
even without `pip install -e .`.
"""

from safeocr.mcp_server import mcp

__all__ = ["mcp"]
