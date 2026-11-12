"""Minimal async client for a local Ollama server (/api/chat, streaming)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ..config import LLMSettings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, cfg: LLMSettings):
        self.cfg = cfg

    async def available(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(base_url=self.cfg.base_url, timeout=1.5) as http:
                tags = (await http.get("/api/tags")).json()
        except (httpx.HTTPError, ValueError):
            return {"reachable": False, "model_installed": False}
        names = {m.get("name") for m in tags.get("models", [])}
        return {"reachable": True, "model_installed": self.cfg.model in names}

    async def chat_stream(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield Ollama stream chunks: {"message": {"content", "tool_calls"?}, "done": bool}."""
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "think": self.cfg.think,
            "options": {"temperature": self.cfg.temperature},
        }
        if tools:
            payload["tools"] = tools
        try:
            async with httpx.AsyncClient(base_url=self.cfg.base_url, timeout=self.cfg.timeout) as http:
                async with http.stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode(errors="replace")
                        raise OllamaError(f"Ollama error {resp.status_code}: {body[:300]}")
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if "error" in chunk:
                            raise OllamaError(chunk["error"])
                        yield chunk
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Ollama is not reachable at {self.cfg.base_url}. Start it with `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError("The local model took too long to answer.") from exc
