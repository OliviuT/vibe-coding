"""Simple telemetry collection and ChatGPT analysis tool.

This module collects a snapshot of local system telemetry data and sends it to
OpenAI's Chat Completions API for lightweight analysis.  It is designed as a
minimal example of wiring telemetry into an LLM-powered workflow, so it focuses
on portability and standard-library dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class TelemetrySnapshot:
    """Container for telemetry data to make serialization explicit."""

    data: Dict[str, Any]

    def to_json(self) -> str:
        """Return a JSON-formatted string representation of the telemetry."""

        return json.dumps(self.data, indent=2, sort_keys=True)


class TelemetryCollector:
    """Collect a snapshot of local system telemetry data."""

    def collect(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            {
                "timestamp": time.time(),
                "platform": self._platform_info(),
                "load_average": self._load_average(),
                "memory": self._memory_info(),
                "process": self._process_info(),
            }
        )

    def _platform_info(self) -> Dict[str, Any]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }

    def _load_average(self) -> Optional[Dict[str, float]]:
        if hasattr(os, "getloadavg"):
            one, five, fifteen = os.getloadavg()
            return {"1min": one, "5min": five, "15min": fifteen}
        return None

    def _memory_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        meminfo_path = "/proc/meminfo"
        if os.path.exists(meminfo_path):
            info.update(self._parse_key_value_file(meminfo_path))
        return info

    def _process_info(self) -> Dict[str, Any]:
        usage = os.times()
        info = {
            "pid": os.getpid(),
            "cpu_time_user": usage.user,
            "cpu_time_system": usage.system,
            "cpu_time_children_user": usage.children_user,
            "cpu_time_children_system": usage.children_system,
            "elapsed": usage.elapsed,
        }

        status_path = f"/proc/{os.getpid()}/status"
        if os.path.exists(status_path):
            info.update(self._parse_key_value_file(status_path))
        return info

    @staticmethod
    def _parse_key_value_file(path: str) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    parsed[key.strip()] = value.strip()
        except OSError:
            # If we fail to read optional telemetry files we simply omit the data.
            pass
        return parsed


class ChatGPTTelemetryAnalyzer:
    """Send telemetry data to the ChatGPT API for high-level analysis."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        request_timeout: Optional[float] = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.request_timeout = request_timeout

    def analyze(self, snapshot: TelemetrySnapshot) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a telemetry analyst. Provide concise feedback "
                        "highlighting potential issues or anomalies."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Here is a telemetry snapshot. Summarize notable points and "
                        "suggest next steps if something looks problematic.\n\n"
                        f"```json\n{snapshot.to_json()}\n```"
                    ),
                },
            ],
            "temperature": 0.2,
        }

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"ChatGPT API returned HTTP {error.code}: {error.read().decode('utf-8', 'ignore')}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Failed to reach ChatGPT API: {error}") from error

        parsed = json.loads(body)
        analysis = self._extract_message(parsed)
        return {"analysis": analysis, "raw_response": parsed}

    @staticmethod
    def _extract_message(response_json: Mapping[str, Any]) -> str:
        choices = response_json.get("choices")
        if not isinstance(choices, Iterable):
            return ""
        for choice in choices:
            message = choice.get("message") if isinstance(choice, Mapping) else None
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
        return ""


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect telemetry and send it to the ChatGPT API for analysis.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="ChatGPT model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--endpoint",
        default="https://api.openai.com/v1/chat/completions",
        help="ChatGPT API endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (default: read from OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect telemetry and print it without calling the ChatGPT API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout for the API request in seconds (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    if not args.collect_only and not args.api_key:
        parser.error("An API key must be provided unless --collect-only is used.")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    collector = TelemetryCollector()
    snapshot = collector.collect()

    if args.collect_only:
        print(snapshot.to_json())
        return 0

    analyzer = ChatGPTTelemetryAnalyzer(
        api_key=args.api_key,
        model=args.model,
        endpoint=args.endpoint,
        request_timeout=args.timeout,
    )

    result = analyzer.analyze(snapshot)
    print(result["analysis"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
