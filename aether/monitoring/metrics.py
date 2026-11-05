"""Low-cardinality in-process metrics.

Deliberately simple: counters, gauges and latency histograms with a fixed label set. No
high-cardinality labels (prediction ids, file names, user ids) are ever used as metric keys,
which keeps cardinality bounded regardless of traffic.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass
class LatencySnapshot:
    """Summary of a latency series."""

    count: int
    mean: float
    p50: float
    p95: float
    max: float


class MetricsRegistry:
    """Thread-safe counters, gauges and bounded latency windows."""

    def __init__(self, max_samples: int = 5000) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.Lock()
        self._started = time.time()

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._latencies[name].append(float(value))

    def counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def gauge(self, name: str) -> float | None:
        with self._lock:
            return self._gauges.get(name)

    def latency(self, name: str) -> LatencySnapshot | None:
        with self._lock:
            values = sorted(self._latencies.get(name, ()))
        if not values:
            return None
        count = len(values)
        return LatencySnapshot(
            count=count,
            mean=sum(values) / count,
            p50=values[int(0.50 * (count - 1))],
            p95=values[min(count - 1, int(0.95 * count))],
            max=values[-1],
        )

    def snapshot(self) -> dict[str, Any]:
        """Full metric snapshot for ``/api/v1/metrics/summary``."""
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            latency_names = list(self._latencies)
        return {
            "uptime_seconds": round(time.time() - self._started, 2),
            "counters": {key: round(value, 4) for key, value in counters.items()},
            "gauges": {key: round(value, 4) for key, value in gauges.items()},
            "latency": {
                name: {
                    "count": snap.count,
                    "mean_ms": round(snap.mean, 3),
                    "p50_ms": round(snap.p50, 3),
                    "p95_ms": round(snap.p95, 3),
                    "max_ms": round(snap.max, 3),
                }
                for name in latency_names
                if (snap := self.latency(name)) is not None
            },
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._latencies.clear()
            self._started = time.time()


#: Process-wide registry used by the API and pipeline.
METRICS = MetricsRegistry()
