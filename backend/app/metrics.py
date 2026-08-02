"""Low-overhead in-process metrics for the local deployment profile."""

from collections import Counter, defaultdict
from threading import Lock


class MetricsRegistry:
    def __init__(self):
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._samples: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)
        self._lock = Lock()

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None):
        return name, tuple(sorted((labels or {}).items()))

    def increment(self, name: str, value: float = 1, **labels: str) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            samples = self._samples[self._key(name, labels)]
            samples.append(value)
            if len(samples) > 2048:
                del samples[:1024]

    def render_prometheus(self) -> str:
        def metric_line(name, labels, value):
            label_text = ""
            if labels:
                label_text = "{" + ",".join(f'{key}="{str(val).replace(chr(34), chr(39))}"' for key, val in labels) + "}"
            return f"{name}{label_text} {value}"

        with self._lock:
            lines = [metric_line(name, labels, value) for (name, labels), value in self._counters.items()]
            lines.extend(metric_line(name, labels, value) for (name, labels), value in self._gauges.items())
            for (name, labels), values in self._samples.items():
                lines.append(metric_line(f"{name}_count", labels, len(values)))
                lines.append(metric_line(f"{name}_sum", labels, round(sum(values), 6)))
        return "\n".join(lines) + "\n"


metrics = MetricsRegistry()
