"""Process-wide coordination for heavyweight ML model construction."""

from threading import RLock


# Transformers exposes model classes through a lazy module. Serializing first
# construction prevents concurrent CLIP consumers from racing that lazy import.
MODEL_LOAD_LOCK = RLock()
