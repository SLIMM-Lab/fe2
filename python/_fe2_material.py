"""Public standalone bridge; initialize Jem before Python worker calls.

The embedded Jive driver supplies a built-in module of the same name. There
Jem is already initialized by the host. Standalone Python must load the native
library on its main thread, including the library's static Jem objects.
"""
import os
import sys
import threading

on_main_thread = threading.current_thread() is threading.main_thread()
if sys.platform == "linux":
    # This also handles a first import of `threading` from a raw Python worker.
    on_main_thread = threading.get_native_id() == os.getpid()
if not on_main_thread:
    raise ImportError("Import _fe2_material on Python's main thread before using worker threads")

from _fe2_material_native import MaterialBatch, MaterialFactory

__all__ = ["MaterialBatch", "MaterialFactory"]
