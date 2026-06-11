"""Data layer — evidence sources and learner-data persistence.

Turns external data into the engine's ``Event`` stream / ``LearnerData``:
open-dataset ``adapters`` (Duolingo SLAM, synthetic), the SLAM ``dataset``
loader, and the file-backed learner ``profiles`` store. Depends only on
``klg_ai.core``.
"""
