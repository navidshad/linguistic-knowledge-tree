"""Core engine — the portable inference kernel.

Pure knowledge-state logic: the static map (``loader``/``graph``), the evidence
pipeline (``events``/``evidence``/``forgetting``), GNN ``propagation``, the
``activation`` engine, ``status`` computation, and ``gaps`` detection. Depends on
nothing else inside ``klg_ai`` — the data, tuning, semantic, and eval layers all
build on this.
"""
