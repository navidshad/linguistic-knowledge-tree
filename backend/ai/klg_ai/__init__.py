"""klg_ai — the Linguistic Knowledge Graph AI engine.

Pure-Python core (no web deps): load the static map, build the prerequisite
graph, fold a learner's events into per-node mastery (with forgetting + GNN
propagation), and compute status and gaps.

`propagation` (PyTorch Geometric) is imported lazily by the engine, so importing
this package — and running the parts that don't propagate — needs only networkx.
"""
from .activation import (
    DEFAULT_CONFIG,
    EngineConfig,
    LearnerProfile,
    TimelineFrame,
    activated_from_events,
    compute_mastery,
    event_span,
    get_activation,
    list_learners,
    mastery_from_events,
    mastery_timeline,
    threshold_activated,
)
from .adapters.synthetic import DEMO_KNOWN, demo_events, generate_events
from .events import SOURCE_WEIGHTS, Event
from .evidence import direct_scores
from .forgetting import recency_weight
from .gaps import connecting_subgraph, gap_scores, interior_gaps
from .graph import build_graph, default_graph
from .loader import DEFAULT_MAP_PATH, Edge, Node, SyntaxMap, load_map
from .profiles import (
    RESERVED_IDS,
    ProfileMeta,
    append_events,
    create_profile,
    delete_profile,
    list_profiles,
    load_conversation,
    load_events,
    load_profile,
    save_conversation,
)
from .status import Status, compute_status

__all__ = [
    # map + graph
    "DEFAULT_MAP_PATH", "load_map", "SyntaxMap", "Node", "Edge",
    "build_graph", "default_graph",
    # status + gaps
    "compute_status", "Status", "interior_gaps", "connecting_subgraph", "gap_scores",
    # evidence -> activation engine
    "Event", "SOURCE_WEIGHTS", "direct_scores", "recency_weight",
    "EngineConfig", "DEFAULT_CONFIG",
    "mastery_from_events", "activated_from_events", "threshold_activated",
    "get_activation", "compute_mastery", "mastery_timeline",
    "TimelineFrame", "event_span",
    "LearnerProfile", "list_learners",
    # synthetic evidence
    "DEMO_KNOWN", "demo_events", "generate_events",
    # persistent profiles
    "ProfileMeta", "RESERVED_IDS", "list_profiles", "load_profile", "load_events",
    "load_conversation", "create_profile", "delete_profile", "append_events",
    "save_conversation",
]
