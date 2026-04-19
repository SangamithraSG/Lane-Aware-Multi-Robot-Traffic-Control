#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║         Lane-Aware Multi-Robot Traffic Control System                     ║
║         ─────────────────────────────────────────────                     ║
║  A production-grade warehouse robot coordination simulator featuring:    ║
║    • Tarjan's SCC Deadlock Detection & Resolution                        ║
║    • Auction-Based Intersection Reservation                              ║
║    • Probabilistic Congestion Forecasting (Exponential Decay)            ║
║    • Elastic Trapezoidal Speed Profiles                                  ║
║    • Ghost Robot Future-Position Prediction                              ║
║    • Priority-Class Traffic Routing (URGENT / NORMAL / IDLE)             ║
║    • Real-Time Dual-Pane Animated Dashboard                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

Architecture
────────────
    ┌──────────────┐       ┌──────────────┐
    │  PathPlanner  │◄──────│  Congestion   │
    │  (A* multi-   │       │  Forecaster   │
    │   factor)     │       │  (exp. decay) │
    └──────┬───────┘       └──────▲───────┘
           │                      │
    ┌──────▼───────────────────────────────┐
    │         TrafficController             │
    │  • Reservation Table (auction-based)  │
    │  • Wait-Graph Construction            │
    │  • Safety Enforcement                 │
    │  • Ghost Prediction Engine            │
    └──────┬──────────────┬────────────────┘
           │              │
    ┌──────▼──────┐  ┌────▼──────────┐
    │  Deadlock    │  │   Visualizer   │
    │  Detector    │  │  (dual-pane    │
    │ (Tarjan SCC) │  │   matplotlib)  │
    └─────────────┘  └───────────────┘
           │
    ┌──────▼──────────────────────────┐
    │         Simulation Engine        │
    │  • Tick-based main loop          │
    │  • Robot state machine driver    │
    │  • Metrics collector             │
    └─────────────────────────────────┘

Data Flow (per tick):
  1. CongestionForecaster updates heatmap
  2. Robots processed by priority (URGENT → NORMAL → IDLE)
  3. Each robot: state-machine transition (IDLE→PLANNING→MOVING→WAITING→DONE)
  4. TrafficController enforces reservations, safety, lane rules
  5. DeadlockDetector runs Tarjan's SCC on wait-graph
  6. Visualizer renders dual-pane frame
"""

import logging
import math
import sys
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any

import heapq
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════════════
# NAMED CONSTANTS — No magic numbers
# ═══════════════════════════════════════════════════════════════════════════

BASE_SPEED            = 1.0    # Base robot speed (units/tick)
DEFAULT_ACCEL         = 0.3    # Acceleration rate (units/tick²)
DEFAULT_DECEL         = 0.5    # Deceleration rate (units/tick²)
MIN_CRAWL_SPEED       = 0.1    # Minimum non-zero speed
MAX_TICKS             = 300    # Maximum simulation ticks before forced stop
CONGESTION_ALPHA      = 0.3    # Exponential decay factor for congestion
CONGESTION_WINDOW     = 50     # Rolling window size for occupancy history
CONGESTION_THRESHOLD  = 0.7    # Score threshold to trigger proactive replan
SAFE_FOLLOWING_DIST   = 0.4    # Fraction of edge length considered safe gap
COLLISION_EMERGENCY   = 0.35   # Euclidean distance for emergency stop
COLLISION_NEAR_MISS   = 1.2    # Euclidean distance for near-miss logging
REPLAN_WAIT_TIMEOUT   = 12     # Ticks before forced replan while waiting
GHOST_LOOKAHEAD       = 3      # Ticks for ghost projection look-ahead
ANIMATION_INTERVAL_MS = 200    # Milliseconds between animation frames
NUM_ROBOTS            = 10     # Total robots in simulation

# Priority weights for auction bid calculation
PRIORITY_WEIGHTS: Dict[str, float] = {
    "URGENT": 3.0,
    "NORMAL": 1.0,
    "IDLE":   0.5,
}

# Lane type speed multipliers (fraction of BASE_SPEED)
LANE_SPEED_MULTIPLIERS: Dict[str, float] = {
    "NORMAL":      1.0,
    "INTERSECTION": 0.3,
    "NARROW":      0.5,
    "HUMAN_ZONE":  0.2,
}

# A* cost weights
CONGESTION_COST_WEIGHT = 2.0
SAFETY_PENALTIES: Dict[str, float] = {
    "NORMAL":       0.0,
    "INTERSECTION": 0.5,
    "NARROW":       0.3,
    "HUMAN_ZONE":   1.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class RobotState(Enum):
    """Robot state machine states."""
    IDLE     = auto()
    PLANNING = auto()
    MOVING   = auto()
    WAITING  = auto()
    DONE     = auto()


class PriorityClass(Enum):
    """Robot priority tiers — lower numeric value = higher priority."""
    URGENT = 1
    NORMAL = 2
    IDLE   = 3


class LaneType(Enum):
    """Categories of warehouse lanes with different speed and safety rules."""
    NORMAL       = auto()
    INTERSECTION = auto()
    NARROW       = auto()
    HUMAN_ZONE   = auto()


# ═══════════════════════════════════════════════════════════════════════════
# EVENT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CollisionEvent:
    """Record of a near-miss or emergency stop event."""
    tick: int
    location: Tuple[float, float]
    robots: Tuple[int, ...]
    event_type: str  # 'near_miss' | 'emergency_stop'


@dataclass
class DeadlockEvent:
    """Record of a deadlock detection and resolution."""
    tick: int
    robots_involved: List[int]
    sacrifice_robot: int
    resolution: str


@dataclass
class ReservationRequest:
    """An auction bid for an intersection reservation slot."""
    robot_id: int
    target: int        # intersection node ID
    bid_score: float
    tick: int


# ═══════════════════════════════════════════════════════════════════════════
# ROBOT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Robot:
    """
    Autonomous warehouse robot with full state machine and velocity profile.

    State Machine:
        IDLE → PLANNING → MOVING ⇄ WAITING → DONE

    Velocity Profile (trapezoidal):
        Accelerate → Cruise → Decelerate
        Adapts to lane speed limits, following distance, and look-ahead congestion.
    """
    id: int
    priority: PriorityClass
    start: int
    goal: int

    # State machine
    state: RobotState = RobotState.IDLE

    # Position tracking
    current_node: int   = -1
    next_node: int      = -1
    edge_progress: float = 0.0   # 0.0 = at current_node, 1.0 = at next_node

    # Path
    path: List[int]       = field(default_factory=list)
    path_index: int       = 0
    path_history: List[int] = field(default_factory=list)

    # Velocity profile
    current_speed: float = 0.0
    target_speed: float  = BASE_SPEED
    max_speed: float     = BASE_SPEED
    accel_rate: float    = DEFAULT_ACCEL
    decel_rate: float    = DEFAULT_DECEL

    # Timing metrics
    start_tick: int      = 0
    end_tick: int         = 0
    travel_time: int     = 0
    wait_time: int       = 0         # current consecutive wait
    total_wait_time: int = 0         # cumulative wait
    replan_count: int    = 0

    # Safety flags
    horn_active: bool          = False
    emergency_stopped: bool    = False
    waiting_for: Optional[int] = None   # robot ID blocking this one
    banned_next_node: Optional[int] = None # temporarily avoid this node during replan
    last_sacrificed_tick: int  = -100   # fairness cooldown
    replan_history: List[int]  = field(default_factory=list) # throttle replanning

    def __post_init__(self):
        self.current_node = self.start
        self.path_history = [self.start]

    @property
    def priority_weight(self) -> float:
        """Numeric weight used for bid calculations."""
        return PRIORITY_WEIGHTS[self.priority.name]

    def get_display_position(self, positions: Dict[int, Tuple[float, float]]
                             ) -> Tuple[float, float]:
        """
        Interpolated (x, y) for smooth visualization.
        Blends between current_node and next_node based on edge_progress.
        """
        if self.next_node >= 0 and self.edge_progress > 0.0:
            x1, y1 = positions[self.current_node]
            x2, y2 = positions[self.next_node]
            p = min(self.edge_progress, 1.0)
            return (x1 + (x2 - x1) * p, y1 + (y2 - y1) * p)
        return positions.get(self.current_node, (0.0, 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# LANE GRAPH — networkx DiGraph wrapper
# ═══════════════════════════════════════════════════════════════════════════

class LaneGraph:
    """
    20-node warehouse floor layout built on a networkx DiGraph.

    Topology includes:
      • Straight corridors          • T-junctions
      • 3 intersection nodes (I1, I2, I3) requiring reservation
      • 2 narrow lane segments (N1 node area) with speed cap
      • 1 human zone (H1–H2) with mandatory slow speed + horn
      • Mix of directed (one-way aisles) and bidirectional edges

    Node Layout (5×4 grid with labels):
        D1(0)──D2(1)──D3(2)──D4(3)      Dock row
        │      │      │      │
        C1(4)──I1(5)──C2(6)──C3(7)      Corridor row
        │      ↓      │      │
        A1(8)──A2(9)──I2(10)─A3(11)     Aisle row
        │      ↓      │      │
        S1(12)─N1(13)─S2(14)─S3(15)     Storage row
        │      ↓      │      │
        H1(16)─H2(17)─I3(18)─P1(19)     Bottom row
    """

    INTERSECTION_NODES: Set[int] = {5, 10, 18}
    NARROW_EDGES: Set[Tuple[int, int]] = {(12, 13), (13, 12), (13, 14), (14, 13)}
    HUMAN_ZONE_EDGES: Set[Tuple[int, int]] = {(16, 17), (17, 16), (17, 18), (18, 17)}

    def __init__(self):
        self.graph = nx.DiGraph()
        self.positions: Dict[int, Tuple[float, float]] = {}
        self.node_labels: Dict[int, str] = {}
        self._build_warehouse()

    def _build_warehouse(self):
        """Construct the full 20-node warehouse graph."""

        # ── Node positions (x, y) ──
        self.positions = {
             0: (0, 8),   1: (3, 8),   2: (6, 8),   3: (9, 8),
             4: (0, 6),   5: (3, 6),   6: (6, 6),   7: (9, 6),
             8: (0, 4),   9: (3, 4),  10: (6, 4),  11: (9, 4),
            12: (0, 2),  13: (3, 2),  14: (6, 2),  15: (9, 2),
            16: (0, 0),  17: (3, 0),  18: (6, 0),  19: (9, 0),
        }

        self.node_labels = {
             0: "D1",  1: "D2",  2: "D3",  3: "D4",
             4: "C1",  5: "I1",  6: "C2",  7: "C3",
             8: "A1",  9: "A2", 10: "I2", 11: "A3",
            12: "S1", 13: "N1", 14: "S2", 15: "S3",
            16: "H1", 17: "H2", 18: "I3", 19: "P1",
        }

        # Add nodes with metadata
        for nid, pos in self.positions.items():
            ntype = "intersection" if nid in self.INTERSECTION_NODES else "normal"
            self.graph.add_node(nid, pos=pos, label=self.node_labels[nid],
                                node_type=ntype)

        # ── Edge definitions: (u, v, bidirectional?) ──
        edge_defs = [
            # Horizontal rows (all bidirectional)
            (0, 1, True),  (1, 2, True),  (2, 3, True),    # Dock
            (4, 5, True),  (5, 6, True),  (6, 7, True),    # Corridor
            (8, 9, True),  (9, 10, True), (10, 11, True),  # Aisle
            (12, 13, True),(13, 14, True),(14, 15, True),   # Storage
            (16, 17, True),(17, 18, True),(18, 19, True),   # Bottom

            # Vertical connections
            (0, 4, True),                     # D1–C1 bidirectional
            (1, 5, True),                     # D2–I1 bidirectional
            (2, 6, True),                     # D3–C2 bidirectional
            (3, 7, True),                     # D4–C3 bidirectional
            (4, 8, True),                     # C1–A1 bidirectional
            (5, 9, False),                    # I1→A2 ONE-WAY (down)
            (6, 10, True),                    # C2–I2 bidirectional
            (7, 11, True),                    # C3–A3 bidirectional
            (8, 12, True),                    # A1–S1 bidirectional
            (9, 13, False),                   # A2→N1 ONE-WAY (down)
            (10, 14, True),                   # I2–S2 bidirectional
            (11, 15, True),                   # A3–S3 bidirectional
            (12, 16, True),                   # S1–H1 bidirectional
            (13, 17, False),                  # N1→H2 ONE-WAY (down)
            (14, 18, True),                   # S2–I3 bidirectional
            (15, 19, True),                   # S3–P1 bidirectional
        ]

        for u, v, bidir in edge_defs:
            self._add_directed_edge(u, v)
            if bidir:
                self._add_directed_edge(v, u)

    # ── internal helpers ──

    def _add_directed_edge(self, u: int, v: int):
        """Add one directed edge with computed length, type, and speed limit."""
        length     = self._euclidean(u, v)
        lane_type  = self._classify_lane(u, v)
        speed_lim  = LANE_SPEED_MULTIPLIERS[lane_type.name] * BASE_SPEED

        self.graph.add_edge(u, v,
                            length=length,
                            lane_type=lane_type,
                            speed_limit=speed_lim,
                            congestion_score=0.0)

    def _euclidean(self, u: int, v: int) -> float:
        x1, y1 = self.positions[u]
        x2, y2 = self.positions[v]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _classify_lane(self, u: int, v: int) -> LaneType:
        """Determine lane type (priority: human_zone > narrow > intersection > normal)."""
        edge = (u, v)
        if edge in self.HUMAN_ZONE_EDGES:
            return LaneType.HUMAN_ZONE
        if edge in self.NARROW_EDGES:
            return LaneType.NARROW
        if u in self.INTERSECTION_NODES or v in self.INTERSECTION_NODES:
            return LaneType.INTERSECTION
        return LaneType.NORMAL

    # ── public API ──

    def get_edge_data(self, u: int, v: int) -> dict:
        """Get all properties of edge (u, v)."""
        return self.graph.edges[u, v]

    def successors(self, node: int) -> List[int]:
        """Get outgoing neighbors."""
        return list(self.graph.successors(node))

    def get_lane_type(self, u: int, v: int) -> LaneType:
        return self.graph.edges[u, v]["lane_type"]

    def update_congestion_score(self, edge: Tuple[int, int], score: float):
        if self.graph.has_edge(*edge):
            self.graph.edges[edge]["congestion_score"] = score


# ═══════════════════════════════════════════════════════════════════════════
# CONGESTION FORECASTER — sliding window + exponential decay
# ═══════════════════════════════════════════════════════════════════════════

class CongestionForecaster:
    """
    Probabilistic congestion scoring with exponential decay.

    score_t = α × current_occupancy + (1 − α) × score_{t−1}

    Scores are amplified by a safety multiplier for hazardous lane types
    (intersections ×1.5, narrow ×1.3, human_zone ×2.0).
    """

    _SAFETY_MULT = {
        LaneType.NORMAL:       1.0,
        LaneType.INTERSECTION: 1.5,
        LaneType.NARROW:       1.3,
        LaneType.HUMAN_ZONE:   2.0,
    }

    def __init__(self, graph: LaneGraph,
                 alpha: float = CONGESTION_ALPHA,
                 window: int = CONGESTION_WINDOW):
        self.graph  = graph
        self.alpha  = alpha
        self.window = window
        self.occupancy_history: Dict[Tuple, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self.scores: Dict[Tuple, float] = defaultdict(float)

    def update(self, tick: int, occupied_edges: Set[Tuple[int, int]]):
        """Recompute congestion scores for every edge using EMA."""
        for edge in self.graph.graph.edges():
            occ = 1.0 if edge in occupied_edges else 0.0
            self.occupancy_history[edge].append(occ)

            prev  = self.scores[edge]
            raw   = self.alpha * occ + (1.0 - self.alpha) * prev
            mult  = self._SAFETY_MULT[self.graph.get_lane_type(*edge)]
            score = min(raw * mult, 2.0)          # cap to avoid runaway

            self.scores[edge] = score
            self.graph.update_congestion_score(edge, score)

    def get_score(self, edge: Tuple[int, int]) -> float:
        return self.scores.get(edge, 0.0)

    def forecast_path(self, path: List[int], hops: int = 3) -> float:
        """
        Returns the **maximum** congestion score over the next `hops` edges
        of `path`, used to decide whether proactive rerouting is needed.
        """
        worst = 0.0
        for i in range(min(hops, len(path) - 1)):
            worst = max(worst, self.get_score((path[i], path[i + 1])))
        return worst


# ═══════════════════════════════════════════════════════════════════════════
# DEADLOCK DETECTOR — Tarjan's Strongly Connected Components
# ═══════════════════════════════════════════════════════════════════════════

class DeadlockDetector:
    """
    Detects deadlocks by running Tarjan's SCC on the robot wait-graph.

    Wait-Graph:
        Node = robot ID
        Edge A → B = "Robot A is blocked, waiting for Robot B to move"
        SCC of size > 1 = deadlock cycle

    Resolution Strategy:
        1. Select the lowest-priority robot in the cycle
        2. Tie-break: longest cumulative wait time
        3. Force the "sacrifice" robot to backtrack one node and replan
    """

    def __init__(self):
        self.events: List[DeadlockEvent] = []
        self.cycle_history: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

    def find_deadlocks(self, wait_graph: Dict[int, Optional[int]]) -> List[List[int]]:
        """
        Run Tarjan's SCC on the wait-graph.

        Args:
            wait_graph: robot_id → robot_id it is waiting for (None if not waiting)

        Returns:
            List of deadlock cycles (each a list of robot IDs).
        """
        # Build adjacency list
        adj:   Dict[int, List[int]] = defaultdict(list)
        nodes: Set[int] = set()

        for rid, blocked_by in wait_graph.items():
            nodes.add(rid)
            if blocked_by is not None:
                adj[rid].append(blocked_by)
                nodes.add(blocked_by)

        # Tarjan's algorithm variables
        index_counter = [0]
        stack:    List[int] = []
        on_stack: Set[int]  = set()
        indices:  Dict[int, int] = {}
        lowlinks: Dict[int, int] = {}
        sccs:     List[List[int]] = []

        def _strongconnect(v: int):
            indices[v]  = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in adj.get(v, []):
                if w not in indices:
                    _strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                component: List[int] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                if len(component) > 1:
                    sccs.append(component)

        for node in nodes:
            if node not in indices:
                _strongconnect(node)

        return sccs

    def resolve(self, cycle: List[int], robots: Dict[int, "Robot"],
                tick: int) -> Optional[int]:
        """
        Pick the sacrifice robot and log the event.

        Returns:
            Robot ID chosen for backtrack, or None if cycle is empty.
        """
        if not cycle:
            return None

        # Fairness: filter out recently sacrificed robots
        cooldown = 25
        valid_candidates = [r for r in cycle if tick - robots[r].last_sacrificed_tick > cooldown]
        if not valid_candidates:
            valid_candidates = cycle

        # Highest priority.value = lowest importance → sacrifice first
        # Tie-break: lowest wait time first to prevent starving long-waiters
        sacrifice_id = max(
            valid_candidates,
            key=lambda rid: (robots[rid].priority.value,
                             -robots[rid].total_wait_time)
        )
        robots[sacrifice_id].last_sacrificed_tick = tick

        msg = (f"Deadlock resolved @ tick {tick}: cycle={cycle}, "
               f"sacrifice=R{sacrifice_id} "
               f"({robots[sacrifice_id].priority.name})")
        self.events.append(DeadlockEvent(
            tick=tick,
            robots_involved=list(cycle),
            sacrifice_robot=sacrifice_id,
            resolution=msg,
        ))
        logging.info(msg)
        return sacrifice_id


# ═══════════════════════════════════════════════════════════════════════════
# PATH PLANNER — A* with multi-factor cost
# ═══════════════════════════════════════════════════════════════════════════

class PathPlanner:
    """
    A* pathfinder with congestion-aware multi-factor cost:

        f(n) = g(n) + h(n) + congestion_penalty + safety_penalty

    Where:
      g(n)               = actual cost from start (edge length)
      h(n)               = Euclidean heuristic to goal
      congestion_penalty = forecasted congestion × weight
      safety_penalty     = additional cost for hazardous lane types
    """

    def __init__(self, graph: LaneGraph, forecaster: CongestionForecaster):
        self.graph      = graph
        self.forecaster = forecaster

    def plan(self, start: int, goal: int,
             blocked_edges: Optional[Set[Tuple[int, int]]] = None,
             penalized_edges: Optional[Set[Tuple[int, int]]] = None,
             banned_next_node: Optional[int] = None,
             replan_count: int = 0,
             banned_edges: Optional[Set[Tuple[int, int]]] = None,
             deadlock_edge_penalty: Optional[Dict[Tuple[int, int], float]] = None,
             personal_penalties: Optional[Dict[Tuple[int, int], float]] = None
             ) -> Optional[List[int]]:
        """
        Compute optimal path from start to goal.

        Args:
            blocked_edges: edges to treat as impassable
            penalized_edges: edges that incur heavy soft limits (+15.0 Ghost Penalty)

        Returns:
            Ordered list of node IDs, or None if no path exists.
        """
        if start == goal:
            return [start]

        blocked = blocked_edges or set()
        counter = 0
        # (f_score, tie-break counter, node, path)
        open_heap: List[Tuple[float, int, int, List[int]]] = [
            (0.0, counter, start, [start])
        ]
        best_g: Dict[int, float] = {start: 0.0}
        closed: Set[int] = set()

        while open_heap:
            f, _, cur, path = heapq.heappop(open_heap)

            if cur == goal:
                return path
            if cur in closed:
                continue
            closed.add(cur)

            for nbr in self.graph.successors(cur):
                edge = (cur, nbr)
                if (blocked_edges and edge in blocked_edges):
                    continue

                edata = self.graph.get_edge_data(cur, nbr)

                # Multi-factor g cost
                base     = edata["length"]
                cong_pen = self.forecaster.get_score(edge) * CONGESTION_COST_WEIGHT
                safe_pen = SAFETY_PENALTIES[edata["lane_type"].name]
                
                # Soft logic overrides
                if cur == start and banned_next_node is not None and nbr == banned_next_node:
                    safe_pen += 50.0 + (replan_count * 5.0)
                if penalized_edges and edge in penalized_edges:
                    safe_pen += 50.0 + (replan_count * 5.0)
                if deadlock_edge_penalty:
                    safe_pen += deadlock_edge_penalty.get(edge, 0.0)
                if personal_penalties:
                    safe_pen += personal_penalties.get(edge, 0.0)

                # Deterministic tie-breaker
                noise = hash((start, cur, nbr)) % 100 * 0.01
                g_new    = best_g[cur] + base + cong_pen + safe_pen + noise

                if nbr not in best_g or g_new < best_g[nbr]:
                    best_g[nbr] = g_new
                    h = self._heuristic(nbr, goal)
                    counter += 1
                    heapq.heappush(open_heap,
                                   (g_new + h, counter, nbr, path + [nbr]))

        return None  # no path

    def _heuristic(self, node: int, goal: int) -> float:
        """Admissible Euclidean distance heuristic."""
        x1, y1 = self.graph.positions[node]
        x2, y2 = self.graph.positions[goal]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# TRAFFIC CONTROLLER — central coordinator
# ═══════════════════════════════════════════════════════════════════════════

class TrafficController:
    """
    Central coordinator managing all traffic subsystems:
      • Auction-based intersection reservation
      • Edge and node occupancy tracking
      • Safety enforcement (following distance, emergency stop, lane rules)
      • Ghost robot future-position prediction
      • Wait-graph construction for deadlock detection
    """

    def __init__(self, graph: LaneGraph, planner: PathPlanner,
                 forecaster: CongestionForecaster, detector: DeadlockDetector):
        self.graph      = graph
        self.planner    = planner
        self.forecaster = forecaster
        self.detector   = detector

        # Reservation table: intersection_node → robot_id
        self.reservations: Dict[int, int] = {}

        # Wait queues per intersection, sorted by descending bid
        self.wait_queues: Dict[int, List[ReservationRequest]] = defaultdict(list)

        # Occupancy maps
        self.edge_occ: Dict[Tuple[int, int], int] = {}   # edge → robot_id
        self.node_occ: Dict[int, int]             = {}   # node → robot_id

        # Safety logs
        self.collision_log: List[CollisionEvent]  = []
        self.near_miss_log: List[CollisionEvent]  = []

        # System metrics
        self.throughput_history: List[float] = []
        self.completed_count = 0
        self.deadlock_count  = 0
        self.conflict_memory: Dict[Tuple[int, int], int] = defaultdict(int)
        self.target_occ: Dict[int, int] = {}
        self.deadlock_edge_penalty: Dict[Tuple[int, int], float] = defaultdict(float)
        self.personal_penalties: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
        self.estop_memory: Dict[Tuple[int, int], int] = {}

    # ── occupancy helpers ──

    def register_node(self, node: int, rid: int):
        self.node_occ[node] = rid

    def unregister_node(self, node: int, rid: int):
        if self.node_occ.get(node) == rid:
            del self.node_occ[node]

    def register_edge(self, edge: Tuple[int, int], rid: int):
        self.edge_occ[edge] = rid

    def unregister_edge(self, edge: Tuple[int, int], rid: int):
        if self.edge_occ.get(edge) == rid:
            del self.edge_occ[edge]

    def register_target(self, node: int, rid: int):
        self.target_occ[node] = rid

    def unregister_target(self, node: int, rid: int):
        if self.target_occ.get(node) == rid:
            del self.target_occ[node]

    def is_node_free(self, node: int, rid: int) -> bool:
        occ = self.node_occ.get(node)
        return occ is None or occ == rid

    def is_edge_free(self, edge: Tuple[int, int], rid: int) -> bool:
        occ = self.edge_occ.get(edge)
        return occ is None or occ == rid

    # ── auction-based intersection reservation ──

    def request_intersection(self, robot: Robot, node: int, tick: int) -> bool:
        """
        Attempt to reserve an intersection node via auction.

        If already reserved by another robot:
          → robot's bid is queued; returns False (must wait)
        Otherwise:
          → reservation granted; returns True
        """
        if node not in LaneGraph.INTERSECTION_NODES:
            return True

        if node in self.reservations:
            holder = self.reservations[node]
            if holder == robot.id:
                return True           # already own it
            
            # Occupied — queue a bid
            bid = self._compute_bid(robot, tick)
            req = ReservationRequest(robot.id, node, bid, tick)
            q = [r for r in self.wait_queues[node] if r.robot_id != robot.id]
            q.append(req)
            # Tiebreak by ID so queue is strictly deterministic
            q.sort(key=lambda r: (r.bid_score, -r.robot_id), reverse=True)
            self.wait_queues[node] = q
            return False

        # Grant reservation
        self.reservations[node] = robot.id
        logging.debug(f"Intersection {node} reserved by R{robot.id}")
        return True

    def release_intersection(self, node: int, rid: int, tick: int):
        """Release reservation; grant to next highest bidder if queued."""
        if node in self.reservations and self.reservations[node] == rid:
            del self.reservations[node]
            q = self.wait_queues.get(node, [])
            if q:
                nxt = q.pop(0)
                self.reservations[node] = nxt.robot_id
                self.wait_queues[node] = q
                logging.info(f"Intersection {node} granted to R{nxt.robot_id} "
                             f"(bid={nxt.bid_score:.2f})")

    def _compute_bid(self, robot: Robot, tick: int) -> float:
        """
        Auction bid = priority_weight × (1 / estimated_wait) × urgency_factor.

        Urgency increases the longer the robot has been travelling relative
        to elapsed simulation time.
        """
        if robot.wait_time > 50:
            return float('inf')

        est_wait = max(1, robot.wait_time + 1)
        urgency  = 1.0 + (robot.travel_time / max(1, tick)) * 0.5
        # FIXED inversion: Wait time now exponentially boosts the bid score
        return robot.priority_weight * est_wait * urgency

    # ── safety systems ──

    def check_following_distance(self, robot: Robot,
                                 robots: Dict[int, Robot]) -> bool:
        """
        Returns True if the lane ahead is clear within safe following distance.
        Checks both same-edge-ahead robots and destination-node occupants.
        """
        if robot.next_node < 0:
            return True

        # Check destination node occupancy by a stationary robot
        if not self.is_node_free(robot.next_node, robot.id):
            occ_id = self.node_occ.get(robot.next_node)
            if occ_id is not None:
                occ = robots.get(occ_id)
                if occ and occ.state in (RobotState.WAITING, RobotState.IDLE):
                    if robot.edge_progress > (1.0 - SAFE_FOLLOWING_DIST):
                        return False

        # Check same-edge, same-direction, ahead
        for oid, other in robots.items():
            if oid == robot.id or other.state != RobotState.MOVING:
                continue
            if (other.current_node == robot.current_node
                    and other.next_node == robot.next_node
                    and other.edge_progress > robot.edge_progress):
                gap = other.edge_progress - robot.edge_progress
                if gap < SAFE_FOLLOWING_DIST:
                    return False
        return True

    def check_emergency_stop(self, robot: Robot, robots: Dict[int, Robot],
                              tick: int) -> Optional[int]:
        """
        Emergency stop: triggered when two robots are within COLLISION_EMERGENCY
        Euclidean distance. Also logs near-miss events.
        Returns True if emergency stop is needed.
        """
        pos = robot.get_display_position(self.graph.positions)

        for oid, other in robots.items():
            if oid == robot.id or other.state == RobotState.DONE:
                continue

            # Only check localized physical neighbors
            if not (robot.current_node == other.current_node or 
                    robot.next_node == other.next_node or
                    robot.current_node == other.next_node or
                    robot.next_node == other.current_node):
                continue

            opos = other.get_display_position(self.graph.positions)
            dist = math.hypot(pos[0] - opos[0], pos[1] - opos[1])

            if dist < COLLISION_EMERGENCY:
                pair = (min(robot.id, oid), max(robot.id, oid))
                last_estop = self.estop_memory.get(pair, -100)
                if tick - last_estop < 5:
                    return None
                self.estop_memory[pair] = tick
                
                self.collision_log.append(CollisionEvent(
                    tick=tick, location=pos,
                    robots=(robot.id, oid), event_type="emergency_stop"))
                logging.warning(f"⚠ EMERGENCY STOP: R{robot.id} ↔ R{oid} "
                                f"dist={dist:.2f}")
                self.conflict_memory[pair] += 1
                return oid

            if dist < COLLISION_NEAR_MISS:
                self.near_miss_log.append(CollisionEvent(
                    tick=tick, location=pos,
                    robots=(robot.id, oid), event_type="near_miss"))
        return None

    # ── ghost robot prediction ──

    def ghost_predict_clear(self, robot: Robot, robots: Dict[int, Robot],
                            path: List[int], start_idx: int) -> bool:
        """
        Simulate "ghost" projections of all other robots for GHOST_LOOKAHEAD ticks.
        Returns True if no conflict detected along robot's planned path segment.
        """
        for oid, other in robots.items():
            if oid == robot.id or other.state == RobotState.DONE:
                continue

            # Project other robot's future nodes
            future: Set[int] = set()
            if other.path and other.path_index < len(other.path):
                for k in range(min(GHOST_LOOKAHEAD,
                                   len(other.path) - other.path_index)):
                    future.add(other.path[other.path_index + k])
            else:
                future.add(other.current_node)

            # Check our planned nodes against their projection
            for k in range(min(GHOST_LOOKAHEAD, len(path) - start_idx)):
                if path[start_idx + k] in future:
                    logging.debug(f"Ghost conflict: R{robot.id} node "
                                  f"{path[start_idx + k]} vs R{oid}")
                    return False
        return True

    # ── wait-graph builder ──

    def build_wait_graph(self, robots: Dict[int, Robot]) -> Dict[int, Optional[int]]:
        """Construct directed wait-graph for Tarjan's deadlock detection."""
        wg: Dict[int, Optional[int]] = {}
        for rid, r in robots.items():
            if r.state == RobotState.WAITING and r.waiting_for is not None:
                wg[rid] = r.waiting_for
        return wg


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZER — dual-pane matplotlib animation
# ═══════════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Real-time dual-pane dashboard using matplotlib:

    Left pane (60%):
      - Warehouse graph with directed edges
      - Edge color = congestion heatmap (green → yellow → red)
      - Nodes colored by type (intersection/narrow/human/normal)
      - Robots as priority-colored markers with velocity arrows
      - Reservation highlights around intersection nodes

    Right pane (3 panels):
      - Throughput curve over time
      - Per-robot wait-time bar chart
      - Live statistics text
    """

    _PRIORITY_CLR = {
        PriorityClass.URGENT: "#FF4444",
        PriorityClass.NORMAL: "#4488FF",
        PriorityClass.IDLE:   "#66BB66",
    }
    _STATE_MKR = {
        RobotState.IDLE:     "o",
        RobotState.PLANNING: "s",
        RobotState.MOVING:   "D",
        RobotState.WAITING:  "X",
        RobotState.DONE:     "*",
    }

    def __init__(self, graph: LaneGraph):
        self.graph = graph
        self.fig   = None
        self.axes: Dict[str, Any] = {}
        self._init_figure()

    def _init_figure(self):
        self.fig = plt.figure(figsize=(19, 10), facecolor="#0f0f23")
        self.fig.canvas.manager.set_window_title(
            "Lane-Aware Multi-Robot Traffic Control System")

        self.fig.suptitle(
            "LANE-AWARE  MULTI-ROBOT  TRAFFIC  CONTROL  SYSTEM",
            fontsize=15, fontweight="bold", color="#00ffcc",
            family="monospace", y=0.98)

        gs = GridSpec(3, 2, figure=self.fig, width_ratios=[3, 1],
                      hspace=0.35, wspace=0.15)

        self.axes["graph"]      = self.fig.add_subplot(gs[:, 0])
        self.axes["throughput"] = self.fig.add_subplot(gs[0, 1])
        self.axes["delay"]      = self.fig.add_subplot(gs[1, 1])
        self.axes["stats"]      = self.fig.add_subplot(gs[2, 1])

        for name, ax in self.axes.items():
            ax.set_facecolor("#12122a")
            for spine in ax.spines.values():
                spine.set_color("#333366")
            ax.tick_params(colors="#666688", labelsize=7)

        self.axes["stats"].axis("off")
        plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])

    # ── per-frame drawing ──

    def draw(self, tick: int, robots: Dict[int, Robot],
             ctrl: TrafficController, forecaster: CongestionForecaster,
             sim_running: bool):
        """Render one complete frame."""
        self._draw_graph(tick, robots, ctrl, forecaster, sim_running)
        self._draw_throughput(ctrl)
        self._draw_delays(robots)
        self._draw_stats(tick, robots, ctrl)

    def _draw_graph(self, tick, robots, ctrl, forecaster, sim_running):
        ax = self.axes["graph"]
        ax.clear()
        ax.set_facecolor("#12122a")
        status = "RUNNING" if sim_running else "COMPLETE"
        ax.set_title(f"Warehouse Layout — Tick {tick}  [{status}]",
                     color="white", fontsize=11, family="monospace")

        pos = self.graph.positions

        # ── edges (congestion heatmap) ──
        for u, v in self.graph.graph.edges():
            score = forecaster.get_score((u, v))
            ec    = self._heat_color(score)
            lw    = 3.0 if (u, v) in ctrl.edge_occ else 1.4

            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy) or 1
            ox, oy = -dy / d * 0.12, dx / d * 0.12  # slight offset

            ax.annotate(
                "", xy=(x2 + ox, y2 + oy), xytext=(x1 + ox, y1 + oy),
                arrowprops=dict(arrowstyle="->", color=ec, lw=lw,
                                connectionstyle="arc3,rad=0.04"))

        # ── nodes ──
        for nid, (x, y) in pos.items():
            ntype = self.graph.graph.nodes[nid].get("node_type", "normal")
            nc    = "#FFB347" if ntype == "intersection" else "#6EC6FF"
            edc   = "white"

            if nid == 13:           # narrow
                nc, edc = "#A0A0A0", "#FFA500"
            elif nid in (16, 17):   # human zone
                nc, edc = "#FFE4B5", "#FF6347"

            # reservation glow
            if nid in ctrl.reservations:
                ax.add_patch(plt.Circle((x, y), 0.55, fill=False,
                             edgecolor="#FFD700", lw=2, ls="--", zorder=4))

            ax.plot(x, y, "o", ms=22, color=nc,
                    mec=edc, mew=1.5, zorder=5)
            ax.text(x, y, self.graph.node_labels[nid],
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="black", zorder=6)

        # ── robots ──
        for rid, rob in robots.items():
            if rob.state == RobotState.DONE:
                gx, gy = pos[rob.goal]
                ax.plot(gx, gy + 0.65, "*", ms=13,
                        color=self._PRIORITY_CLR[rob.priority],
                        mec="white", mew=0.5, zorder=8)
                continue

            rx, ry = rob.get_display_position(pos)
            clr    = self._PRIORITY_CLR[rob.priority]
            mkr    = self._STATE_MKR[rob.state]

            ax.plot(rx, ry, mkr, ms=15, color=clr,
                    mec="white", mew=1.5, zorder=10)
            ax.text(rx, ry + 0.45, f"R{rid}", ha="center", va="bottom",
                    fontsize=7, color="white", fontweight="bold", zorder=11)

            # velocity vector arrow
            if rob.state == RobotState.MOVING and rob.current_speed > 0 and rob.next_node >= 0:
                tx, ty = pos[rob.next_node]
                dx, dy = tx - rx, ty - ry
                d = math.hypot(dx, dy) or 1
                alen = rob.current_speed * 0.6
                ax.annotate("", xy=(rx + dx / d * alen, ry + dy / d * alen),
                            xytext=(rx, ry),
                            arrowprops=dict(arrowstyle="->", color=clr, lw=2))

            # indicators
            if rob.horn_active:
                ax.text(rx + 0.35, ry - 0.35, "♪", fontsize=10,
                        color="#FFD700", zorder=11)
            if rob.emergency_stopped:
                ax.text(rx - 0.35, ry - 0.35, "⛔", fontsize=9, zorder=11)

        # ── legend ──
        handles = [
            mpatches.Patch(color="#FF4444", label="URGENT"),
            mpatches.Patch(color="#4488FF", label="NORMAL"),
            mpatches.Patch(color="#66BB66", label="IDLE"),
            mpatches.Patch(color="#FFB347", label="Intersection"),
            mpatches.Patch(color="#A0A0A0", label="Narrow"),
            mpatches.Patch(color="#FFE4B5", label="Human Zone"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=7,
                  facecolor="#0f0f23", edgecolor="#333366",
                  labelcolor="white")

        ax.set_xlim(-1.5, 10.5)
        ax.set_ylim(-1.5, 9.5)
        ax.set_aspect("equal")

    def _draw_throughput(self, ctrl):
        ax = self.axes["throughput"]
        ax.clear()
        ax.set_facecolor("#12122a")
        ax.set_title("Throughput (completed/tick)", color="white", fontsize=9,
                     family="monospace")
        if ctrl.throughput_history:
            ax.plot(ctrl.throughput_history, color="#00ffaa", lw=1.5)
            ax.fill_between(range(len(ctrl.throughput_history)),
                            ctrl.throughput_history, alpha=0.15, color="#00ffaa")
        ax.tick_params(colors="#666688", labelsize=6)
        ax.set_xlabel("tick", color="#666688", fontsize=7)
        for s in ax.spines.values():
            s.set_color("#333366")

    def _draw_delays(self, robots):
        ax = self.axes["delay"]
        ax.clear()
        ax.set_facecolor("#12122a")
        ax.set_title("Robot Wait Times", color="white", fontsize=9,
                     family="monospace")
        ids    = [f"R{r.id}" for r in robots.values()]
        waits  = [r.total_wait_time for r in robots.values()]
        colors = [self._PRIORITY_CLR[r.priority] for r in robots.values()]
        ax.bar(ids, waits, color=colors, edgecolor="white", linewidth=0.4)
        ax.tick_params(colors="#666688", labelsize=6)
        ax.set_ylabel("ticks", color="#666688", fontsize=7)
        for s in ax.spines.values():
            s.set_color("#333366")

    def _draw_stats(self, tick, robots, ctrl):
        ax = self.axes["stats"]
        ax.clear()
        ax.set_facecolor("#12122a")
        ax.axis("off")

        done   = sum(1 for r in robots.values() if r.state == RobotState.DONE)
        active = sum(1 for r in robots.values()
                     if r.state in (RobotState.MOVING, RobotState.WAITING,
                                    RobotState.PLANNING))
        avg_w  = sum(r.total_wait_time for r in robots.values()) / max(1, len(robots))
        dl     = len(ctrl.detector.events)
        nm     = len(ctrl.near_miss_log)
        es     = len(ctrl.collision_log)

        txt = (
            f"{'━' * 28}\n"
            f"  Tick         {tick:>6}\n"
            f"  Completed  {done:>3}/{len(robots)}\n"
            f"  Active     {active:>6}\n"
            f"  Avg Wait   {avg_w:>6.1f}\n"
            f"  Deadlocks  {dl:>6}\n"
            f"  Near-Miss  {nm:>6}\n"
            f"  E-Stops    {es:>6}\n"
            f"  Reserved   {len(ctrl.reservations):>6}\n"
            f"{'━' * 28}"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontfamily="monospace", fontsize=9, color="#00ffcc",
                va="top", linespacing=1.5)

    @staticmethod
    def _heat_color(score: float) -> str:
        """Map 0..1+ → green → yellow → red."""
        s = max(0.0, min(1.0, score))
        if s < 0.5:
            r, g = int(s * 2 * 255), 255
        else:
            r, g = 255, int((1.0 - (s - 0.5) * 2) * 255)
        return f"#{r:02x}{g:02x}00"


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class Simulation:
    """
    Main simulation engine — orchestrates tick-by-tick execution:

        1. Update congestion forecaster (heatmap refresh)
        2. Process robots by priority order (URGENT first)
        3. State machine: IDLE → PLANNING → MOVING ⇄ WAITING → DONE
        4. Deadlock detection via Tarjan's SCC on wait-graph
        5. Metrics collection and dashboard rendering
    """

    def __init__(self):
        self.graph      = LaneGraph()
        self.forecaster = CongestionForecaster(self.graph)
        self.detector   = DeadlockDetector()
        self.planner    = PathPlanner(self.graph, self.forecaster)
        self.controller = TrafficController(
            self.graph, self.planner, self.forecaster, self.detector)
        self.visualizer = Visualizer(self.graph)

        self.robots: Dict[int, Robot] = {}
        self.tick    = 0
        self.running = True

        self._init_robots()

    def _init_robots(self):
        """Spawn 10 robots with diverse priorities and cross-traffic routes."""
        configs = [
            # (id, priority,        start, goal)
            ( 0,  PriorityClass.URGENT,   0, 19),  # D1 → P1 full diagonal
            ( 1,  PriorityClass.NORMAL,   3, 16),  # D4 → H1 cross
            ( 2,  PriorityClass.URGENT,  16,  3),  # H1 → D4 counter-flow
            ( 3,  PriorityClass.NORMAL,   7, 12),  # C3 → S1
            ( 4,  PriorityClass.IDLE,    19,  0),  # P1 → D1 reverse diagonal
            ( 5,  PriorityClass.NORMAL,   4, 11),  # C1 → A3
            ( 6,  PriorityClass.URGENT,  11,  4),  # A3 → C1 counter-flow
            ( 7,  PriorityClass.NORMAL,   2, 17),  # D3 → H2
            ( 8,  PriorityClass.IDLE,    15,  1),  # S3 → D2
            ( 9,  PriorityClass.NORMAL,   8, 19),  # A1 → P1
        ]
        for rid, pri, start, goal in configs:
            r = Robot(id=rid, priority=pri, start=start, goal=goal)
            self.robots[rid] = r
            self.controller.register_node(r.current_node, rid)
            logging.info(f"Spawned R{rid} ({pri.name}): "
                         f"{self.graph.node_labels[start]} → "
                         f"{self.graph.node_labels[goal]}")

    # ── main tick ──

    def tick_step(self):
        """Execute one simulation tick."""
        self.tick += 1

        # 1. Congestion update
        self.forecaster.update(self.tick, set(self.controller.edge_occ.keys()))

        # 2. Process robots by priority (URGENT first, then NORMAL, then IDLE)
        priority_order = sorted(self.robots.keys(),
                                key=lambda rid: self.robots[rid].priority.value)

        for rid in priority_order:
            rob = self.robots[rid]
            if rob.state == RobotState.DONE:
                continue
            if rob.state == RobotState.IDLE:
                self._do_idle(rob)
            elif rob.state == RobotState.PLANNING:
                self._do_planning(rob)
            elif rob.state == RobotState.MOVING:
                self._do_moving(rob)
            elif rob.state == RobotState.WAITING:
                self._do_waiting(rob)

        # 3. Deadlock detection + resolution
        self._handle_deadlocks()

        # 4. Metrics
        done = sum(1 for r in self.robots.values() if r.state == RobotState.DONE)
        self.controller.throughput_history.append(done / max(1, self.tick))

        # 5. Termination check
        if all(r.state == RobotState.DONE for r in self.robots.values()):
            self.running = False
        if self.tick >= MAX_TICKS:
            self.running = False

    # ── state handlers ──

    def _do_idle(self, rob: Robot):
        """Transition IDLE → PLANNING, with priority-based staggering."""
        stagger = {PriorityClass.URGENT: 0, PriorityClass.NORMAL: 15,
                   PriorityClass.IDLE: 30}
        wait_ticks = stagger[rob.priority] + rob.id * 3
        if self.tick <= wait_ticks:
            return
            
        # Ensure spawn node is clear of other active robots
        for oid, other in self.robots.items():
            if oid != rob.id and other.state != RobotState.DONE:
                if other.current_node == rob.current_node and other.edge_progress == 0.0:
                    return

        rob.state      = RobotState.PLANNING
        rob.start_tick = self.tick
        logging.info(f"R{rob.id}: IDLE → PLANNING")

    def _do_planning(self, rob: Robot):
        """Compute path using A* + ghost prediction + congestion forecast."""
        banned = rob.banned_next_node
        
        # Calculate conflict multiplier from memory
        conflict_weight = sum(v for k, v in self.controller.conflict_memory.items() if rob.id in k)
        effective_replan = rob.replan_count + conflict_weight

        path = self.planner.plan(rob.current_node, rob.goal, banned_next_node=banned, replan_count=effective_replan, deadlock_edge_penalty=self.controller.deadlock_edge_penalty, personal_penalties=self.controller.personal_penalties[rob.id])

        if path is None:
            logging.warning(f"R{rob.id}: no path {rob.current_node}→{rob.goal}")
            rob.state     = RobotState.WAITING
            rob.wait_time = REPLAN_WAIT_TIMEOUT - 3   # retry soon
            return

        # Ghost prediction: check for future conflicts
        if not self.controller.ghost_predict_clear(rob, self.robots, path, 0):
            penalized = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
            alt = self.planner.plan(rob.current_node, rob.goal, penalized_edges=penalized, banned_next_node=banned, replan_count=effective_replan, deadlock_edge_penalty=self.controller.deadlock_edge_penalty, personal_penalties=self.controller.personal_penalties[rob.id])
            if alt and len(alt) > 1:
                path = alt
                logging.info(f"R{rob.id}: soft-rerouted (ghost penalty +50.0)")

        # Congestion forecast: proactive reroute if any hop > threshold
        worst_cong = self.forecaster.forecast_path(path, hops=GHOST_LOOKAHEAD)
        if worst_cong > CONGESTION_THRESHOLD:
            bad = {(path[i], path[i + 1]) for i in range(len(path) - 1)
                   if self.forecaster.get_score((path[i], path[i + 1]))
                   > CONGESTION_THRESHOLD}
            alt = self.planner.plan(rob.current_node, rob.goal, bad, banned_next_node=banned, replan_count=effective_replan, deadlock_edge_penalty=self.controller.deadlock_edge_penalty, personal_penalties=self.controller.personal_penalties[rob.id])
            if alt:
                path = alt
                logging.info(f"R{rob.id}: congestion reroute "
                             f"(worst={worst_cong:.2f})")

        rob.banned_next_node = None
        rob.path          = path
        rob.path_index    = 0
        rob.edge_progress = 0.0
        
        nxt = path[1] if len(path) > 1 else -1
        rob.next_node = nxt
        if nxt >= 0:
            edge = (rob.current_node, nxt)
            self.controller.register_edge(edge, rob.id)
            self.controller.register_target(nxt, rob.id)
            
        rob.state         = RobotState.MOVING

        logging.info(f"R{rob.id}: PLANNING → MOVING  path="
                     f"{[self.graph.node_labels[n] for n in path]}")

    def _do_moving(self, rob: Robot):
        """Execute one tick of movement along the planned path."""
        rob.travel_time += 1

        # Arrived at goal?
        if rob.current_node == rob.goal or rob.path_index >= len(rob.path) - 1:
            self._arrive(rob)
            return

        nxt  = rob.path[rob.path_index + 1]
        edge = (rob.current_node, nxt)

        # Validate edge exists
        if not self.graph.graph.has_edge(*edge):
            logging.error(f"R{rob.id}: invalid edge {edge}, replanning")
            rob.state = RobotState.PLANNING
            rob.replan_count += 1
            return

        edata     = self.graph.get_edge_data(*edge)
        lane_type = edata["lane_type"]
        lane_spd  = edata["speed_limit"]

        # Human zone → horn signal
        rob.horn_active = (lane_type == LaneType.HUMAN_ZONE)

        # ── gate checks (must pass ALL to proceed) ──

        # Gate 1: intersection reservation
        if nxt in LaneGraph.INTERSECTION_NODES:
            if not self.controller.request_intersection(rob, nxt, self.tick):
                self._enter_wait(rob, ctrl_node=nxt)
                return

        # Gate 2: destination node free
        if not self.controller.is_node_free(nxt, rob.id):
            occ = self.controller.node_occ.get(nxt)
            self._enter_wait(rob, blocker=occ)
            return

        # Gate 2.5: target node reserved by another incoming robot
        occ2 = self.controller.target_occ.get(nxt)
        if occ2 is not None and occ2 != rob.id:
            self._enter_wait(rob, blocker=occ2)
            return

        # Gate 3: edge free
        if not self.controller.is_edge_free(edge, rob.id):
            occ3 = self.controller.edge_occ.get(edge)
            self._enter_wait(rob, blocker=occ3)
            return

        # Gate 4: reverse direction for ALL bidirectional lanes
        rev = (nxt, rob.current_node)
        if self.graph.graph.has_edge(*rev):
            if not self.controller.is_edge_free(rev, rob.id):
                occ4 = self.controller.edge_occ.get(rev)
                self._enter_wait(rob, blocker=occ4)
                return

        # Gate 5: following distance → reduce speed
        if not self.controller.check_following_distance(rob, self.robots):
            lane_spd *= 0.3

        # Gate 6: emergency stop
        blocking_id = self.controller.check_emergency_stop(rob, self.robots, self.tick)
        if blocking_id is not None:
            rob.emergency_stopped = True
            rob.current_speed = 0.0
            self._enter_wait(rob, blocker=blocking_id)
            return
        rob.emergency_stopped = False

        # ── velocity update (trapezoidal profile) ──
        rob.target_speed = min(lane_spd, rob.max_speed)
        self._update_velocity(rob)

        # ── progress ──
        edge_len = edata["length"]
        delta    = rob.current_speed / edge_len if edge_len > 0 else 1.0

        # Registration is now securely handled upstream before transition
        rob.edge_progress += delta

        if rob.edge_progress > 0.0:
            if self.controller.node_occ.get(rob.current_node) == rob.id:
                self.controller.unregister_node(rob.current_node, rob.id)

        # ── arrival at next node ──
        if rob.edge_progress >= 1.0:
            self.controller.unregister_edge(edge, rob.id)
            self.controller.unregister_target(nxt, rob.id)

            if rob.current_node in LaneGraph.INTERSECTION_NODES:
                self.controller.release_intersection(
                    rob.current_node, rob.id, self.tick)

            rob.current_node = nxt
            rob.path_index  += 1
            rob.edge_progress = 0.0
            rob.path_history.append(nxt)
            self.controller.register_node(nxt, rob.id)

            if rob.path_index < len(rob.path) - 1:
                rob.next_node = rob.path[rob.path_index + 1]
            else:
                rob.next_node = -1

            if rob.current_node == rob.goal:
                self._arrive(rob)

    def _do_waiting(self, rob: Robot):
        """Increment wait counters; attempt to resume or trigger replan."""
        rob.wait_time       += 1
        rob.total_wait_time += 1
        rob.travel_time     += 1

        # Attempt to proceed
        can_go = True
        if rob.path_index < len(rob.path) - 1:
            nxt  = rob.path[rob.path_index + 1]
            edge = (rob.current_node, nxt)

            if nxt in LaneGraph.INTERSECTION_NODES:
                if not self.controller.request_intersection(rob, nxt, self.tick):
                    can_go = False
            if can_go and not self.controller.is_node_free(nxt, rob.id):
                can_go = False
            if can_go and not self.controller.is_edge_free(edge, rob.id):
                can_go = False
            if can_go:
                rev = (nxt, rob.current_node)
                if self.graph.graph.has_edge(*rev):
                    if not self.controller.is_edge_free(rev, rob.id):
                        can_go = False

        if can_go:
            if rob.path_index < len(rob.path) - 1:
                nxt  = rob.path[rob.path_index + 1]
                edge = (rob.current_node, nxt)
                self.controller.register_edge(edge, rob.id)
                self.controller.register_target(nxt, rob.id)
                rob.next_node = nxt

            rob.state       = RobotState.MOVING
            rob.wait_time   = 0
            rob.waiting_for = None
            logging.debug(f"R{rob.id}: WAITING → MOVING")
        elif rob.wait_time >= REPLAN_WAIT_TIMEOUT and rob.edge_progress == 0.0:
            rob.replan_history = [t for t in rob.replan_history if self.tick - t <= 50]
            if len(rob.replan_history) < 3:
                logging.info(f"R{rob.id}: replan (waited {rob.wait_time} ticks)")
                rob.state        = RobotState.PLANNING
                rob.replan_count += 1
                rob.wait_time    = 0
                rob.waiting_for  = None
                rob.replan_history.append(self.tick)

    # ── helpers ──

    def _enter_wait(self, rob: Robot, blocker: Optional[int] = None,
                    ctrl_node: Optional[int] = None):
        """Transition a robot to WAITING state."""
        rob.state     = RobotState.WAITING
        rob.wait_time = 0
        if blocker is not None:
            rob.waiting_for = blocker
        elif ctrl_node is not None and ctrl_node in self.controller.reservations:
            rob.waiting_for = self.controller.reservations[ctrl_node]
        else:
            rob.waiting_for = None

    def _arrive(self, rob: Robot):
        """Handle robot reaching its goal."""
        rob.state         = RobotState.DONE
        rob.end_tick      = self.tick
        rob.current_speed = 0.0
        rob.horn_active   = False
        self.controller.completed_count += 1

        if rob.current_node in LaneGraph.INTERSECTION_NODES:
            self.controller.release_intersection(
                rob.current_node, rob.id, self.tick)

        # Unregister so DONE robots don't block traffic
        self.controller.unregister_node(rob.current_node, rob.id)

        logging.info(
            f"R{rob.id}: ✓ ARRIVED {self.graph.node_labels[rob.goal]}  "
            f"travel={rob.travel_time} wait={rob.total_wait_time} "
            f"replans={rob.replan_count}")

    def _update_velocity(self, rob: Robot):
        """
        Trapezoidal velocity profile:
          Phase 1 (Accelerate): speed += accel_rate  until target_speed
          Phase 2 (Cruise):     speed = target_speed
          Phase 3 (Decelerate): speed -= decel_rate  when near goal / obstacle
        """
        remaining = len(rob.path) - rob.path_index - 1
        if remaining <= 2:
            # Decelerate near end-of-path
            rob.current_speed = max(MIN_CRAWL_SPEED,
                                    rob.current_speed - rob.decel_rate)
        elif rob.current_speed < rob.target_speed:
            rob.current_speed = min(rob.target_speed,
                                    rob.current_speed + rob.accel_rate)
        else:
            rob.current_speed = rob.target_speed

    def _handle_deadlocks(self):
        """Run Tarjan's SCC, resolve each deadlock cycle."""
        wg     = self.controller.build_wait_graph(self.robots)
        cycles = self.detector.find_deadlocks(wg)

        for cycle in cycles:
            self.controller.deadlock_count += 1
            
            # Global Cycle detection
            cycle_key = tuple(sorted(cycle))
            self.detector.cycle_history[cycle_key].append(self.tick)
            history = [t for t in self.detector.cycle_history[cycle_key] if self.tick - t <= 50]
            self.detector.cycle_history[cycle_key] = history

            # Apply +100 cumulative penalty to ALL edges in cycle instantly
            for rid in cycle:
                c_rob = self.robots[rid]
                if c_rob.path_index < len(c_rob.path) - 1:
                    c_edge = (c_rob.current_node, c_rob.path[c_rob.path_index + 1])
                    self.controller.deadlock_edge_penalty[c_edge] += 100.0
            logging.warning(f"⚠ CYCLE PENALTY: All edges in cycle {cycle_key} heavily penalized (+100 cumulative)!")

            sid = self.detector.resolve(cycle, self.robots, self.tick)
            if sid is None:
                continue

            rob = self.robots[sid]

            # Banned next node logic -> Personal Penalty Diversification
            if rob.path_index < len(rob.path) - 1:
                sac_edge = (rob.current_node, rob.path[rob.path_index + 1])
                rob.banned_next_node = rob.path[rob.path_index + 1]
                self.controller.personal_penalties[sid][sac_edge] += 200.0
                logging.info(f"R{sid}: deadlock sacrifice. Personal penalty +200 on {sac_edge}")

            if rob.next_node >= 0:
                edge = (rob.current_node, rob.next_node)
                self.controller.unregister_edge(edge, rob.id)
                self.controller.unregister_target(rob.next_node, rob.id)
                if rob.next_node in LaneGraph.INTERSECTION_NODES:
                    self.controller.release_intersection(rob.next_node, rob.id, self.tick)

            rob.state        = RobotState.PLANNING
            rob.replan_count += 1
            rob.wait_time    = 0
            rob.waiting_for  = None
            rob.edge_progress = 0.0

    # ── summary ──

    def print_summary(self):
        """Print comprehensive terminal summary table."""
        print("\n" + "═" * 72)
        print("║   LANE-AWARE MULTI-ROBOT TRAFFIC CONTROL — SIMULATION SUMMARY    ║")
        print("═" * 72)
        print(f"  Total Simulation Time : {self.tick} ticks")
        print(f"  Deadlocks Detected    : {self.controller.deadlock_count}")
        print(f"  Deadlocks Resolved    : {len(self.detector.events)}")
        print(f"  Near-Misses           : {len(self.controller.near_miss_log)}")
        print(f"  Emergency Stops       : {len(self.controller.collision_log)}")
        print("─" * 72)
        print(f"  {'Robot':>5} │ {'Priority':>8} │ {'Travel':>6} │ {'Wait':>6} │ "
              f"{'Replans':>7} │ {'Status':>8}")
        print("─" * 72)
        for r in sorted(self.robots.values(), key=lambda x: x.id):
            print(f"  {'R' + str(r.id):>5} │ {r.priority.name:>8} │ "
                  f"{r.travel_time:>6} │ {r.total_wait_time:>6} │ "
                  f"{r.replan_count:>7} │ {r.state.name:>8}")
        print("─" * 72)

        total_w = sum(r.total_wait_time for r in self.robots.values())
        done_c  = sum(1 for r in self.robots.values() if r.state == RobotState.DONE)
        thru    = done_c / max(1, self.tick)
        avg_d   = total_w / max(1, len(self.robots))

        total_edges = self.graph.graph.number_of_edges()
        util_sum = 0.0
        for h in self.forecaster.occupancy_history.values():
            if h:
                util_sum += sum(h) / len(h)
        utilization = util_sum / max(1, total_edges) * 100

        print(f"  Throughput       : {thru:.4f} robots/tick")
        print(f"  Average Delay    : {avg_d:.1f} ticks")
        print(f"  Completed        : {done_c}/{len(self.robots)}")
        print(f"  Lane Utilization : {utilization:.1f}%")
        print("═" * 72)

        if self.detector.events:
            print("\n  Deadlock Resolution Log:")
            for ev in self.detector.events:
                print(f"    Tick {ev.tick:>3}: cycle={ev.robots_involved} "
                      f"→ sacrifice R{ev.sacrifice_robot}")
            print()

        if self.controller.collision_log:
            print("  Emergency Stop Log:")
            for ev in self.controller.collision_log[:10]:
                print(f"    Tick {ev.tick:>3}: {ev.event_type} at "
                      f"({ev.location[0]:.1f},{ev.location[1]:.1f}) "
                      f"robots={ev.robots}")
            print()
        print("═" * 72)

    # ── entry point ──

    def run(self):
        """Launch simulation with live animation."""
        logging.info("Simulation starting…")

        def _animate(frame):
            if self.running:
                self.tick_step()
            self.visualizer.draw(
                self.tick, self.robots, self.controller,
                self.forecaster, self.running)
            return []

        _anim = animation.FuncAnimation(
            self.visualizer.fig, _animate,
            frames=MAX_TICKS, interval=ANIMATION_INTERVAL_MS,
            blit=False, repeat=False, cache_frame_data=False)

        plt.show()

        # After window close → print summary
        self.print_summary()


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     Lane-Aware Multi-Robot Traffic Control System               ║
    ║     ─────────────────────────────────────────────               ║
    ║     Differentiating Features:                                   ║
    ║       ◆ Tarjan's SCC Deadlock Detection & Resolution            ║
    ║       ◆ Auction-Based Intersection Reservation                  ║
    ║       ◆ Probabilistic Congestion Forecasting (Exp. Decay)       ║
    ║       ◆ Elastic Trapezoidal Speed Profiles                      ║
    ║       ◆ Ghost Robot Future-Position Prediction                  ║
    ║       ◆ Priority-Class Traffic Routing                          ║
    ║       ◆ Real-Time Dual-Pane Animated Dashboard                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    sim = Simulation()
    sim.run()
