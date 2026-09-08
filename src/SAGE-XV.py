#!/usr/bin/env python3
"""SAGE XV: circuit-and-topology-aware embedding with canonical LightSABRE routing."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import networkx as nx
import numpy as np

try:
    from qiskit import QuantumCircuit, qasm2, transpile
    from qiskit.circuit import ClassicalRegister, QuantumRegister
    from qiskit.transpiler import CouplingMap
except ModuleNotFoundError:
    QuantumCircuit = None
    ClassicalRegister = None
    QuantumRegister = None
    CouplingMap = None
    transpile = None

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from routing_input import canonical_interactions, canonical_operations, canonical_qasm, canonical_qubit_names, interaction_fingerprint
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from routing_input import canonical_interactions, canonical_operations, canonical_qasm, canonical_qubit_names, interaction_fingerprint

EXPERTS = ("arithmetic", "dense_global", "general")


def resolve_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path




@dataclass(frozen=True)
class RouteMetrics:
    # Routing result metrics used for candidate comparison.
    swap_count: int
    depth: int

    @property
    def cost(self) -> float:
        return 3.0 * self.swap_count + self.depth



def load_topology(path: Path) -> tuple[int, list[tuple[int, int]]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    edges = [tuple(map(int, edge)) for edge in data["edges"]]
    return int(data["num_qubits"]), edges


def make_grid_topology(min_qubits: int) -> tuple[int, list[tuple[int, int]]]:
    """Create one square grid large enough for every circuit in the dataset."""
    side = math.ceil(math.sqrt(min_qubits))
    edges = []
    for row in range(side):
        for column in range(side):
            qubit = row * side + column
            if column + 1 < side:
                edges.append((qubit, qubit + 1))
            if row + 1 < side:
                edges.append((qubit, qubit + side))
    return side * side, edges




@dataclass(frozen=True)
class TopologyClassification:
    profile: str
    features: dict[str, float | int]


def classify_topology(num_qubits: int, edges: list[tuple[int, int]]) -> TopologyClassification:
    """Classify a coupling graph from structure, never from a topology filename."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    graph.add_edges_from(tuple(sorted(edge)) for edge in edges)
    components = nx.number_connected_components(graph)
    degrees = [degree for _, degree in graph.degree()]
    edge_count = graph.number_of_edges()
    mean_degree = statistics.fmean(degrees) if degrees else 0.0
    max_degree = max(degrees, default=0)
    degree_cv = statistics.pstdev(degrees) / max(1e-9, mean_degree) if len(degrees) > 1 else 0.0
    cycle_rank = max(0, edge_count - num_qubits + components)
    articulation_count = sum(1 for _ in nx.articulation_points(graph))
    bridge_count = sum(1 for _ in nx.bridges(graph))
    clustering = nx.average_clustering(graph) if num_qubits else 0.0
    degree_one = sum(degree == 1 for degree in degrees)
    degree_two = sum(degree == 2 for degree in degrees)
    degree_three = sum(degree == 3 for degree in degrees)

    connected = components == 1
    if connected and num_qubits >= 3 and edge_count == num_qubits and degree_two == num_qubits:
        profile = "ring_like"
    elif connected and edge_count == max(0, num_qubits - 1) and (
        num_qubits <= 2 or (degree_one == 2 and degree_two == num_qubits - 2)
    ):
        profile = "path_like"
    elif (
        connected
        and max_degree >= 4
        and clustering <= 0.05
        and bridge_count <= max(1, edge_count // 100)
        and articulation_count <= max(1, num_qubits // 100)
    ):
        profile = "grid_like"
    elif (
        connected
        and max_degree <= 3
        and cycle_rank > 0
        and clustering <= 0.10
        and degree_three >= max(1, num_qubits // 12)
    ):
        profile = "heavy_hex_like"
    else:
        profile = "irregular_sparse"

    return TopologyClassification(profile, {
        "physical_qubits": num_qubits,
        "edges": edge_count,
        "components": components,
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "degree_cv": degree_cv,
        "clustering": clustering,
        "cycle_rank": cycle_rank,
        "articulation_points": articulation_count,
        "articulation_ratio": articulation_count / max(1, num_qubits),
        "bridges": bridge_count,
        "bridge_ratio": bridge_count / max(1, edge_count),
    })

def load_two_qubit_gates(qasm_path: Path) -> tuple[int, list[tuple[int, int]]]:
    """Load the shared canonical two-qubit interaction sequence."""
    return canonical_interactions(qasm_path)

@lru_cache(maxsize=32)
def _cached_all_pairs_shortest_paths(
    num_qubits: int,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    dist = np.full((num_qubits, num_qubits), 10**6, dtype=np.int32)
    for i in range(num_qubits):
        dist[i, i] = 0
    for a, b in edges:
        dist[a, b] = 1
        dist[b, a] = 1
    for k in range(num_qubits):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    dist.flags.writeable = False
    return dist


def all_pairs_shortest_paths(num_qubits: int, edges: list[tuple[int, int]]) -> np.ndarray:
    normalized = tuple(sorted(tuple(sorted(edge)) for edge in edges))
    return _cached_all_pairs_shortest_paths(num_qubits, normalized)


def add_two_qubit_depth(layers: list[int], a: int, b: int) -> None:
    layer = max(layers[a], layers[b]) + 1
    layers[a] = layer
    layers[b] = layer


@dataclass
class FrontLayerScores:
    nodes: list[tuple[int, int]] = field(default_factory=list)
    qubits: list[tuple[int, int] | None] = field(default_factory=list)

    @classmethod
    def from_ctx(cls, ctx) -> "FrontLayerScores":
        out = cls(qubits=[None] * ctx.topology().num_qubits())
        for a, b in ctx.front_layer().physical_pairs():
            index = len(out.nodes)
            out.nodes.append((a, b))
            out.qubits[a] = (index, b)
            out.qubits[b] = (index, a)
        return out

    def __len__(self) -> int:
        return len(self.nodes)

    def is_active(self, qubit: int) -> bool:
        return self.qubits[qubit] is not None

    def iter_active(self):
        for a, b in self.nodes:
            yield a
            yield b

    def total_score(self, topology) -> float:
        return sum(float(topology.distance(a, b)) for a, b in self.nodes)
    def score_delta(self, swap: tuple[int, int], topology) -> float:
        a, b = swap
        delta = 0.0
        if self.qubits[a] is not None:
            _, c = self.qubits[a]
            delta += float(topology.distance(b, c) - topology.distance(a, c))
        if self.qubits[b] is not None:
            _, c = self.qubits[b]
            delta += float(topology.distance(a, c) - topology.distance(b, c))
        return delta


@dataclass
class ExtendedSetScores:
    qubits: list[list[int]]
    length: int = 0

    @classmethod
    def new(cls, num_qubits: int) -> "ExtendedSetScores":
        return cls([[] for _ in range(num_qubits)])

    def push(self, a: int, b: int) -> None:
        self.qubits[a].append(b)
        self.qubits[b].append(a)
        self.length += 1

    def __len__(self) -> int:
        return self.length

    def total_score(self, topology) -> float:
        total = 0.0
        for a, others in enumerate(self.qubits):
            for b in others:
                total += float(topology.distance(a, b))
        return total * 0.5

    def score_delta(self, swap: tuple[int, int], topology) -> float:
        a, b = swap
        total = 0.0
        for other in self.qubits[a]:
            if other != b:
                total += float(topology.distance(b, other) - topology.distance(a, other))
        for other in self.qubits[b]:
            if other != a:
                total += float(topology.distance(a, other) - topology.distance(b, other))
        return total


def build_extended_set(ctx, max_size: int) -> ExtendedSetScores:
    out = ExtendedSetScores.new(ctx.topology().num_qubits())
    if max_size == 0:
        return out

    precomputed = ctx.precomputed_extended_set_logical_pairs()
    if precomputed:
        for a, b in precomputed[:max_size]:
            out.push(ctx.layout().physical_of_logical(a), ctx.layout().physical_of_logical(b))
        return out

    # Only copy predecessor counts for nodes reached by the bounded lookahead.
    pending: dict[int, int] = {}
    to_visit = list(ctx.front_layer().node_ids())
    i = 0
    while i < len(to_visit) and len(out) < max_size:
        node_id = to_visit[i]
        for successor in ctx.circuit().node(node_id).successors():
            remaining = pending.get(successor, ctx.remaining_counts[successor]) - 1
            pending[successor] = remaining
            if remaining == 0:
                a, b = ctx.circuit().node(successor).two_qubit_pair()
                out.push(ctx.layout().physical_of_logical(a), ctx.layout().physical_of_logical(b))
                to_visit.append(successor)
        i += 1

    return out


def enumerate_candidate_swaps(topology, front_layer: FrontLayerScores) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for phys in front_layer.iter_active():
        for neighbor in topology.neighbors(phys):
            if neighbor > phys or not front_layer.is_active(neighbor):
                out.append((phys, neighbor))
    return out


def ensure_connected_subset(topology, subset: list[int]) -> None:
    if not subset:
        return
    allowed = set(subset)
    seen = {subset[0]}
    queue = deque([subset[0]])
    while queue:
        node = queue.popleft()
        for nxt in topology.neighbors(node):
            if nxt in allowed and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    if len(seen) != len(allowed):
        raise ValueError("selected layout subset is not connected")


def choose_dense_layout_subset(topology, logical_size: int, target_component: list[int]) -> list[int]:
    if logical_size > len(target_component):
        raise ValueError("logical component exceeds target component size")
    if logical_size == len(target_component):
        return list(target_component)

    component = set(target_component)
    degree = {q: sum(1 for n in topology.neighbors(q) if n in component) for q in target_component}
    chosen = [max(target_component, key=lambda q: (degree[q], -q))]
    chosen_set = set(chosen)

    while len(chosen) < logical_size:
        candidates = [q for p in chosen for q in topology.neighbors(p) if q in component and q not in chosen_set]
        if not candidates:
            candidates = [q for q in target_component if q not in chosen_set]
        nxt = max(
            candidates,
            key=lambda q: (sum(1 for n in topology.neighbors(q) if n in chosen_set), degree[q], -q),
        )
        chosen.append(nxt)
        chosen_set.add(nxt)

    ensure_connected_subset(topology, chosen)
    return chosen


def assign_components_to_target(
    logical_components: list[list[int]], target_components: list[list[int]]
) -> list[tuple[list[int], int]]:
    logical_sorted = sorted(logical_components, key=len, reverse=True)
    target_sorted = sorted(enumerate(target_components), key=lambda item: len(item[1]), reverse=True)
    free_capacity = {idx: len(component) for idx, component in target_sorted}
    assignments: list[tuple[list[int], int]] = []

    for logical in logical_sorted:
        target_idx = next((idx for idx, _ in target_sorted if free_capacity[idx] >= len(logical)), None)
        if target_idx is None:
            raise ValueError(f"logical component of size {len(logical)} cannot fit any target")
        free_capacity[target_idx] -= len(logical)
        assignments.append((logical, target_idx))

    return assignments


def choose_disjoint_aware_layout(ctx) -> list[int]:
    circuit = ctx.circuit()
    topology = ctx.topology()
    num_logical = circuit.num_logical_qubits()
    if not circuit.used_logical_qubits():
        return list(range(num_logical))

    logical_components = circuit.logical_interaction_components()
    target_components = topology.connected_components()
    if not target_components:
        raise ValueError("topology has no connected components")

    assignments = assign_components_to_target(logical_components, target_components)
    mapping = [-1] * num_logical
    used_physical = [False] * topology.num_qubits()

    for logical_component, target_component_idx in assignments:
        if not logical_component:
            continue
        target_component = target_components[target_component_idx]
        local = choose_dense_layout_subset(topology, len(logical_component), target_component)
        for logical, physical in zip(logical_component, local):
            mapping[logical] = physical
            used_physical[physical] = True

    free_physical = (q for q, used in enumerate(used_physical) if not used)
    for logical, physical in enumerate(mapping):
        if physical == -1:
            mapping[logical] = next(free_physical)

    return mapping


@dataclass
class CandidatePolicy:
    basic_weight: float = 1.0
    lookahead_weight: float = 0.5
    lookahead_size: int = 20
    set_scaling: str = "size"
    use_decay: bool = True
    decay_increment: float = 0.001
    decay_reset: int = 5
    best_epsilon: float = 1e-10
    decay_state: list[float] = field(default_factory=list)
    heavy_hex_center_bias: float = 0.0
    topology_profile: str = "unknown"

    def refresh_decay_state(self, ctx) -> None:
        if not self.use_decay:
            return
        num_qubits = ctx.topology().num_qubits()
        if len(self.decay_state) != num_qubits:
            self.decay_state = [1.0] * num_qubits
        if ctx.swaps_since_progress() == 0:
            self.decay_state = [1.0] * num_qubits
            return
        last_swap = ctx.last_applied_swap()
        if last_swap is None:
            return
        a, b = last_swap
        reset = max(self.decay_reset, 1)
        if ctx.swaps_since_progress() % reset == 0:
            self.decay_state = [1.0] * num_qubits
        else:
            self.decay_state[a] += self.decay_increment
            self.decay_state[b] += self.decay_increment

    def choose_best_initial_layout(self, ctx, rng: random.Random | None = None) -> list[int]:
        self.decay_state.clear()
        return choose_disjoint_aware_layout(ctx)

    def rank_swaps(self, ctx, refresh_decay: bool = True) -> list[tuple[tuple[int, int], float]]:
        """Return LightSABRE candidates in deterministic score order."""
        self.refresh_decay_state(ctx)
        front_layer = FrontLayerScores.from_ctx(ctx)
        candidates = enumerate_candidate_swaps(ctx.topology(), front_layer)
        if not candidates:
            return []

        extended_set = build_extended_set(ctx, self.lookahead_size)

        def scale(weight: float, size: int) -> float:
            return weight if self.set_scaling == "constant" else (0.0 if size == 0 else weight / float(size))

        basic_weight = scale(self.basic_weight, len(front_layer))
        lookahead_weight = scale(self.lookahead_weight, len(extended_set))
        absolute_score = basic_weight * front_layer.total_score(ctx.topology())
        swap_scores = [
            (swap, basic_weight * front_layer.score_delta(swap, ctx.topology()))
            for swap in candidates
        ]
        if len(extended_set) and self.lookahead_weight != 0.0:
            absolute_score += lookahead_weight * extended_set.total_score(ctx.topology())
            swap_scores = [
                (swap, score + lookahead_weight * extended_set.score_delta(swap, ctx.topology()))
                for swap, score in swap_scores
            ]
        if self.use_decay:
            swap_scores = [
                (swap, (absolute_score + score) * max(self.decay_state[swap[0]], self.decay_state[swap[1]]))
                for swap, score in swap_scores
            ]
        else:
            swap_scores = [(swap, absolute_score + score) for swap, score in swap_scores]
        if self.topology_profile == "heavy_hex_like" and self.heavy_hex_center_bias > 0.0:
            topology = ctx.topology()
            centrality = [sum(topology.distance(node, other) for other in range(topology.num_qubits())) for node in range(topology.num_qubits())]
            scale = max(1.0, abs(float(absolute_score)))
            swap_scores = [
                (swap, score + self.heavy_hex_center_bias * scale * max(centrality[swap[0]], centrality[swap[1]]) / max(1, topology.num_qubits()))
                for swap, score in swap_scores
            ]
        return sorted(swap_scores, key=lambda item: item[1])

    def choose_best_swap(self, ctx, rng: random.Random | None = None) -> tuple[int, int] | None:
        ranked = self.rank_swaps(ctx)
        return ranked[0][0] if ranked else None

@dataclass
class RoutingNode:
    pair: tuple[int, int]
    successor_ids: list[int] = field(default_factory=list)

    def successors(self) -> list[int]:
        return self.successor_ids

    def two_qubit_pair(self) -> tuple[int, int]:
        return self.pair


class RoutingCircuit:
    def __init__(self, num_logical: int, gates: list[tuple[int, int]]):
        self.num_logical = num_logical
        self.nodes = [RoutingNode(pair) for pair in gates]
        self.predecessor_counts = [0] * len(self.nodes)
        last_on_qubit: list[int | None] = [None] * num_logical
        for node_id, (a, b) in enumerate(gates):
            preds = {last_on_qubit[a], last_on_qubit[b]} - {None}
            self.predecessor_counts[node_id] = len(preds)
            for pred in preds:
                self.nodes[pred].successor_ids.append(node_id)
            last_on_qubit[a] = node_id
            last_on_qubit[b] = node_id

    def node(self, node_id: int) -> RoutingNode:
        return self.nodes[node_id]

    def num_logical_qubits(self) -> int:
        return self.num_logical

    def used_logical_qubits(self) -> list[int]:
        return sorted({q for node in self.nodes for q in node.pair})

    def logical_interaction_components(self) -> list[list[int]]:
        used = self.used_logical_qubits()
        adjacency = {q: set() for q in used}
        for a, b in (node.pair for node in self.nodes):
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        components: list[list[int]] = []
        seen: set[int] = set()
        for start in used:
            if start in seen:
                continue
            queue = deque([start])
            seen.add(start)
            component = []
            while queue:
                q = queue.popleft()
                component.append(q)
                for nxt in adjacency.get(q, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            components.append(sorted(component))
        return components


class RoutingTopology:
    def __init__(self, num_qubits: int, edges: list[tuple[int, int]]):
        self._num_qubits = num_qubits
        self._neighbors = [[] for _ in range(num_qubits)]
        for a, b in edges:
            self._neighbors[a].append(b)
            self._neighbors[b].append(a)
        self._dist = all_pairs_shortest_paths(num_qubits, edges)

    def num_qubits(self) -> int:
        return self._num_qubits

    def neighbors(self, qubit: int) -> list[int]:
        return self._neighbors[qubit]

    def distance(self, a: int, b: int) -> int:
        return int(self._dist[a, b])

    def connected_components(self) -> list[list[int]]:
        components: list[list[int]] = []
        seen: set[int] = set()
        for start in range(self._num_qubits):
            if start in seen:
                continue
            queue = deque([start])
            seen.add(start)
            component = []
            while queue:
                q = queue.popleft()
                component.append(q)
                for nxt in self._neighbors[q]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            components.append(component)
        return components


class RoutingLayout:
    def __init__(self, logical_qubits: int, physical_qubits: int, logical_to_physical: list[int]):
        self.logical_to_physical = list(logical_to_physical)
        self.physical_to_logical = [-1] * physical_qubits
        for logical, physical in enumerate(self.logical_to_physical):
            self.physical_to_logical[physical] = logical

    def physical_of_logical(self, logical: int) -> int:
        return self.logical_to_physical[logical]

    def apply_swap(self, a: int, b: int) -> None:
        log_a = self.physical_to_logical[a]
        log_b = self.physical_to_logical[b]
        self.physical_to_logical[a], self.physical_to_logical[b] = log_b, log_a
        if log_a != -1:
            self.logical_to_physical[log_a] = b
        if log_b != -1:
            self.logical_to_physical[log_b] = a


class RoutingRemaining:
    def __init__(self, counts: list[int]):
        self.counts = counts

    def remaining_predecessor_counts(self) -> list[int]:
        return self.counts


class RoutingFrontLayer:
    def __init__(self, ctx: "RoutingContext"):
        self.ctx = ctx

    def node_ids(self) -> list[int]:
        return sorted(self.ctx.front_ids)

    def physical_pairs(self) -> list[tuple[int, int]]:
        out = []
        for node_id in self.node_ids():
            a, b = self.ctx.circuit_obj.node(node_id).two_qubit_pair()
            out.append((self.ctx.layout_obj.physical_of_logical(a), self.ctx.layout_obj.physical_of_logical(b)))
        return out


class RoutingContext:
    def __init__(self, circuit: RoutingCircuit, topology: RoutingTopology, layout: RoutingLayout | None = None):
        self.circuit_obj = circuit
        self.topology_obj = topology
        self.layout_obj = layout
        self.remaining_counts = list(circuit.predecessor_counts)
        self.done = [False] * len(circuit.nodes)
        # Dict preserves deterministic gate order while supporting O(1) updates.
        self.front_ids = dict.fromkeys(
            node_id for node_id, count in enumerate(self.remaining_counts) if count == 0
        )
        self._swaps_since_progress = 0
        self._last_swap: tuple[int, int] | None = None

    def circuit(self) -> RoutingCircuit:
        return self.circuit_obj

    def topology(self) -> RoutingTopology:
        return self.topology_obj

    def layout(self) -> RoutingLayout:
        if self.layout_obj is None:
            raise RuntimeError("layout is not initialized")
        return self.layout_obj

    def remaining(self) -> RoutingRemaining:
        return RoutingRemaining(self.remaining_counts)

    def front_layer(self) -> RoutingFrontLayer:
        return RoutingFrontLayer(self)

    def precomputed_extended_set_logical_pairs(self) -> list[tuple[int, int]]:
        return []

    def swaps_since_progress(self) -> int:
        return self._swaps_since_progress

    def last_applied_swap(self) -> tuple[int, int] | None:
        return self._last_swap


def execute_routable_gates(ctx: RoutingContext, layers: list[int]) -> list[int]:
    """Execute adjacent front-layer gates and return their node IDs."""
    executed = []
    layout = ctx.layout_obj
    circuit = ctx.circuit_obj
    topology = ctx.topology_obj
    for node_id in sorted(ctx.front_ids):
        logical_a, logical_b = circuit.nodes[node_id].pair
        phys_a = layout.logical_to_physical[logical_a]
        phys_b = layout.logical_to_physical[logical_b]
        if topology.distance(phys_a, phys_b) != 1:
            continue
        add_two_qubit_depth(layers, phys_a, phys_b)
        ctx.done[node_id] = True
        ctx.front_ids.pop(node_id, None)
        executed.append(node_id)
        for successor in circuit.nodes[node_id].successor_ids:
            ctx.remaining_counts[successor] -= 1
            if ctx.remaining_counts[successor] == 0:
                ctx.front_ids[successor] = None
    if executed:
        ctx._swaps_since_progress = 0
        ctx._last_swap = None
    return executed


class HybridRoutingEnv:
    """Minimal LightSABRE environment for rank selection and model-free Beam."""

    def __init__(
        self,
        logical_qubits: int,
        gates: list[tuple[int, int]],
        physical_qubits: int,
        edges: list[tuple[int, int]],
        lookahead_size: int = 20,
        top_k: int = 5,
        objective: str = "swap",
        depth_weight: float = 0.05,
    ):
        del objective, depth_weight
        if logical_qubits > physical_qubits:
            raise ValueError("logical qubits cannot exceed physical qubits")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.logical_qubits = logical_qubits
        self.gates = gates
        self.physical_qubits = physical_qubits
        self.edges = sorted({tuple(sorted(edge)) for edge in edges})
        if not self.edges:
            raise ValueError("topology must contain at least one edge")
        self.top_k = top_k
        self.topology = RoutingTopology(physical_qubits, self.edges)
        self.circuit = RoutingCircuit(logical_qubits, gates)
        self.policy = CandidatePolicy(lookahead_size=lookahead_size)
        interaction_pairs = {tuple(sorted(pair)) for pair in gates}
        used_qubits = sorted({qubit for pair in gates for qubit in pair})
        interaction_degree = [0] * logical_qubits
        unique_degree = [0] * logical_qubits
        for a, b in gates:
            interaction_degree[a] += 1
            interaction_degree[b] += 1
        for a, b in interaction_pairs:
            unique_degree[a] += 1
            unique_degree[b] += 1
        used_degrees = [unique_degree[q] for q in used_qubits]
        self.simple_chain = bool(
            len(used_qubits) >= 2
            and len(self.circuit.logical_interaction_components()) == 1
            and len(gates) == len(interaction_pairs) == len(used_qubits) - 1
            and max(used_degrees, default=0) <= 2
        )
        max_load = max(interaction_degree, default=1) or 1
        self.logical_interaction_load = np.asarray(interaction_degree, dtype=np.float32) / max_load
        # Canonical Qiskit LightSABRE baseline is injected after environment construction.
        self.lightsabre_metrics = RouteMetrics(0, 0)
        self.lightsabre_swaps = self.lightsabre_metrics.swap_count
        self.max_swaps = max(200, 2 * self.lightsabre_swaps)
        self.rescue_after = 8
        self.ctx: RoutingContext | None = None
        self.layers: list[int] = []
        self.swap_count = 0
        self.executed_gates = 0
        self.no_progress_steps = 0
        self.rescue_count = 0
        self.route_events: list[tuple[str, object]] = []
        self.initial_mapping: list[int] = []
        self.candidates: list[tuple[int, int]] = []
        self.candidate_scores: list[float] = []

    def _ranked_candidates(self) -> tuple[list[tuple[int, int]], list[float]]:
        assert self.ctx is not None
        ranked = self.policy.rank_swaps(self.ctx)[:self.top_k]
        return [swap for swap, _ in ranked], [float(score) for _, score in ranked]

    def _advance(self) -> int:
        assert self.ctx is not None
        executed = 0
        while True:
            executed_ids = execute_routable_gates(self.ctx, self.layers)
            newly_executed = len(executed_ids)
            if newly_executed:
                self.route_events.extend(("gate", node_id) for node_id in executed_ids)
            executed += newly_executed
            if not newly_executed:
                break
        self.executed_gates += executed
        self.candidates, self.candidate_scores = self._ranked_candidates()
        return executed

    def reset(self, seed=None, options=None):
        self.policy = copy.copy(self.policy)
        self.policy.decay_state = []
        initial_mapping = (options or {}).get("initial_mapping")
        if initial_mapping is None:
            initial_mapping = self.policy.choose_best_initial_layout(
                RoutingContext(self.circuit, self.topology), random.Random(seed)
            )
        if len(initial_mapping) != self.logical_qubits or len(set(initial_mapping)) != len(initial_mapping):
            raise ValueError("initial_mapping must map every logical qubit to a unique physical qubit")
        if min(initial_mapping, default=0) < 0 or max(initial_mapping, default=-1) >= self.physical_qubits:
            raise ValueError("initial_mapping contains an invalid physical qubit")
        self.ctx = RoutingContext(
            self.circuit,
            self.topology,
            RoutingLayout(self.logical_qubits, self.physical_qubits, list(initial_mapping)),
        )
        self.layers = [0] * self.physical_qubits
        self.swap_count = 0
        self.executed_gates = 0
        self.no_progress_steps = 0
        self.rescue_count = 0
        self.route_events = []
        self.initial_mapping = list(initial_mapping)
        self.candidates = []
        self.candidate_scores = []
        self._advance()
        return None, {}

    def step(self, action):
        if self.ctx is None:
            raise RuntimeError("call reset before step")
        if self.executed_gates == len(self.gates):
            return None, 0.0, True, False, {
                "swap_count": self.swap_count,
                "depth": max(self.layers, default=0),
                "rescue_count": self.rescue_count,
            }
        action = int(action)
        if action < 0 or action >= self.top_k:
            raise ValueError(f"invalid edge action: {action}")
        if not self.candidates:
            return None, 0.0, False, True, {"reason": "no legal swap candidates"}
        requested_rank = action if action < len(self.candidates) else 0
        rescue_needed = self.no_progress_steps >= self.rescue_after
        rescued = rescue_needed and requested_rank != 0
        selected_rank = 0 if rescue_needed or self.simple_chain else requested_rank
        swap = self.candidates[selected_rank]
        last_swap = self.ctx.last_applied_swap()
        if last_swap == swap and self.no_progress_steps and len(self.candidates) > 1:
            selected_rank = next(
                (index for index, candidate in enumerate(self.candidates) if candidate != last_swap),
                selected_rank,
            )
            swap = self.candidates[selected_rank]
        self.ctx.layout().apply_swap(*swap)
        self.rescue_count += int(rescued)
        add_two_qubit_depth(self.layers, *swap)
        self.ctx._last_swap = swap
        self.ctx._swaps_since_progress += 1
        self.route_events.append(("swap", swap))
        self.swap_count += 1
        executed = self._advance()
        self.no_progress_steps = 0 if executed else self.no_progress_steps + 1
        terminated = self.executed_gates == len(self.gates)
        truncated = self.swap_count >= self.max_swaps
        return None, 0.0, terminated, truncated, {
            "swap": swap,
            "executed_gates": executed,
            "swap_count": self.swap_count,
            "depth": max(self.layers, default=0),
            "rescued": rescued,
            "rescue_count": self.rescue_count,
        }

def clone_env(env: HybridRoutingEnv) -> HybridRoutingEnv:
    """Copy only mutable routing state; circuit and topology stay shared."""
    out = copy.copy(env)
    out.policy = copy.copy(env.policy)
    out.policy.decay_state = list(env.policy.decay_state)
    out.layers = list(env.layers)
    out.route_events = list(env.route_events)
    out.initial_mapping = list(env.initial_mapping)
    out.candidates = list(env.candidates)
    out.candidate_scores = list(env.candidate_scores)
    if env.ctx is not None:
        out.ctx = copy.copy(env.ctx)
        out.ctx.layout_obj = RoutingLayout(
            env.logical_qubits,
            env.physical_qubits,
            env.ctx.layout().logical_to_physical,
        )
        out.ctx.remaining_counts = list(env.ctx.remaining_counts)
        out.ctx.done = list(env.ctx.done)
        out.ctx.front_ids = dict(env.ctx.front_ids)
    return out

def metric_key(metrics: RouteMetrics, objective: str, depth_weight: float) -> tuple[float, ...]:
    if objective == "depth":
        return float(metrics.depth), float(metrics.swap_count)
    if objective == "swap-depth":
        return (
            float(metrics.swap_count) + depth_weight * float(metrics.depth),
            float(metrics.swap_count),
            float(metrics.depth),
        )
    return float(metrics.swap_count), float(metrics.depth)

@dataclass(frozen=True)
class HybridEvaluation:
    metrics: RouteMetrics | None
    rescues: int
    baseline: RouteMetrics
    method: str
    beam_calls: int
    layouts_tried: int
    route_events: tuple[tuple[str, object], ...] = ()
    initial_mapping: tuple[int, ...] = ()
    final_mapping: tuple[int, ...] = ()
    baseline_events: tuple[tuple[str, object], ...] = ()
    baseline_initial_mapping: tuple[int, ...] = ()
    baseline_final_mapping: tuple[int, ...] = ()
    runtime_seconds: float = 0.0
    baseline_circuit: object | None = None
    beam_tried: int = 0
    beam_changed: int = 0
    beam_improved: int = 0







def _qiskit_compact_circuit(qasm_path: Path):
    if QuantumCircuit is None or CouplingMap is None:
        raise RuntimeError("Qiskit is required for the canonical LightSABRE baseline")
    source = QuantumCircuit.from_qasm_str(canonical_qasm(qasm_path))
    active = sorted({source.find_bit(q).index for inst in source.data for q in inst.qubits})
    compact = QuantumCircuit(len(active))
    physical = {old: new for new, old in enumerate(active)}
    compact.global_phase = source.global_phase
    for inst in source.data:
        if inst.clbits:
            continue
        qargs = [compact.qubits[physical[source.find_bit(q).index]] for q in inst.qubits]
        compact.append(inst.operation.copy(), qargs)
    return compact


def _qiskit_two_qubit_depth(circuit) -> int:
    layers = [0] * circuit.num_qubits
    for inst in circuit.data:
        if len(inst.qubits) != 2:
            continue
        a, b = (circuit.find_bit(q).index for q in inst.qubits)
        layer = max(layers[a], layers[b]) + 1
        layers[a] = layers[b] = layer
    return max(layers, default=0)


def _replay_complete_cx_trace(
    decomposed,
    physical_qubits: int,
    edges: list[tuple[int, int]],
    trace,
):
    """Replay a SAGE CX/SWAP trace while retaining complete static-QASM semantics."""
    if QuantumRegister is None or ClassicalRegister is None:
        return None
    cx_indices = [
        index for index, instruction in enumerate(decomposed.data)
        if instruction.operation.name == "cx" and len(instruction.qubits) == 2
    ]
    if not cx_indices:
        return None
    last_cx = cx_indices[-1]
    for index, instruction in enumerate(decomposed.data):
        operation = instruction.operation
        if getattr(operation, "condition", None) is not None or instruction.is_control_flow():
            return None
        if index < last_cx and (operation.name in {"measure", "reset", "barrier"} or instruction.clbits):
            return None
        if operation.name == "reset":
            return None

    qreg = QuantumRegister(physical_qubits, "q")
    cregs = [ClassicalRegister(register.size, register.name) for register in decomposed.cregs]
    routed = QuantumCircuit(qreg, *cregs)
    routed.global_phase = decomposed.global_phase
    layout = list(trace[2])
    physical_to_logical = [-1] * physical_qubits
    for logical, physical in enumerate(layout):
        physical_to_logical[physical] = logical

    predecessors: list[set[int]] = []
    last_on_qubit: dict[int, int] = {}
    last_on_clbit: dict[int, int] = {}
    for index, instruction in enumerate(decomposed.data):
        deps = {
            last_on_qubit[decomposed.find_bit(qubit).index]
            for qubit in instruction.qubits
            if decomposed.find_bit(qubit).index in last_on_qubit
        }
        deps.update(
            last_on_clbit[decomposed.find_bit(clbit).index]
            for clbit in instruction.clbits
            if decomposed.find_bit(clbit).index in last_on_clbit
        )
        predecessors.append(deps)
        for qubit in instruction.qubits:
            last_on_qubit[decomposed.find_bit(qubit).index] = index
        for clbit in instruction.clbits:
            last_on_clbit[decomposed.find_bit(clbit).index] = index

    emitted: set[int] = set()
    visiting: set[int] = set()

    def emit(index: int, selected_cx: int | None = None) -> None:
        if index in emitted:
            return
        if index in visiting:
            raise RuntimeError("cyclic complete-circuit dependency")
        visiting.add(index)
        for predecessor in sorted(predecessors[index]):
            emit(predecessor)
        instruction = decomposed.data[index]
        if instruction.operation.name == "cx" and index != selected_cx:
            raise RuntimeError("SAGE CX order conflicts with complete-circuit dependencies")
        qargs = [qreg[layout[decomposed.find_bit(qubit).index]] for qubit in instruction.qubits]
        cargs = [routed.clbits[decomposed.find_bit(clbit).index] for clbit in instruction.clbits]
        routed.append(instruction.operation.copy(), qargs, cargs)
        emitted.add(index)
        visiting.remove(index)

    edge_set = {tuple(sorted(edge)) for edge in edges}
    try:
        for kind, payload in trace[1]:
            if kind == "swap":
                a, b = map(int, payload)
                if tuple(sorted((a, b))) not in edge_set:
                    return None
                routed.swap(qreg[a], qreg[b])
                logical_a, logical_b = physical_to_logical[a], physical_to_logical[b]
                physical_to_logical[a], physical_to_logical[b] = logical_b, logical_a
                if logical_a >= 0:
                    layout[logical_a] = b
                if logical_b >= 0:
                    layout[logical_b] = a
                continue
            gate_id = int(payload)
            instruction_index = cx_indices[gate_id]
            instruction = decomposed.data[instruction_index]
            physical_pair = tuple(sorted(
                layout[decomposed.find_bit(qubit).index] for qubit in instruction.qubits
            ))
            if physical_pair not in edge_set:
                return None
            emit(instruction_index, selected_cx=instruction_index)
        for index in range(len(decomposed.data)):
            emit(index, selected_cx=index if decomposed.data[index].operation.name == "cx" else None)
    except (IndexError, RuntimeError):
        return None
    return routed


def _static_complete_source(source) -> bool:
    """Return whether routing can be replayed before terminal measurements.

    Barriers are compiler directives, not state-changing operations, so they
    do not make an otherwise static circuit dynamic.  The complete exporter
    restores every barrier after routing.
    """
    for index, instruction in enumerate(source.data):
        operation = instruction.operation
        if getattr(operation, "condition", None) is not None or instruction.is_control_flow():
            return False
        if operation.name == "measure":
            measured = set(instruction.qubits)
            if any(
                later.operation.name not in {"measure", "barrier"}
                and measured.intersection(later.qubits)
                for later in source.data[index + 1:]
            ):
                return False
    return True


def _complete_from_cirq_trace(source, qasm_path: Path, physical_qubits: int, edges, trace):
    """Restore terminal measurements around SAGE's original Cirq decomposition."""
    if not _static_complete_source(source):
        return None
    try:
        cirq_routed = build_routed_circuit(
            qasm_path,
            physical_qubits,
            edges,
            trace[1],
            trace[2],
        )
        quantum_qasm = str(cirq.QasmOutput(
            cirq_routed.all_operations(), cirq.LineQubit.range(physical_qubits)
        ))
        quantum = QuantumCircuit.from_qasm_str(quantum_qasm)
    except Exception:
        return None

    cregs = [ClassicalRegister(register.size, register.name) for register in source.cregs]
    complete = QuantumCircuit(QuantumRegister(physical_qubits, "q"), *cregs)
    complete.global_phase = quantum.global_phase
    for instruction in quantum.data:
        qargs = [complete.qubits[quantum.find_bit(qubit).index] for qubit in instruction.qubits]
        complete.append(instruction.operation.copy(), qargs)

    final_mapping = list(trace[3])
    canonical_by_name = {
        name: logical for logical, name in enumerate(canonical_qubit_names(qasm_path))
    }
    source_logical = {}
    source_physical = {}
    used_physical = set(final_mapping)
    free_physical = iter(
        physical for physical in range(physical_qubits) if physical not in used_physical
    )
    for qubit in source.qubits:
        locations = source.find_bit(qubit).registers
        if not locations:
            return None
        register, offset = locations[0]
        logical = canonical_by_name.get(f"{register.name}_{offset}")
        source_logical[qubit] = logical
        source_physical[qubit] = (
            next(free_physical) if logical is None else final_mapping[logical]
        )
    inactive = {qubit for qubit, logical in source_logical.items() if logical is None}
    for instruction in source.data:
        if (
            instruction.operation.name not in {"measure", "barrier"}
            and instruction.qubits
            and all(qubit in inactive for qubit in instruction.qubits)
        ):
            qargs = [complete.qubits[source_physical[qubit]] for qubit in instruction.qubits]
            complete.append(instruction.operation.copy(), qargs)
    # Barriers carry no quantum semantics.  Preserve their positions in a
    # second pass by mapping each source segment to the corresponding routed
    # segment; terminal measurements still use the final routing layout.
    source_segments = []
    current = []
    barriers = []
    for instruction in source.data:
        if instruction.operation.name == "barrier":
            source_segments.append(current)
            current = []
            barriers.append(instruction)
        elif instruction.operation.name != "measure":
            current.append(instruction)
    source_segments.append(current)
    if barriers:
        routed_quantum = list(complete.data)
        rebuilt = QuantumCircuit(QuantumRegister(physical_qubits, "q"), *cregs)
        rebuilt.global_phase = complete.global_phase
        consumed = 0
        total_source = sum(len(segment) for segment in source_segments)
        for segment_index, segment in enumerate(source_segments):
            target = len(routed_quantum) if segment_index == len(source_segments) - 1 else round(
                len(routed_quantum) * sum(len(item) for item in source_segments[:segment_index + 1])
                / max(1, total_source)
            )
            for instruction in routed_quantum[consumed:target]:
                qargs = [rebuilt.qubits[complete.find_bit(qubit).index] for qubit in instruction.qubits]
                rebuilt.append(instruction.operation.copy(), qargs)
            consumed = target
            if segment_index < len(barriers):
                barrier = barriers[segment_index]
                qargs = [rebuilt.qubits[source_physical[qubit]] for qubit in barrier.qubits]
                rebuilt.barrier(*qargs)
        complete = rebuilt
    for instruction in source.data:
        if instruction.operation.name != "measure":
            continue
        qargs = [complete.qubits[source_physical[qubit]] for qubit in instruction.qubits]
        cargs = [complete.clbits[source.find_bit(clbit).index] for clbit in instruction.clbits]
        complete.append(instruction.operation.copy(), qargs, cargs)
    return complete


def _qiskit_canonical_trace(qasm_path: Path, routed, physical_qubits: int):
    """Recover LightSABRE's initial layout and SWAP/gate trace on canonical CXs."""
    logical_qubits, gates = canonical_interactions(qasm_path)
    transpile_layout = getattr(routed, "layout", None)
    if transpile_layout is None or transpile_layout.initial_layout is None:
        return None
    initial = [None] * logical_qubits
    for virtual, input_index in transpile_layout.input_qubit_mapping.items():
        if input_index < logical_qubits:
            initial[input_index] = transpile_layout.initial_layout[virtual]
    if any(physical is None for physical in initial):
        return None

    layout = RoutingLayout(logical_qubits, physical_qubits, initial)
    circuit = RoutingCircuit(logical_qubits, gates)
    remaining = list(circuit.predecessor_counts)
    front = {node_id for node_id, count in enumerate(remaining) if count == 0}
    events = []
    for instruction in routed.data:
        name = instruction.operation.name
        pair = tuple(routed.find_bit(qubit).index for qubit in instruction.qubits)
        if name == "swap":
            events.append(("swap", pair))
            layout.apply_swap(*pair)
            continue
        if name != "cx" or len(pair) != 2:
            continue
        logical_pair = (
            layout.physical_to_logical[pair[0]],
            layout.physical_to_logical[pair[1]],
        )
        matches = [node_id for node_id in sorted(front) if gates[node_id] == logical_pair]
        if not matches:
            return None
        node_id = matches[0]
        events.append(("gate", node_id))
        front.remove(node_id)
        for successor in circuit.node(node_id).successors():
            remaining[successor] -= 1
            if remaining[successor] == 0:
                front.add(successor)
    if len([event for event in events if event[0] == "gate"]) != len(gates):
        return None
    metrics = RouteMetrics(
        sum(event[0] == "swap" for event in events),
        _qiskit_two_qubit_depth(routed),
    )
    return metrics, tuple(events), tuple(initial), tuple(layout.logical_to_physical)


def route_qiskit_lightsabre_baseline(qasm_path: Path, physical_qubits: int, edges: list[tuple[int, int]], seed: int, restarts: int = 6):
    """Canonical Qiskit LightSABRE baseline with the same SWAP/depth definitions."""
    source = _qiskit_compact_circuit(qasm_path)
    directed = []
    for a, b in sorted({tuple(sorted(edge)) for edge in edges}):
        directed.extend(((a, b), (b, a)))
    coupling = CouplingMap(directed)
    best = None
    total_restarts = max(1, int(restarts))
    for restart in range(total_restarts):
        routed = transpile(
            source,
            coupling_map=coupling,
            basis_gates=["u", "cx", "swap"],
            layout_method="sabre",
            routing_method="sabre",
            optimization_level=0,
            seed_transpiler=int(seed) + restart,
        )
        metrics = RouteMetrics(int(routed.count_ops().get("swap", 0)), _qiskit_two_qubit_depth(routed))
        key = (metrics.swap_count, metrics.depth)
        if best is None or key < best[0]:
            best = (key, metrics, routed)
    if best is None:
        raise RuntimeError("Qiskit LightSABRE returned no route")
    return best[1], best[2]

def route_lightsabre_trace(case, lookahead_size: int, seed: int, restarts: int = 6, qasm_path: Path | None = None):
    del lookahead_size
    if qasm_path is None:
        raise ValueError("qasm_path is required for the canonical Qiskit LightSABRE baseline")
    _, _, physical_qubits, edges = case
    metrics, routed = route_qiskit_lightsabre_baseline(qasm_path, physical_qubits, edges, seed, restarts)
    return metrics, (), (), (), routed
@dataclass(frozen=True)
class LoadedCase:
    name: str
    qasm_path: Path
    circuit: tuple[int, list[tuple[int, int]]]
    topology: tuple[int, list[tuple[int, int]]]
    topology_name: str

    @property
    def route_case(self):
        logical, gates = self.circuit
        physical, edges = self.topology
        return logical, gates, physical, edges


@dataclass
class ExternalReferences:
    """Best known SABRE/TKET SWAP counts indexed by fingerprint or circuit name."""

    by_fingerprint: dict[str, int] = field(default_factory=dict)
    by_name: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, fingerprint: str | None, swaps: int) -> None:
        self.by_name[name] = min(swaps, self.by_name.get(name, swaps))
        if fingerprint:
            self.by_fingerprint[fingerprint] = min(swaps, self.by_fingerprint.get(fingerprint, swaps))

    def best(self, name: str, fingerprint: str) -> int | None:
        return self.by_fingerprint.get(fingerprint, self.by_name.get(name))


def load_external_references(paths: list[Path]) -> ExternalReferences:
    """Read tab/comma benchmark logs containing sabre_swaps or tket_swaps."""
    references = ExternalReferences()
    for raw_path in paths:
        path = resolve_path(raw_path)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"reference result not found: {raw_path}")
        header: list[str] | None = None
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            delimiter = "\t" if "\t" in line else ","
            fields = [item.strip() for item in line.split(delimiter)]
            if "circuit" in fields and ({"sabre_swaps", "tket_swaps"} & set(fields)):
                header = fields
                continue
            if header is None or len(fields) < len(header) or fields[0] == "START":
                continue
            row = dict(zip(header, fields))
            swap_text = row.get("sabre_swaps") or row.get("tket_swaps")
            try:
                references.add(row["circuit"], row.get("fingerprint") or None, int(swap_text))
            except (KeyError, TypeError, ValueError):
                continue
    return references

def load_case_manifest(path: Path, require_train: bool = True) -> tuple[list[LoadedCase], list[LoadedCase]]:
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    def load_split(name: str) -> list[LoadedCase]:
        cases = []
        for item in data.get(name, []):
            qasm_path = (path.parent / item["qasm_path"]).resolve()
            topology_path = (path.parent / item["topology_path"]).resolve()
            logical, gates = load_two_qubit_gates(qasm_path)
            topology = load_topology(topology_path)
            if logical > topology[0]:
                raise ValueError(f"{qasm_path.name}: topology has fewer physical qubits")
            cases.append(LoadedCase(item.get("id", qasm_path.stem), qasm_path, (logical, gates), topology, str(topology_path)))
        return cases

    train_cases, test_cases = load_split("train"), load_split("test")
    if not test_cases or (require_train and not train_cases):
        raise ValueError("case manifest must contain test cases and, for training, train cases")
    if not train_cases:
        train_cases = test_cases[:1]
    return train_cases, test_cases


def load_dataset_cases(args: argparse.Namespace) -> tuple[list[LoadedCase], list[LoadedCase], str]:
    manifest = resolve_path(args.case_manifest)
    if manifest is not None:
        train_cases, test_cases = load_case_manifest(manifest, require_train=not args.eval_only)
        return train_cases, test_cases, str(manifest)

    dataset_dir = resolve_path(args.dataset_dir)
    if dataset_dir is None:
        raise ValueError("dataset directory is required")
    train_qasms = sorted((dataset_dir / "train").glob("*.qasm"))
    test_qasms = sorted((dataset_dir / "test").glob("*.qasm"))
    if not test_qasms or (not train_qasms and not args.eval_only):
        raise ValueError("dataset directory must contain test/*.qasm and, unless --eval-only, train/*.qasm")

    loaded = []
    for path in [*train_qasms, *test_qasms]:
        try:
            loaded.append((path, *load_two_qubit_gates(path)))
        except Exception as exc:
            print(f"skip invalid/unparseable QASM: {path.name}: {exc}", flush=True)
    if not loaded:
        raise ValueError("no parseable circuits remain")

    topology_path = resolve_path(args.topology)
    if topology_path is None:
        topology = make_grid_topology(max(logical for _, logical, _ in loaded))
        topology_name = f"auto {math.isqrt(topology[0])}x{math.isqrt(topology[0])} grid"
    else:
        topology = load_topology(topology_path)
        topology_name = str(topology_path)

    loaded_by_path = {path: (logical, gates) for path, logical, gates in loaded}
    def make_cases(paths):
        return [
            LoadedCase(path.stem, path, loaded_by_path[path], topology, topology_name)
            for path in paths if path in loaded_by_path
        ]
    train_cases, test_cases = make_cases(train_qasms), make_cases(test_qasms)
    if not test_cases or (not train_cases and not args.eval_only):
        raise ValueError("no parseable circuits remain in the test set and, unless --eval-only, train set")
    if any(case.circuit[0] > topology[0] for case in [*train_cases, *test_cases]):
        raise ValueError("topology has fewer physical qubits than a dataset circuit")
    return train_cases, test_cases, str(dataset_dir)


def load_routable_operations(qasm_path: Path):
    operations, logical_of = canonical_operations(qasm_path)
    return [
        (op, tuple(logical_of[qubit] for qubit in op.qubits))
        for op in operations
        if 1 <= len(op.qubits) <= 2 and all(qubit in logical_of for qubit in op.qubits)
    ]


def build_routed_circuit(
    qasm_path: Path,
    physical_qubits: int,
    edges: list[tuple[int, int]],
    events: tuple[tuple[str, object], ...],
    initial_mapping: tuple[int, ...],
):
    operations = load_routable_operations(qasm_path)
    two_qubit_ops = [(op, logicals) for op, logicals in operations if len(logicals) == 2]
    tagged = []
    two_qubit_index = 0
    for op, logicals in operations:
        if len(logicals) == 2:
            tagged.append((op, logicals, two_qubit_index))
            two_qubit_index += 1
        else:
            tagged.append((op, logicals, None))

    before_node: dict[int | None, list[tuple[object, int]]] = {}
    next_node: dict[int, int] = {}
    for op, logicals, node_id in reversed(tagged):
        if node_id is not None:
            for logical in logicals:
                next_node[logical] = node_id
        else:
            before_node.setdefault(next_node.get(logicals[0]), []).append((op, logicals[0]))
    for pending in before_node.values():
        pending.reverse()

    layout = RoutingLayout(len(initial_mapping), physical_qubits, list(initial_mapping))
    physical = cirq.LineQubit.range(physical_qubits)
    routed = cirq.Circuit()
    def emit_pending(node_id: int | None) -> None:
        for operation, logical in before_node.pop(node_id, ()):
            routed.append(operation.with_qubits(physical[layout.physical_of_logical(logical)]))

    for kind, payload in events:
        if kind == "swap":
            a, b = payload
            routed.append(cirq.SWAP(physical[a], physical[b]))
            layout.apply_swap(a, b)
            continue
        node_id = int(payload)
        emit_pending(node_id)
        operation, logicals = two_qubit_ops[node_id]
        mapped = [physical[layout.physical_of_logical(logical)] for logical in logicals]
        routed.append(operation.with_qubits(*mapped))
    emit_pending(None)
    if before_node:
        raise RuntimeError(f"route trace did not execute gates required by pending operations: {sorted(before_node)}")

    allowed = {tuple(sorted(edge)) for edge in edges}
    illegal = []
    for index, operation in enumerate(routed.all_operations()):
        if len(operation.qubits) != 2:
            continue
        a, b = (int(qubit.x) for qubit in operation.qubits)
        if tuple(sorted((a, b))) not in allowed:
            illegal.append((index, str(operation), a, b))
    if illegal:
        raise RuntimeError(f"routed circuit contains illegal two-qubit gate: {illegal[0]}")
    return routed


def validate_route_trace(
    loaded_case: LoadedCase,
    result: HybridEvaluation,
    use_baseline: bool,
) -> None:
    """Replay a route trace and prove every executed two-qubit gate is adjacent."""
    if use_baseline and result.baseline_circuit is not None:
        edge_set = {tuple(sorted(edge)) for edge in loaded_case.topology[1]}
        for instruction in result.baseline_circuit.data:
            if len(instruction.qubits) != 2:
                continue
            pair = tuple(sorted(result.baseline_circuit.find_bit(q).index for q in instruction.qubits))
            if pair not in edge_set:
                raise RuntimeError(f"{loaded_case.name}: Qiskit LightSABRE produced non-adjacent gate on {pair}")
        return
    events = result.baseline_events if use_baseline else result.route_events
    initial = result.baseline_initial_mapping if use_baseline else result.initial_mapping
    expected_final = result.baseline_final_mapping if use_baseline else result.final_mapping
    logical_qubits, gates = loaded_case.circuit
    physical_qubits, edges = loaded_case.topology
    edge_set = {tuple(sorted(edge)) for edge in edges}
    layout = RoutingLayout(logical_qubits, physical_qubits, list(initial))
    executed = set()
    for kind, payload in events:
        if kind == "swap":
            a, b = payload
            if tuple(sorted((a, b))) not in edge_set:
                raise RuntimeError(f"{loaded_case.name}: illegal SWAP edge {(a, b)}")
            layout.apply_swap(a, b)
            continue
        node_id = int(payload)
        if node_id in executed or not 0 <= node_id < len(gates):
            raise RuntimeError(f"{loaded_case.name}: invalid gate event {node_id}")
        logical_a, logical_b = gates[node_id]
        physical_pair = tuple(sorted((
            layout.physical_of_logical(logical_a),
            layout.physical_of_logical(logical_b),
        )))
        if physical_pair not in edge_set:
            raise RuntimeError(f"{loaded_case.name}: non-adjacent gate {node_id} on {physical_pair}")
        executed.add(node_id)
    if len(executed) != len(gates):
        raise RuntimeError(f"{loaded_case.name}: executed {len(executed)}/{len(gates)} gates")
    if tuple(layout.logical_to_physical) != tuple(expected_final):
        raise RuntimeError(f"{loaded_case.name}: final mapping mismatch")

def write_routed_qasm(
    loaded_case: LoadedCase,
    result: HybridEvaluation,
    use_baseline: bool,
    output_dir: Path,
    seed: int = 42,
    restarts: int = 1,
    layout_candidates: int = 6,
) -> tuple[Path, int, RouteMetrics, RouteMetrics]:
    """Export a complete routed circuit, including classical operations.

    Export uses the original QASM and Qiskit's semantics-preserving transpiler.
    SAGE layouts are regenerated from Qiskit's real CX decomposition instead
    of reusing only the canonical Cirq layout.  The better complete route is
    written, so complete export cannot regress against LightSABRE.
    """
    if QuantumCircuit is None or transpile is None or qasm2 is None:
        raise RuntimeError("Qiskit is required for complete QASM export")
    physical_qubits, edges = loaded_case.topology
    condition_restorations: dict[tuple[str, str], str] = {}
    original_source_text = ""

    def load_complete_source(path: Path):
        nonlocal original_source_text
        text = path.read_text(encoding="utf-8-sig")
        original_source_text = text
        # Qiskit's OpenQASM 2 exporter serializes conditions through a signed
        # machine integer.  Wide classical registers (for example cc_n151)
        # can legally use much larger literals.  Route with unique small
        # placeholders, then restore the exact literals in the emitted text.
        condition_pattern = re.compile(
            r"(?P<prefix>\bif\s*\(\s*(?P<register>[A-Za-z_]\w*)\s*==\s*)"
            r"(?P<literal>\d+)(?=\s*\))"
        )
        used_literals = {
            int(match.group("literal")) for match in condition_pattern.finditer(text)
        }
        placeholders: dict[tuple[str, str], str] = {}
        next_placeholder = 0

        def replace_wide_condition(match):
            nonlocal next_placeholder
            literal = match.group("literal")
            if int(literal) <= (1 << 63) - 1:
                return match.group(0)
            key = (match.group("register"), literal)
            placeholder = placeholders.get(key)
            if placeholder is None:
                while next_placeholder in used_literals:
                    next_placeholder += 1
                placeholder = str(next_placeholder)
                used_literals.add(next_placeholder)
                next_placeholder += 1
                placeholders[key] = placeholder
                condition_restorations[(key[0], placeholder)] = literal
            return match.group("prefix") + placeholder

        text = condition_pattern.sub(replace_wide_condition, text)
        try:
            return QuantumCircuit.from_qasm_str(text)
        except Exception as original_error:
            qregs = re.findall(r"(?m)^\s*qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;", text)
            cregs = {
                name for name, _ in
                re.findall(r"(?m)^\s*creg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;", text)
            }
            repairs = []
            if len(qregs) == 1:
                qreg_name = qregs[0][0]
                declared_qregs = {name for name, _ in qregs}
                measured_qregs = set(re.findall(
                    r"(?m)^\s*measure\s+([A-Za-z_]\w*)\s*\[", text
                ))
                unknown = measured_qregs - declared_qregs
                if len(unknown) == 1:
                    old = next(iter(unknown))
                    text = re.sub(
                        rf"(?m)^(\s*measure\s+){re.escape(old)}(\s*\[)",
                        rf"\g<1>{qreg_name}\g<2>",
                        text,
                    )
                    repairs.append(f"measurement register {old}->{qreg_name}")
            targets = re.findall(
                r"(?m)^\s*measure\s+[^;]+->\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]",
                text,
            )
            missing = {}
            for name, index in targets:
                if name not in cregs:
                    missing[name] = max(missing.get(name, 0), int(index) + 1)
            if missing:
                declarations = "\n".join(
                    f"creg {name}[{size}];" for name, size in sorted(missing.items())
                )
                qreg_matches = list(re.finditer(r"(?m)^\s*qreg\s+[^;]+;", text))
                if qreg_matches:
                    position = qreg_matches[-1].end()
                    text = text[:position] + "\n" + declarations + text[position:]
                    repairs.append("added " + ", ".join(
                        f"creg {name}[{size}]" for name, size in sorted(missing.items())
                    ))
            if not repairs:
                raise original_error
            try:
                circuit = QuantumCircuit.from_qasm_str(text)
            except Exception:
                raise original_error
            print(
                f"complete QASM source repair: {path.name}: " + "; ".join(repairs),
                flush=True,
            )
            return circuit

    source = load_complete_source(loaded_case.qasm_path)
    if source.num_qubits > physical_qubits:
        raise ValueError(
            f"{loaded_case.name}: complete circuit has {source.num_qubits} qubits "
            f"but topology has only {physical_qubits}"
        )
    directed = []
    for a, b in sorted({tuple(sorted(edge)) for edge in edges}):
        directed.extend(((a, b), (b, a)))
    coupling = CouplingMap(directed)

    def qasm2_compatible(circuit):
        """Lower if-without-else blocks to per-instruction QASM 2 conditions."""
        lowered = circuit.copy_empty_like()
        lowered.global_phase = circuit.global_phase
        for instruction in circuit.data:
            operation = instruction.operation
            if operation.name != "if_else":
                lowered.append(operation.copy(), instruction.qubits, instruction.clbits)
                continue
            blocks = operation.blocks
            if len(blocks) != 1 or any(
                item.operation.name in {"if_else", "while_loop", "for_loop", "switch_case"}
                for item in blocks[0].data
            ):
                lowered.append(operation.copy(), instruction.qubits, instruction.clbits)
                continue
            body = blocks[0]
            for inner in body.data:
                qargs = [instruction.qubits[body.find_bit(qubit).index] for qubit in inner.qubits]
                cargs = [instruction.clbits[body.find_bit(clbit).index] for clbit in inner.clbits]
                with lowered.if_test(operation.condition):
                    lowered.append(inner.operation.copy(), qargs, cargs)
        return lowered

    def route(initial_layout, trials: int):
        best = None
        for restart in range(max(1, int(trials))):
            kwargs = dict(
                coupling_map=coupling,
                basis_gates=["u", "cx", "swap"],
                routing_method="sabre",
                optimization_level=0,
                seed_transpiler=int(seed) + restart,
            )
            if initial_layout is None:
                kwargs["layout_method"] = "sabre"
            else:
                kwargs["initial_layout"] = initial_layout
                kwargs["layout_method"] = None
            candidate = transpile(source, **kwargs)
            metrics = RouteMetrics(
                int(candidate.count_ops().get("swap", 0)),
                _qiskit_two_qubit_depth(candidate),
            )
            key = (metrics.swap_count, metrics.depth)
            if best is None or key < best[0]:
                best = (key, metrics, candidate)
        return best

    baseline = None
    choices = []
    canonical_baseline = None
    if _static_complete_source(source):
        canonical_metrics = result.baseline
        canonical_routed = result.baseline_circuit
        if canonical_routed is None:
            canonical_metrics, canonical_routed = route_qiskit_lightsabre_baseline(
                loaded_case.qasm_path, physical_qubits, edges, seed, restarts
            )
        canonical_trace = _qiskit_canonical_trace(
            loaded_case.qasm_path, canonical_routed, physical_qubits
        )
        if canonical_trace is not None:
            canonical_complete = _complete_from_cirq_trace(
                source, loaded_case.qasm_path, physical_qubits, edges, canonical_trace
            )
            if canonical_complete is not None:
                canonical_baseline = (
                    metric_key(canonical_metrics, "swap", 0.0),
                    "canonical_lightsabre_complete_trace",
                    canonical_metrics,
                    canonical_complete,
                    list(canonical_trace[2]),
                )
                choices.append(canonical_baseline)

    # Static circuits compare and export the already-computed canonical
    # LightSABRE/SAGE traces.  Only dynamic circuits need a second complete
    # Qiskit routing problem; its candidates cannot affect static selection.
    if canonical_baseline is not None:
        baseline = (canonical_baseline[0], canonical_baseline[2], canonical_baseline[3])
        decomposed = None
        real_cx_gates = []
    else:
        decomposed = transpile(
            source,
            basis_gates=["u", "cx"],
            optimization_level=0,
            seed_transpiler=int(seed),
        )
        real_cx_gates = [
            tuple(decomposed.find_bit(qubit).index for qubit in instruction.qubits)
            for instruction in decomposed.data
            if instruction.operation.name == "cx" and len(instruction.qubits) == 2
        ]
        baseline = route(None, restarts)
        choices.append((baseline[0], "complete_lightsabre", baseline[1], baseline[2], None))

    # This is the exact SAGE routing problem used by the canonical benchmark:
    # Cirq's original two-qubit decomposition plus the real one-qubit gates.
    # Replaying it restores the high routing gain without dropping terminal
    # measurements. Dynamic circuits remain on the conservative Qiskit path.
    canonical_sage = None
    if result.metrics is not None and result.route_events:
        canonical_trace = (
            result.metrics,
            result.route_events,
            result.initial_mapping,
            result.final_mapping,
        )
        canonical_complete = _complete_from_cirq_trace(
            source, loaded_case.qasm_path, physical_qubits, edges, canonical_trace
        )
        if canonical_complete is not None:
            canonical_metrics = RouteMetrics(
                result.metrics.swap_count,
                _qiskit_two_qubit_depth(canonical_complete),
            )
            canonical_sage = (
                metric_key(canonical_metrics, "swap", 0.0),
                "canonical_sage_complete_trace",
                canonical_metrics,
                canonical_complete,
                list(result.initial_mapping),
            )
            choices.append(canonical_sage)
    # Keep the canonical winner as one candidate, but no longer let it be the
    # sole SAGE representation of the complete routing problem.
    search_initial = result.initial_mapping
    canonical_layout = None
    if search_initial:
        names = canonical_qubit_names(loaded_case.qasm_path)
        if len(names) != len(search_initial):
            raise RuntimeError(f"{loaded_case.name}: canonical layout length mismatch")
        physical_by_name = dict(zip(names, map(int, search_initial)))
        expanded = [None] * source.num_qubits
        used = set()
        for index, qubit in enumerate(source.qubits):
            locations = source.find_bit(qubit).registers
            if not locations:
                continue
            register, offset = locations[0]
            physical = physical_by_name.get(f"{register.name}_{offset}")
            if physical is not None:
                expanded[index] = physical
                used.add(physical)
        free = iter(qubit for qubit in range(physical_qubits) if qubit not in used)
        canonical_layout = [next(free) if physical is None else physical for physical in expanded]

    sage_routes = []
    if real_cx_gates:
        full_classification = classify_circuit((source.num_qubits, real_cx_gates))
        topology_profile = classify_topology(physical_qubits, edges).profile
        full_classification = replace(full_classification, topology_profile=topology_profile)
        full_env = DenseHybridRoutingEnv(
            source.num_qubits,
            real_cx_gates,
            physical_qubits,
            edges,
            lookahead_size=20 if full_classification.routing_profile == "local" else 64,
            baseline_lookahead_size=20,
            top_k=5,
            objective="swap",
            depth_weight=0.05,
            routing_profile=full_classification.routing_profile,
        )
        full_env.circuit_pattern = full_classification.circuit_pattern
        full_env.topology_profile = topology_profile
        full_env.policy.topology_profile = topology_profile
        full_env.lightsabre_metrics = baseline[1]
        full_env.lightsabre_swaps = baseline[1].swap_count
        # Prescoring must also see seeds that start worse but improve under
        # Beam; final selection still guarantees no LightSABRE regression.
        full_env.max_swaps = max(200, 2 * baseline[1].swap_count)
        pool = generate_structure_aware_layouts(
            full_env,
            max(8, int(layout_candidates) * 2),
            int(seed),
            4,
        )
        if full_env.physical_qubits <= 49 and full_env.logical_qubits <= 49:
            pool.extend(layout for layout, _ in _medium_sabre_refined_layouts(full_env, pool))
        if canonical_layout is not None:
            pool.append(canonical_layout)

        ranked = []
        seen = set()
        for index, layout in enumerate(pool):
            key = tuple(layout)
            if key in seen:
                continue
            seen.add(key)
            prescore = _rank0_metrics(full_env, layout)
            if prescore is not None:
                ranked.append((metric_key(prescore, "swap", 0.0), index, layout))
        ranked.sort(key=lambda item: (item[0], item[1]))
        selected_layouts = ranked[:max(1, int(layout_candidates))]
        if canonical_layout is not None and all(layout != canonical_layout for _, _, layout in selected_layouts):
            canonical_score = _rank0_metrics(full_env, canonical_layout)
            if canonical_score is not None:
                selected_layouts.append((
                    metric_key(canonical_score, "swap", 0.0), len(pool), canonical_layout
                ))

        beam_args = argparse.Namespace(
            seed=int(seed),
            beam_width=4,
            beam_depth=4,
            beam_branch=3,
            beam_interval=4,
            objective="swap",
            depth_weight=0.05,
        )
        trace_candidates = []
        for selected_index, (_, _, layout) in enumerate(selected_layouts):
            candidate = route(layout, 1)
            method = "real_cx_sage_layout" if layout != canonical_layout else "canonical_sage_layout"
            sage_routes.append((candidate[0], method, candidate[1], candidate[2], layout))
            choices.append(sage_routes[-1])
            trace = _rank0_trace(full_env, layout)
            if trace is not None:
                trace_candidates.append(trace)
                replayed = _replay_complete_cx_trace(
                    decomposed, physical_qubits, edges, trace
                )
                if replayed is not None:
                    replay_metrics = RouteMetrics(
                        int(replayed.count_ops().get("swap", 0)),
                        _qiskit_two_qubit_depth(replayed),
                    )
                    replay_method = (
                        "real_cx_sage_trace"
                        if layout != canonical_layout else "canonical_seed_real_cx_sage_trace"
                    )
                    sage_routes.append((
                        metric_key(replay_metrics, "swap", 0.0),
                        replay_method,
                        replay_metrics,
                        replayed,
                        layout,
                    ))
                    choices.append(sage_routes[-1])
            if selected_index < 2:
                beam_trace = _beam_trace(full_env, layout, beam_args)
                if beam_trace is not None:
                    replayed = _replay_complete_cx_trace(
                        decomposed, physical_qubits, edges, beam_trace
                    )
                    if replayed is not None:
                        replay_metrics = RouteMetrics(
                            int(replayed.count_ops().get("swap", 0)),
                            _qiskit_two_qubit_depth(replayed),
                        )
                        sage_routes.append((
                            metric_key(replay_metrics, "swap", 0.0),
                            "real_cx_sage_beam_trace",
                            replay_metrics,
                            replayed,
                            layout,
                        ))
                        choices.append(sage_routes[-1])

        # If bounded rank-0/Beam still trails LightSABRE, reuse SAGE's existing
        # routing-scored evolution on the real CX problem. This is deliberately
        # capped; complete QASM export is validation, not an unbounded search.
        if (
            trace_candidates
            and min(trace[0].swap_count for trace in trace_candidates) >= baseline[1].swap_count
            and len(real_cx_gates) <= 1500
        ):
            evolution_args = argparse.Namespace(
                seed=int(seed),
                evolution_budget=32 if len(real_cx_gates) >= 600 else 64,
                evolution_population=8,
                evolution_seeds=min(8, len(selected_layouts)),
                _extra_deadline=time.perf_counter() + 20.0,
            )
            evolved, _ = _evolve_layout_by_routing(
                full_env,
                [layout for _, _, layout in selected_layouts],
                evolution_args,
                max(0, baseline[1].swap_count - 1),
            )
            if evolved is not None:
                replayed = _replay_complete_cx_trace(
                    decomposed, physical_qubits, edges, evolved
                )
                if replayed is not None:
                    replay_metrics = RouteMetrics(
                        int(replayed.count_ops().get("swap", 0)),
                        _qiskit_two_qubit_depth(replayed),
                    )
                    sage_routes.append((
                        metric_key(replay_metrics, "swap", 0.0),
                        "real_cx_sage_evolved_trace",
                        replay_metrics,
                        replayed,
                        list(evolved[2]),
                    ))
                    choices.append(sage_routes[-1])

    sage_best = min(sage_routes, key=lambda item: item[0]) if sage_routes else None
    if canonical_sage is not None and (sage_best is None or canonical_sage[0] < sage_best[0]):
        sage_best = canonical_sage
    comparison_baseline = canonical_baseline if canonical_baseline is not None else choices[0]
    comparable_choices = [
        choice for choice in choices
        if choice[1] in {
            "canonical_lightsabre_complete_trace",
            "canonical_sage_complete_trace",
        }
    ] if canonical_baseline is not None else choices
    dumped = None
    export_errors = []
    for _, selected, metrics, routed, expanded in sorted(comparable_choices, key=lambda item: item[0]):
        try:
            dumped = qasm2.dumps(qasm2_compatible(routed))
            break
        except Exception as error:
            export_errors.append(f"{selected}: {type(error).__name__}: {error}")
            continue
    if dumped is None:
        detail = export_errors[-1] if export_errors else "no export candidates"
        raise RuntimeError(
            f"{loaded_case.name}: no complete route can be exported as OpenQASM 2; {detail}"
        )
    for (register, placeholder), literal in condition_restorations.items():
        dumped, replacements = re.subn(
            rf"(?P<prefix>\bif\s*\(\s*{re.escape(register)}\s*==\s*)"
            rf"{re.escape(placeholder)}(?=\s*\))",
            lambda match, value=literal: match.group("prefix") + value,
            dumped,
        )
        if replacements == 0:
            raise RuntimeError(
                f"{loaded_case.name}: failed to restore wide condition {register}=={literal}"
            )
    reparsed = QuantumCircuit.from_qasm_str(dumped)
    validation_source = (
        QuantumCircuit.from_qasm_str(original_source_text)
        if condition_restorations else source
    )
    for operation in ("measure", "reset"):
        before = int(validation_source.count_ops().get(operation, 0))
        after = int(reparsed.count_ops().get(operation, 0))
        if before != after:
            raise RuntimeError(
                f"{loaded_case.name}: complete export changed {operation} count {before}->{after}"
            )
    if validation_source.num_clbits != reparsed.num_clbits:
        raise RuntimeError(
            f"{loaded_case.name}: complete export changed classical-bit count "
            f"{validation_source.num_clbits}->{reparsed.num_clbits}"
        )
    def conditional_non_swap_count(circuit, inherited=False) -> int:
        total = 0
        for instruction in circuit.data:
            operation = instruction.operation
            conditional = inherited or getattr(operation, "condition", None) is not None
            blocks = getattr(operation, "blocks", ())
            if blocks:
                total += sum(
                    conditional_non_swap_count(block, conditional) for block in blocks
                )
            elif conditional and operation.name != "swap":
                total += 1
        return total
    if conditional_non_swap_count(validation_source) != conditional_non_swap_count(reparsed):
        raise RuntimeError(
            f"{loaded_case.name}: complete export changed conditioned operation count "
            f"{conditional_non_swap_count(validation_source)}->"
            f"{conditional_non_swap_count(reparsed)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{loaded_case.name}_routed.qasm"
    output_path.write_text(dumped, encoding="utf-8")
    mapping_path = output_dir / f"{loaded_case.name}_mapping.json"
    mapping_path.write_text(json.dumps({
        "circuit": loaded_case.name,
        "topology": loaded_case.topology_name,
        "search_selected": "lightsabre_fallback" if use_baseline else "sage_xv",
        "full_export_selected": selected,
        "full_export_initial_layout": expanded,
        "full_export_swaps": metrics.swap_count,
        "full_export_two_qubit_depth": metrics.depth,
        "full_lightsabre_swaps": baseline[1].swap_count,
        "full_lightsabre_two_qubit_depth": baseline[1].depth,
        "full_comparison_baseline": comparison_baseline[1],
        "full_comparison_lightsabre_swaps": comparison_baseline[2].swap_count,
        "full_comparison_lightsabre_two_qubit_depth": comparison_baseline[2].depth,
        "full_sage_layout_swaps": None if sage_best is None else sage_best[2].swap_count,
        "full_sage_layout_two_qubit_depth": None if sage_best is None else sage_best[2].depth,
        "full_raw_best_swaps": None if sage_best is None else sage_best[2].swap_count,
        "full_raw_best_two_qubit_depth": None if sage_best is None else sage_best[2].depth,
        "full_raw_best_method": None if sage_best is None else sage_best[1],
        "full_layout_source": "canonical_cirq_and_qiskit_real_cx",
        "full_layout_candidates": len(sage_routes),
        "full_real_cx_gates": len(real_cx_gates),
        "full_export_swap_reduction": comparison_baseline[2].swap_count - metrics.swap_count,
        "full_export_optimization_percent": (
            0.0
            if comparison_baseline[2].swap_count == 0
            else 100.0 * (comparison_baseline[2].swap_count - metrics.swap_count) / comparison_baseline[2].swap_count
        ),
        "source_qubits": source.num_qubits,
        "source_clbits": source.num_clbits,
        "source_measurements": int(source.count_ops().get("measure", 0)),
        "source_resets": int(source.count_ops().get("reset", 0)),
        "complete_qasm": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"complete QASM export: {loaded_case.name} selected={selected} "
        f"swaps={metrics.swap_count} baseline_swaps={comparison_baseline[2].swap_count} "
        f"sage_layout_swaps={None if sage_best is None else sage_best[2].swap_count} "
        f"layout_candidates={len(sage_routes)} depth={metrics.depth}",
        flush=True,
    )
    return output_path, int(len(routed.data)), metrics, comparison_baseline[2]


def write_mapping_json(
    loaded_case: LoadedCase,
    result: HybridEvaluation,
    use_baseline: bool,
    output_dir: Path,
) -> Path:
    initial = result.baseline_initial_mapping if use_baseline else result.initial_mapping
    final = result.baseline_final_mapping if use_baseline else result.final_mapping
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{loaded_case.name}_mapping.json"
    path.write_text(json.dumps({
        "circuit": loaded_case.name,
        "topology": loaded_case.topology_name,
        "logical_to_physical_initial": list(initial),
        "logical_to_physical_final": list(final),
        "selected": "lightsabre_fallback" if use_baseline else "sage_xv",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def _scale(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (value - low) / max(1e-9, high - low)))


def circuit_features(circuit: tuple[int, list[tuple[int, int]]]) -> dict[str, float | int]:
    """Extract name-independent interaction-graph and gate-sequence features."""
    logical_qubits, gates = circuit
    normalized = [tuple(sorted((int(a), int(b)))) for a, b in gates if a != b]
    active = sorted({qubit for gate in normalized for qubit in gate})
    pair_counts = Counter(normalized)
    adjacency = {qubit: set() for qubit in active}
    for a, b in pair_counts:
        adjacency[a].add(b)
        adjacency[b].add(a)

    components = []
    unseen = set(active)
    while unseen:
        root = unseen.pop()
        component = {root}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            new_nodes = adjacency[node] & unseen
            unseen.difference_update(new_nodes)
            component.update(new_nodes)
            queue.extend(new_nodes)
        components.append(component)

    diameter = 0
    for component in components:
        for root in component:
            distances = {root: 0}
            queue = deque([root])
            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node] & component:
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            diameter = max(diameter, max(distances.values(), default=0))

    clustering = []
    for node in active:
        neighbors = adjacency[node]
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        if possible == 0:
            clustering.append(0.0)
            continue
        links = sum(1 for a in neighbors for b in adjacency[a] if a < b and b in neighbors)
        clustering.append(links / possible)

    overlaps = [
        bool(set(normalized[index - 1]) & set(normalized[index]))
        for index in range(1, len(normalized))
    ]
    last_seen: dict[int, int] = {}
    reuse_gaps = []
    for index, gate in enumerate(normalized):
        for qubit in gate:
            if qubit in last_seen:
                reuse_gaps.append(index - last_seen[qubit])
            last_seen[qubit] = index

    windows = [tuple(normalized[index:index + 3]) for index in range(max(0, len(normalized) - 2))]
    active_count = len(active)
    gate_count = len(normalized)
    unique_pairs = len(pair_counts)
    possible_pairs = active_count * (active_count - 1) / 2
    degrees = [len(adjacency[node]) for node in active]
    mean_gap = statistics.fmean(reuse_gaps) if reuse_gaps else float(active_count or 1)
    mean_degree = statistics.fmean(degrees) if degrees else 0.0
    max_degree = max(degrees, default=0)
    hub = max(active, key=lambda node: (len(adjacency[node]), -node), default=-1)
    second_degree = sorted(degrees, reverse=True)[1] if len(degrees) > 1 else 0
    hub_edges = sum(1 for pair in pair_counts if hub in pair)
    hub_gate_count = sum(count for pair, count in pair_counts.items() if hub in pair)
    hub_gate_flags = [hub in gate for gate in normalized]
    hub_overlaps = [
        hub_gate_flags[index - 1] and hub_gate_flags[index]
        for index in range(1, len(hub_gate_flags))
    ]
    degree_total = sum(degrees)
    degree_entropy = 0.0
    if degree_total > 0 and len(degrees) > 1:
        probabilities = [degree / degree_total for degree in degrees if degree]
        degree_entropy = -sum(p * math.log(p) for p in probabilities) / math.log(len(degrees))
    first_seen_pair: dict[tuple[int, int], int] = {}
    for index, pair in enumerate(normalized):
        first_seen_pair.setdefault(pair, index)
    early_cutoff = max(1, gate_count // 4)
    return {
        "logical_qubits": logical_qubits,
        "active_qubits": active_count,
        "active_fraction": active_count / max(1, logical_qubits),
        "two_qubit_gates": gate_count,
        "unique_interactions": unique_pairs,
        "interaction_density": unique_pairs / max(1.0, possible_pairs),
        "unique_interaction_ratio": unique_pairs / max(1, gate_count),
        "repeated_interaction_ratio": 1.0 - unique_pairs / max(1, gate_count),
        "gates_per_active_qubit": gate_count / max(1, active_count),
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "mean_degree_ratio": mean_degree / max(1, active_count - 1),
        "max_degree_ratio": max_degree / max(1, active_count - 1),
        "degree_std": statistics.pstdev(degrees) if len(degrees) > 1 else 0.0,
        "degree_cv": statistics.pstdev(degrees) / max(1e-9, mean_degree) if len(degrees) > 1 else 0.0,
        "degree_entropy": degree_entropy,
        "hub_edge_coverage": hub_edges / max(1, unique_pairs),
        "hub_degree_gap": (max_degree - second_degree) / max(1, active_count - 1),
        "leaf_leaf_ratio": 1.0 - hub_edges / max(1, unique_pairs),
        "hub_gate_ratio": hub_gate_count / max(1, gate_count),
        "hub_consecutive_ratio": statistics.fmean(hub_overlaps) if hub_overlaps else 0.0,
        "early_unique_ratio": sum(index < early_cutoff for index in first_seen_pair.values()) / max(1, unique_pairs),
        "components": len(components),
        "diameter": diameter,
        "clustering": statistics.fmean(clustering) if clustering else 0.0,
        "consecutive_overlap_ratio": statistics.fmean(overlaps) if overlaps else 0.0,
        "mean_reuse_gap": mean_gap,
        "temporal_locality": 1.0 - min(1.0, mean_gap / max(1, active_count)),
        "pattern_repetition": 1.0 - len(set(windows)) / max(1, len(windows)),
    }


@dataclass(frozen=True)
class Classification:
    expert: str
    routing_profile: str
    confidence: float
    scores: dict[str, float]
    features: dict[str, float | int]
    circuit_pattern: str = "generic"
    topology_profile: str = "unknown"


def classify_circuit_pattern(features: dict[str, float | int]) -> str:
    """Recognize QFT-, BV-, and CC-like structures without using circuit names."""
    active = int(features["active_qubits"])
    gates = int(features["two_qubit_gates"])
    unique = int(features["unique_interactions"])
    possible = active * (active - 1) // 2
    if (
        active >= 4
        and float(features["interaction_density"]) >= 0.80
        and abs(gates - 2 * unique) <= max(1, unique // 100)
        and float(features["degree_cv"]) <= 0.15
        and float(features["degree_entropy"]) >= 0.98
        and float(features["unique_interaction_ratio"]) >= 0.45
    ):
        return "qft_like"
    if (
        _is_star_hub(features)
        and gates == unique
        and gates == active - 1
        and float(features["repeated_interaction_ratio"]) <= 1e-9
    ):
        return "bv_like"
    if (
        _is_star_hub(features)
        and gates == active
        and unique == active - 1
        and 0.0 < float(features["repeated_interaction_ratio"]) <= 0.15
    ):
        return "cc_like"
    return "generic"

def _is_star_hub(features: dict[str, float | int]) -> bool:
    """Identify a genuine one-hub graph without confusing dense regular graphs."""
    return (
        float(features["max_degree_ratio"]) >= 0.75
        and float(features["hub_edge_coverage"]) >= 0.80
        and float(features["leaf_leaf_ratio"]) <= 0.20
        and float(features["unique_interaction_ratio"]) >= 0.65
        and float(features["gates_per_active_qubit"]) >= 0.8
    )


def _is_paired_hub(features: dict[str, float | int]) -> bool:
    """Detect decomposed cswap/KNN graphs: one hub plus repeated leaf pairs."""
    return (
        float(features["max_degree_ratio"]) >= 0.45
        and float(features["hub_degree_gap"]) >= 0.40
        and float(features["hub_edge_coverage"]) >= 0.45
        and float(features["leaf_leaf_ratio"]) <= 0.55
        and float(features["hub_gate_ratio"]) >= 0.40
        and float(features["consecutive_overlap_ratio"]) >= 0.75
        and float(features["repeated_interaction_ratio"]) >= 0.60
        and float(features["unique_interaction_ratio"]) <= 0.35
    )


def _is_large_paired_hub(features: dict[str, float | int]) -> bool:
    return _is_paired_hub(features) and int(features["active_qubits"]) >= 128


def _is_hub_global(features: dict[str, float | int]) -> bool:
    repeat = float(features["repeated_interaction_ratio"])
    unique_ratio = float(features["unique_interaction_ratio"])
    repeated_hub = (
        float(features["max_degree_ratio"]) >= 0.45
        and float(features["hub_edge_coverage"]) >= 0.35
        and float(features["hub_gate_ratio"]) >= 0.45
        and 0.15 <= float(features["hub_consecutive_ratio"]) <= 0.85
        and 0.45 <= repeat <= 0.90
        and 0.05 <= unique_ratio <= 0.50
    )
    return _is_star_hub(features) or repeated_hub


def _is_global_symmetric(features: dict[str, float | int]) -> bool:
    """Detect dense graphs with many near-equivalent logical placements."""
    return (
        float(features["degree_cv"]) <= 0.20
        and float(features["degree_entropy"]) >= 0.85
        and float(features["mean_degree_ratio"]) >= 0.50
        and float(features["gates_per_active_qubit"]) >= 10.0
    )


def _is_front_loaded_repeated(features: dict[str, float | int]) -> bool:
    """Detect bounded circuits whose early overlapping interactions deserve extra search."""
    return (
        350 <= int(features["two_qubit_gates"]) <= 1500
        and int(features["active_qubits"]) <= 80
        and float(features["consecutive_overlap_ratio"]) >= 0.50
        and float(features["repeated_interaction_ratio"]) >= 0.15
    )


def _medium_specializations(
    features: dict[str, float | int], routing_profile: str, physical_qubits: int
) -> dict[str, bool]:
    """Enable extra candidates only on Medium-sized hardware."""
    medium = physical_qubits <= 49 and int(features["active_qubits"]) <= 49
    return {
        "repeat_temporal": medium
        and float(features["repeated_interaction_ratio"]) >= 0.90
        and float(features["gates_per_active_qubit"]) >= 40.0
        and int(features["two_qubit_gates"]) < 1000,
        "long_temporal": medium
        and routing_profile == "local"
        and int(features["two_qubit_gates"]) >= 1000,
        "symmetric_refine": medium
        and routing_profile == "symmetric"
        and int(features["active_qubits"]) <= 32,
        "hub_refine": medium and routing_profile == "hub_sequence",
        "dense_refine": medium and routing_profile == "dense_temporal",
        "small_nonlocal_refine": medium and routing_profile in {"dense_temporal", "symmetric", "hub_sequence"}
        and int(features["two_qubit_gates"]) >= 80
        and float(features["interaction_density"]) >= 0.12,
    }

def classify_circuit(circuit: tuple[int, list[tuple[int, int]]]) -> Classification:
    features = circuit_features(circuit)
    repeat = float(features["repeated_interaction_ratio"])
    unique_ratio = float(features["unique_interaction_ratio"])
    arithmetic = (
        0.35 * repeat
        + 0.20 * _scale(float(features["gates_per_active_qubit"]), 3.0, 20.0)
        + 0.20 * float(features["temporal_locality"])
        + 0.15 * float(features["pattern_repetition"])
        + 0.10 * float(features["consecutive_overlap_ratio"])
    )
    dense = (
        0.25 * _scale(float(features["interaction_density"]), 0.03, 0.35)
        + 0.25 * _scale(float(features["mean_degree"]), 3.0, 12.0)
        + 0.20 * unique_ratio
        + 0.15 * (1.0 - repeat)
        + 0.15 * _scale(float(features["gates_per_active_qubit"]), 2.0, 12.0)
    )
    hub = (
        0.40 * _scale(float(features["max_degree_ratio"]), 0.30, 0.60)
        + 0.25 * _scale(float(features["hub_edge_coverage"]), 0.30, 0.85)
        + 0.20 * _scale(float(features["hub_consecutive_ratio"]), 0.30, 0.80)
        + 0.15 * _scale(unique_ratio, 0.15, 0.50)
    )
    hub_global = _is_hub_global(features) or _is_large_paired_hub(features)
    global_symmetric = _is_global_symmetric(features)
    global_score = max(dense, hub) if hub_global else dense
    scores = {"arithmetic": arithmetic, "dense_global": global_score}
    if global_symmetric:
        expert, winner, runner = "general", 1.0 - max(arithmetic, dense), min(arithmetic, dense)
    elif (dense >= 0.56 and dense >= arithmetic - 0.02) or hub_global:
        expert, winner, runner = "dense_global", global_score, arithmetic
    elif arithmetic >= 0.62 and arithmetic >= dense - 0.05:
        expert, winner, runner = "arithmetic", arithmetic, dense
    else:
        expert, winner, runner = "general", 1.0 - max(arithmetic, dense), min(arithmetic, dense)

    if hub_global:
        routing_profile = "hub_sequence"
        if _is_star_hub(features):
            expert = "general"
    elif global_symmetric:
        routing_profile = "symmetric"
    elif expert == "dense_global" or dense >= 0.56:
        routing_profile = "dense_temporal"
    else:
        routing_profile = "local"
    confidence = max(0.35, min(0.99, 0.5 + 0.5 * (winner - runner)))
    return Classification(expert, routing_profile, confidence, scores, features, classify_circuit_pattern(features))


def _interaction_weights(
    gates: list[tuple[int, int]], profile: str
) -> Counter[tuple[int, int]]:
    weights: Counter[tuple[int, int]] = Counter()
    window = max(32, min(512, int(math.sqrt(max(1, len(gates))) * 8)))
    for index, (a, b) in enumerate(gates):
        pair = tuple(sorted((a, b)))
        if profile == "early":
            weight = 0.2 + 4.0 / (1.0 + index / window)
        elif profile == "front":
            weight = 5.0 if index < window else 0.1
        else:
            weight = 1.0
        weights[pair] += weight
    if profile == "repeat":
        for pair, weight in list(weights.items()):
            weights[pair] = weight * math.sqrt(weight)
    elif profile == "unique":
        for pair in weights:
            weights[pair] = 1.0
    return weights


def _medium_interaction_weights(
    gates: list[tuple[int, int]], mode: str
) -> Counter[tuple[int, int]]:
    """Combine repetition with a slow time decay for long Medium circuits."""
    tau = max(64.0, len(gates) / (3.0 if mode == "repeat_temporal" else 2.0))
    weights: Counter[tuple[int, int]] = Counter()
    counts = Counter(tuple(sorted((a, b))) for a, b in gates)
    for index, (a, b) in enumerate(gates):
        pair = tuple(sorted((a, b)))
        repeat_weight = math.sqrt(counts[pair]) if mode == "repeat_temporal" else 1.0
        weights[pair] += repeat_weight * (0.25 + math.exp(-index / tau))
    return weights


def _windowed_interaction_weights(
    gates: list[tuple[int, int]], phase: float, window_fraction: float = 0.35
) -> Counter[tuple[int, int]]:
    """Focus a deterministic layout on one execution phase while retaining global structure."""
    count = max(1, len(gates))
    width = max(16.0, count * window_fraction)
    center = phase * max(0, count - 1)
    frequencies = Counter(tuple(sorted(pair)) for pair in gates)
    weights: Counter[tuple[int, int]] = Counter()
    for index, pair in enumerate(gates):
        normalized = tuple(sorted(pair))
        temporal = math.exp(-((index - center) / width) ** 2)
        weights[normalized] += 0.10 + 2.5 * temporal + 0.20 * math.sqrt(frequencies[normalized])
    return weights


def _front_exponential_weights(
    gates: list[tuple[int, int]], fraction: float = 0.08
) -> Counter[tuple[int, int]]:
    """Strongly prioritize the opening segment without discarding repeated interactions."""
    tau = max(8.0, len(gates) * fraction)
    frequencies = Counter(tuple(sorted(pair)) for pair in gates)
    weights: Counter[tuple[int, int]] = Counter()
    for index, pair in enumerate(gates):
        normalized = tuple(sorted(pair))
        weights[normalized] += (
            0.05
            + 3.0 * math.exp(-index / tau)
            + 0.10 * math.sqrt(frequencies[normalized])
        )
    return weights


def _layout_pair_cost(
    layout: list[int], topology, weights: Counter[tuple[int, int]]
) -> float:
    return sum(weight * max(0, topology.distance(layout[a], layout[b]) - 1) for (a, b), weight in weights.items())


def _refine_layout_by_logical_swaps(
    env,
    initial: list[int],
    weights: Counter[tuple[int, int]],
    passes: int = 3,
    candidate_limit: int = 24,
) -> list[int]:
    """Locally improve our layout by swapping influential logical assignments."""
    layout = list(initial)
    loads = Counter()
    for (a, b), weight in weights.items():
        loads[a] += weight
        loads[b] += weight
    logicals = [q for q, _ in loads.most_common(candidate_limit)]
    best_cost = _layout_pair_cost(layout, env.topology, weights)
    for _ in range(max(1, passes)):
        best_move = None
        for index, a in enumerate(logicals):
            for b in logicals[index + 1:]:
                layout[a], layout[b] = layout[b], layout[a]
                cost = _layout_pair_cost(layout, env.topology, weights)
                layout[a], layout[b] = layout[b], layout[a]
                key = (cost, a, b)
                if cost + 1e-9 < best_cost and (best_move is None or key < best_move):
                    best_move = key
        if best_move is None:
            break
        best_cost, a, b = best_move
        layout[a], layout[b] = layout[b], layout[a]
    return layout


def generate_targeted_layouts(env, seeds: list[list[int]]) -> list[list[int]]:
    """Generate pure SAGE layouts for known hard structures without external compilers."""
    candidates: list[list[int]] = []
    seen = {tuple(seed) for seed in seeds}
    base = seeds[0]
    weight_sets = [
        _windowed_interaction_weights(env.gates, phase, fraction)
        for phase, fraction in ((0.0, 0.22), (0.25, 0.30), (0.50, 0.35), (0.75, 0.30), (1.0, 0.22))
    ]
    weight_sets.extend((
        _interaction_weights(env.gates, "repeat"),
        _medium_interaction_weights(env.gates, "repeat_temporal"),
        _medium_interaction_weights(env.gates, "long_temporal"),
    ))
    features = circuit_features((env.logical_qubits, env.gates))
    if _is_front_loaded_repeated(features):
        weight_sets[:0] = [
            _front_exponential_weights(env.gates, fraction)
            for fraction in (0.05, 0.08, 0.12)
        ]
    for weights in weight_sets:
        region_limit = 8 if env.physical_qubits <= 49 else 4
        for region_candidate in _region_embedding_layouts(env, weights, region_limit):
            key = tuple(region_candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(region_candidate)
        embedded = _weighted_embedding_layout(env, base, weights)
        for candidate in (embedded, None if embedded is None else _refine_layout_by_logical_swaps(env, embedded, weights)):
            if candidate is None:
                continue
            key = tuple(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    for seed in seeds[:3]:
        reverse = _reverse_rank0_layout(env, seed)
        forward = None if reverse is None else _rank0_final_layout(env, reverse, env.gates)
        for candidate in (reverse, forward):
            if candidate is not None and tuple(candidate) not in seen:
                seen.add(tuple(candidate))
                candidates.append(candidate)
    return candidates

def _target_policy_profiles(env) -> list[tuple[int, float, float]]:
    """Small deterministic policy grid; Medium gets a little more exploration."""
    if env.physical_qubits <= 49:
        lookaheads = (8, 12, 20, 32, 48, 64)
        weights = (0.25, 0.50, 0.75)
    else:
        lookaheads = (20, 32, 64)
        weights = (0.35, 0.50, 0.75)
    return [(lookahead, weight, 0.001) for lookahead in lookaheads for weight in weights]


def _best_profile_trace(env, initial: list[int]):
    """Find the best SAGE rank-0 parameters for one fixed in-house layout."""
    original = (
        env.policy.lookahead_size,
        env.policy.lookahead_weight,
        env.policy.decay_increment,
    )
    best = None
    best_profile = original
    try:
        for lookahead, weight, decay in _target_policy_profiles(env):
            env.policy.lookahead_size = lookahead
            env.policy.lookahead_weight = weight
            env.policy.decay_increment = decay
            candidate = _rank0_trace(env, initial)
            if candidate is not None and (
                best is None
                or metric_key(candidate[0], "swap", 0.0) < metric_key(best[0], "swap", 0.0)
            ):
                best = candidate
                best_profile = (lookahead, weight, decay)
    finally:
        env.policy.lookahead_size, env.policy.lookahead_weight, env.policy.decay_increment = original
    return best, best_profile


def _layout_neighbors(
    env,
    layout: list[int],
    weights: Counter[tuple[int, int]],
    logical_limit: int,
    empty_limit: int,
) -> list[list[int]]:
    """Swap logical assignments or move an influential logical qubit into an empty site."""
    loads = Counter()
    adjacency: dict[int, Counter[int]] = {q: Counter() for q in range(env.logical_qubits)}
    for (a, b), weight in weights.items():
        loads[a] += weight
        loads[b] += weight
        adjacency[a][b] += weight
        adjacency[b][a] += weight
    logicals = [q for q, _ in loads.most_common(max(2, logical_limit))]
    out: list[list[int]] = []
    for index, a in enumerate(logicals):
        for b in logicals[index + 1:]:
            candidate = list(layout)
            candidate[a], candidate[b] = candidate[b], candidate[a]
            out.append(candidate)

    unused = set(range(env.physical_qubits)) - set(layout)
    for logical in logicals:
        best_empty = sorted(
            unused,
            key=lambda physical: (
                sum(
                    weight * env.topology.distance(physical, layout[other])
                    for other, weight in adjacency[logical].items()
                ),
                -len(env.topology.neighbors(physical)),
                physical,
            ),
        )[: max(1, empty_limit)]
        for physical in best_empty:
            candidate = list(layout)
            candidate[logical] = physical
            out.append(candidate)
    return out


def _search_layout_by_routing(
    env,
    initial: list[int],
    args: argparse.Namespace,
    external_target: int | None,
):
    """Hill-climb the initial mapping using actual routed SWAP count as the score."""
    best, profile = _best_profile_trace(env, initial)
    if best is None:
        return None, 0
    best_layout = list(best[2])
    evaluations = len(_target_policy_profiles(env))
    weights = _windowed_interaction_weights(env.gates, 0.25, 0.40)
    weights.update(_windowed_interaction_weights(env.gates, 0.75, 0.40))
    budget = max(1, int(args.layout_search_budget))
    rounds = max(1, int(args.layout_search_rounds))
    for _ in range(rounds):
        neighbors = _layout_neighbors(
            env,
            best_layout,
            weights,
            int(args.layout_search_logicals),
            int(args.layout_search_empty),
        )
        neighbors.sort(key=lambda layout: (_layout_pair_cost(layout, env.topology, weights), tuple(layout)))
        improved = None
        lookahead, lookahead_weight, decay = profile
        original = (
            env.policy.lookahead_size,
            env.policy.lookahead_weight,
            env.policy.decay_increment,
        )
        try:
            env.policy.lookahead_size = lookahead
            env.policy.lookahead_weight = lookahead_weight
            env.policy.decay_increment = decay
            for candidate_layout in neighbors[: max(0, budget - evaluations)]:
                candidate = _rank0_trace(env, candidate_layout)
                evaluations += 1
                if candidate is None:
                    continue
                if metric_key(candidate[0], "swap", 0.0) < metric_key(best[0], "swap", 0.0):
                    if improved is None or metric_key(candidate[0], "swap", 0.0) < metric_key(improved[0], "swap", 0.0):
                        improved = candidate
                if external_target is not None and candidate[0].swap_count <= external_target:
                    improved = candidate
                    break
                if evaluations >= budget:
                    break
        finally:
            env.policy.lookahead_size, env.policy.lookahead_weight, env.policy.decay_increment = original
        if improved is None:
            break
        best = improved
        best_layout = list(best[2])
        if external_target is not None and best[0].swap_count <= external_target:
            break
        if evaluations >= budget:
            break
    return best, evaluations


def _mutate_layout(env, layout: list[int], rng: random.Random) -> list[int]:
    """Move a heavily interacting distant pair closer instead of blind swapping."""
    candidate = list(layout)
    logical_qubits = len(candidate)
    if logical_qubits < 2:
        return candidate

    weights = Counter(tuple(sorted(pair)) for pair in env.gates if pair[0] != pair[1])
    ranked = sorted(
        weights,
        key=lambda pair: (
            weights[pair] * max(0, env.topology.distance(candidate[pair[0]], candidate[pair[1]]) - 1),
            weights[pair],
            pair,
        ),
        reverse=True,
    )[:10]
    if not ranked:
        a, b = rng.sample(range(logical_qubits), 2)
        candidate[a], candidate[b] = candidate[b], candidate[a]
        return candidate

    a, b = rng.choice(ranked)
    unused = set(range(env.physical_qubits)) - set(candidate)
    moves = []
    for logical, other in ((a, b), (b, a)):
        other_physical = candidate[other]
        for physical in env.topology.neighbors(other_physical):
            if physical in unused:
                moves.append((env.topology.distance(physical, other_physical), logical, physical))
    if moves:
        _, logical, physical = min(moves)
        candidate[logical] = physical
    else:
        candidate[a], candidate[b] = candidate[b], candidate[a]
    return candidate

def _evolution_rng(seed: int, layout: list[int]) -> random.Random:
    mixed = seed & 0xFFFFFFFFFFFFFFFF
    for physical in layout:
        mixed = (mixed * 6364136223846793005 + physical + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return random.Random(mixed)



def _evolution_budget(env, args: argparse.Namespace) -> int:
    """Scale probes inversely with complete-route cost."""
    budget = max(0, int(args.evolution_budget))
    gates = max(1, len(env.gates))
    if gates >= 5000 or env.logical_qubits >= 128:
        return 0
    if gates >= 1500 or env.logical_qubits >= 64:
        return min(budget, 16)
    if gates >= 600:
        return min(budget, 32)
    if gates >= 200:
        return min(budget, 64)
    return min(budget, 120)

def _evolve_layout_by_routing(
    env,
    seeds: list[list[int]],
    args: argparse.Namespace,
    external_target: int,
):
    """Evolve only in-house layouts, scored by SAGE's complete routed SWAP count."""
    budget = _evolution_budget(env, args)
    population_size = max(2, int(args.evolution_population))
    seed_limit = max(1, int(args.evolution_seeds))
    valid_seeds = []
    seen_seeds = set()
    for seed in seeds:
        key = tuple(seed)
        if (
            len(seed) == env.logical_qubits
            and len(set(seed)) == len(seed)
            and all(0 <= physical < env.physical_qubits for physical in seed)
            and key not in seen_seeds
        ):
            seen_seeds.add(key)
            valid_seeds.append(list(seed))
        if len(valid_seeds) >= seed_limit:
            break
    if budget <= 0 or not valid_seeds:
        return None, 0

    best, profile = _best_profile_trace(env, valid_seeds[0])
    evaluations = len(_target_policy_profiles(env))
    if best is None:
        return None, evaluations
    ranked = [(best[0].swap_count, best[0].depth, tuple(best[2]), best)]
    seen = {tuple(best[2])}
    if best[0].swap_count <= external_target or evaluations >= budget:
        return best, evaluations

    original = (
        env.policy.lookahead_size,
        env.policy.lookahead_weight,
        env.policy.decay_increment,
    )
    rng = _evolution_rng(args.seed, valid_seeds[0])
    no_improve = 0
    best_key = ranked[0][:2]
    try:
        env.policy.lookahead_size, env.policy.lookahead_weight, env.policy.decay_increment = profile
        for seed in valid_seeds[1:]:
            if not _extra_budget_available(args):
                break
            key = tuple(seed)
            if key in seen or evaluations >= budget:
                continue
            seen.add(key)
            trace = _rank0_trace(env, seed)
            evaluations += 1
            if trace is None:
                continue
            ranked.append((trace[0].swap_count, trace[0].depth, key, trace))
            ranked.sort(key=lambda item: item[:3])
            ranked = ranked[:population_size]
            if trace[0].swap_count <= external_target:
                return trace, evaluations

        while evaluations < budget and ranked and _extra_budget_available(args):
            parents = [list(item[2]) for item in ranked[:population_size]]
            offspring = []
            attempts = 0
            batch = max(population_size * 4, 16)
            while len(offspring) < batch and attempts < batch * 10:
                attempts += 1
                candidate = _mutate_layout(env, rng.choice(parents), rng)
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                offspring.append((key, candidate))
            if not offspring:
                break
            for key, candidate in offspring:
                if evaluations >= budget or not _extra_budget_available(args):
                    break
                trace = _rank0_trace(env, candidate)
                evaluations += 1
                if trace is None:
                    continue
                ranked.append((trace[0].swap_count, trace[0].depth, key, trace))
                if trace[0].swap_count <= external_target:
                    return trace, evaluations
            ranked.sort(key=lambda item: item[:3])
            ranked = ranked[:population_size]
            if ranked[0][:2] < best_key:
                best_key = ranked[0][:2]
                no_improve = 0
            else:
                no_improve += len(offspring)
                if no_improve >= 15:
                    break
    finally:
        env.policy.lookahead_size, env.policy.lookahead_weight, env.policy.decay_increment = original
    return ranked[0][3], evaluations

def _weighted_embedding_layout(
    env, base: list[int], weights: Counter[tuple[int, int]]
) -> list[int] | None:
    components = env.circuit.logical_interaction_components()
    used = {logical for component in components for logical in component}
    components.extend([[logical] for logical in range(env.logical_qubits) if logical not in used])
    candidate = list(base)
    for component in components:
        physicals = [base[logical] for logical in component]
        assignment = _embed_component(component, physicals, env.topology, weights)
        for logical, physical in assignment.items():
            candidate[logical] = physical
    return candidate if len(set(candidate)) == len(candidate) else None

def _embed_component(
    logicals: list[int],
    physicals: list[int],
    topology,
    weights: Counter[tuple[int, int]],
) -> dict[int, int]:
    if len(logicals) <= 1:
        return dict(zip(logicals, physicals))
    adjacency = {logical: {} for logical in logicals}
    for (a, b), weight in weights.items():
        if a in adjacency and b in adjacency:
            adjacency[a][b] = weight
            adjacency[b][a] = weight
    weighted_degree = {logical: sum(adjacency[logical].values()) for logical in logicals}
    centrality = {
        physical: sum(topology.distance(physical, other) for other in physicals)
        for physical in physicals
    }
    first_logical = max(logicals, key=lambda q: (weighted_degree[q], -q))
    first_physical = min(
        physicals,
        key=lambda p: (centrality[p], -len(topology.neighbors(p)), p),
    )
    mapping = {first_logical: first_physical}
    unused_logical = set(logicals) - {first_logical}
    unused_physical = set(physicals) - {first_physical}
    while unused_logical:
        logical = max(
            unused_logical,
            key=lambda q: (
                sum(adjacency[q].get(other, 0.0) for other in mapping),
                weighted_degree[q],
                -q,
            ),
        )
        mapped_neighbors = [
            (other, weight)
            for other, weight in adjacency[logical].items()
            if other in mapping
        ]
        physical = min(
            unused_physical,
            key=lambda p: (
                sum(
                    weight * topology.distance(p, mapping[other])
                    for other, weight in mapped_neighbors
                ),
                centrality[p],
                -len(topology.neighbors(p)),
                p,
            ),
        )
        mapping[logical] = physical
        unused_logical.remove(logical)
        unused_physical.remove(physical)

    # Refining only influential qubits keeps the setup cost small on large circuits.
    candidates = sorted(logicals, key=lambda q: (-weighted_degree[q], q))[:32]
    for _ in range(2):
        improved = False
        for index, a in enumerate(candidates):
            for b in candidates[index + 1:]:
                pa, pb = mapping[a], mapping[b]
                delta = 0.0
                for other, weight in adjacency[a].items():
                    if other != b:
                        delta += weight * (
                            topology.distance(pb, mapping[other])
                            - topology.distance(pa, mapping[other])
                        )
                for other, weight in adjacency[b].items():
                    if other != a:
                        delta += weight * (
                            topology.distance(pa, mapping[other])
                            - topology.distance(pb, mapping[other])
                        )
                if delta < -1e-9:
                    mapping[a], mapping[b] = pb, pa
                    improved = True
        if not improved:
            break
    return mapping


def _compact_physical_region(env, size: int) -> list[int] | None:
    """Choose a small connected hardware region without inheriting the baseline placement."""
    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    components = [component for component in nx.connected_components(graph) if len(component) >= size]
    if not components:
        return None
    component = min(components, key=lambda nodes: (len(nodes), min(nodes)))
    seed = min(
        component,
        key=lambda node: (
            sum(env.topology.distance(node, other) for other in component),
            -graph.degree[node],
            node,
        ),
    )
    selected = {seed}
    frontier = set(graph.neighbors(seed)) & component
    while len(selected) < size and frontier:
        node = min(
            frontier,
            key=lambda candidate: (
                sum(env.topology.distance(candidate, other) for other in selected),
                -sum(neighbor not in selected for neighbor in graph.neighbors(candidate)),
                candidate,
            ),
        )
        selected.add(node)
        frontier.remove(node)
        frontier.update(set(graph.neighbors(node)) & component - selected)
    return sorted(selected) if len(selected) == size else None


def _compact_physical_regions(env, size: int, limit: int = 8) -> list[list[int]]:
    """Return several compact connected regions instead of committing to one chip center."""
    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    starts = sorted(
        range(env.physical_qubits),
        key=lambda node: (
            sum(env.topology.distance(node, other) for other in range(env.physical_qubits)),
            -graph.degree[node],
            node,
        ),
    )
    stride = max(1, len(starts) // max(1, limit * 2))
    sampled = list(dict.fromkeys([*starts[:limit], *starts[::stride][:limit]]))
    regions = []
    seen = set()
    for start in sampled:
        selected = {start}
        frontier = set(graph.neighbors(start))
        while len(selected) < size and frontier:
            node = min(
                frontier,
                key=lambda candidate: (
                    sum(env.topology.distance(candidate, other) for other in selected),
                    -sum(neighbor not in selected for neighbor in graph.neighbors(candidate)),
                    candidate,
                ),
            )
            selected.add(node)
            frontier.remove(node)
            frontier.update(set(graph.neighbors(node)) - selected)
        key = tuple(sorted(selected))
        if len(key) == size and key not in seen:
            seen.add(key)
            regions.append(list(key))
    return regions


def _region_embedding_layouts(
    env, weights: Counter[tuple[int, int]], limit: int = 8
) -> list[list[int]]:
    """Embed the logical graph into several compact physical regions using SAGE's own mapper."""
    layouts = []
    logicals = list(range(env.logical_qubits))
    for region in _compact_physical_regions(env, env.logical_qubits, limit):
        assignment = _embed_component(logicals, region, env.topology, weights)
        candidate = [assignment[logical] for logical in logicals]
        if len(set(candidate)) == len(candidate):
            layouts.append(candidate)
    return layouts

def _connected_for_spectral(graph: nx.Graph, order: list[int]) -> nx.Graph:
    """Add negligible deterministic links so spectral_layout handles isolated components."""
    out = graph.copy()
    components = [sorted(component) for component in nx.connected_components(out)]
    components.sort(key=lambda component: order.index(component[0]) if component[0] in order else component[0])
    for left, right in zip(components, components[1:]):
        out.add_edge(left[0], right[0], weight=1e-6)
    return out


def _temporal_interaction_weights(gates: list[tuple[int, int]], temporal: bool) -> Counter[tuple[int, int]]:
    if not temporal:
        return _interaction_weights(gates, "uniform")
    tau = max(16.0, math.sqrt(max(1, len(gates))) * 8.0)
    weights: Counter[tuple[int, int]] = Counter()
    for index, (a, b) in enumerate(gates):
        weights[tuple(sorted((a, b)))] += math.exp(-index / tau)
    return weights


def _spectral_layout(
    env, base: list[int], seed: int, temporal: bool = False
) -> list[int] | None:
    """Align 2D logical and physical spectral embeddings over eight orientations."""
    del seed
    physicals = _compact_physical_region(env, env.logical_qubits)
    if physicals is None:
        return None
    weights = _temporal_interaction_weights(env.gates, temporal)
    first_gate = {logical: len(env.gates) for logical in range(env.logical_qubits)}
    for index, (a, b) in enumerate(env.gates):
        first_gate[a] = min(first_gate[a], index)
        first_gate[b] = min(first_gate[b], index)
    logical_order = sorted(range(env.logical_qubits), key=lambda logical: (first_gate[logical], logical))

    logical_graph = nx.Graph()
    logical_graph.add_nodes_from(logical_order)
    logical_graph.add_weighted_edges_from((a, b, float(weight)) for (a, b), weight in weights.items())
    logical_graph = _connected_for_spectral(logical_graph, logical_order)
    physical_graph = nx.Graph()
    physical_graph.add_nodes_from(physicals)
    physical_graph.add_edges_from((a, b) for a, b in env.edges if a in physical_graph and b in physical_graph)
    try:
        logical_coords = nx.spectral_layout(logical_graph, dim=2, weight="weight", scale=1.0)
        physical_coords = nx.spectral_layout(physical_graph, dim=2, weight="weight", scale=1.0)
    except (nx.NetworkXException, np.linalg.LinAlgError, ValueError):
        return None

    best: tuple[float, tuple[int, ...], list[int]] | None = None
    for swap_axes in (False, True):
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                oriented = {
                    physical: np.asarray((
                        sign_x * coords[int(swap_axes)],
                        sign_y * coords[1 - int(swap_axes)],
                    ))
                    for physical, coords in physical_coords.items()
                }
                unused = set(physicals)
                candidate = list(base)
                for logical in logical_order:
                    coordinate = logical_coords[logical]
                    physical = min(
                        unused,
                        key=lambda node: (
                            float(np.sum((coordinate - oriented[node]) ** 2)),
                            sum(env.topology.distance(node, other) for other in physicals),
                            node,
                        ),
                    )
                    candidate[logical] = physical
                    unused.remove(physical)
                cost = sum(
                    weight * env.topology.distance(candidate[a], candidate[b])
                    for (a, b), weight in weights.items()
                )
                key = (cost, tuple(candidate), candidate)
                if best is None or key[:2] < best[:2]:
                    best = key
    return None if best is None else best[2]


def _ring_layouts(env, base: list[int], limit: int = 8) -> list[list[int]]:
    """Rotate and reverse logical order on a ring; bounded and deterministic."""
    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    cycles = nx.cycle_basis(graph)
    if not cycles:
        return []
    ring = cycles[0]
    logical_order = list(range(env.logical_qubits))
    layouts = []
    seen = set()
    stride = max(1, len(ring) // max(1, limit // 2))
    for order in (ring, [ring[0], *reversed(ring[1:])]):
        for offset in range(0, len(ring), stride):
            candidate = list(base)
            physical_order = order[offset:] + order[:offset]
            for logical, physical in zip(logical_order, physical_order):
                candidate[logical] = physical
            key = tuple(candidate)
            if len(set(candidate)) == len(candidate) and key not in seen:
                seen.add(key)
                layouts.append(candidate)
                if len(layouts) >= limit:
                    return layouts
    return layouts

def _heavy_hex_ring_layouts(env, base: list[int], limit: int = 5) -> list[list[int]]:
    """Map the densest logical core onto complete, unique Heavy-hex walks."""
    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    cycles = sorted(nx.cycle_basis(graph), key=lambda cycle: (-len(cycle), tuple(sorted(cycle))))
    if not cycles:
        return []
    logical = nx.Graph()
    logical.add_nodes_from(range(env.logical_qubits))
    logical.add_edges_from(env.gates)
    core = list(nx.core_number(logical).items()) if logical.number_of_edges() else []
    logical_order = [node for node, _ in sorted(core, key=lambda item: (-item[1], -logical.degree[item[0]], item[0]))]
    logical_seen = set(logical_order)
    logical_order.extend(node for node in range(env.logical_qubits) if node not in logical_seen)
    walk_pool = _graph_walks(env, limit=max(5, limit))
    graph_order = sorted(
        graph,
        key=lambda node: (-graph.degree[node], sum(env.topology.distance(node, other) for other in graph), node),
    )
    layouts: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for cycle in cycles[:max(2, limit)]:
        for physical_order in (cycle, list(reversed(cycle))):
            order: list[int] = []
            for source in (physical_order, *(walk_pool or []), graph_order):
                for node in source:
                    if node not in order:
                        order.append(node)
                    if len(order) >= env.logical_qubits:
                        break
                if len(order) >= env.logical_qubits:
                    break
            if len(order) < env.logical_qubits or len(set(order)) != len(order):
                continue
            candidate = [-1] * env.logical_qubits
            used: set[int] = set()
            for logical_qubit, physical_qubit in zip(logical_order, order[:env.logical_qubits]):
                candidate[logical_qubit] = physical_qubit
                used.add(physical_qubit)
            # Preserve the base layout only for positions not covered by the
            # Heavy-hex embedding, then fill any remaining holes uniquely.
            base_free = [physical for physical in base if physical not in used]
            all_free = [physical for physical in range(env.physical_qubits) if physical not in used]
            filler = iter(base_free + [physical for physical in all_free if physical not in base_free])
            for logical_qubit, physical_qubit in enumerate(candidate):
                if physical_qubit < 0:
                    candidate[logical_qubit] = next(filler)
            key = tuple(candidate)
            if len(set(candidate)) == len(candidate) and key not in seen:
                seen.add(key)
                layouts.append(candidate)
                if len(layouts) >= limit:
                    return layouts
    return layouts


def _heavy_hex_center_ring_layout(env, base: list[int]) -> list[int] | None:
    """Place high-load logical qubits on a Heavy-hex ring and its branches."""
    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    cycles = nx.cycle_basis(graph)
    if not cycles or env.logical_qubits > env.physical_qubits:
        return None
    ring = max(cycles, key=lambda cycle: (len(cycle), tuple(sorted(cycle))))
    ring_set = set(ring)
    branch = [
        node for node in graph
        if node not in ring_set and any(neighbor in ring_set for neighbor in graph.neighbors(node))
    ]
    branch.sort(key=lambda node: (
        -sum(neighbor in ring_set for neighbor in graph.neighbors(node)),
        sum(env.topology.distance(node, other) for other in graph),
        node,
    ))
    remaining = [
        node for node in graph
        if node not in ring_set and node not in set(branch)
    ]
    remaining.sort(key=lambda node: (
        sum(env.topology.distance(node, other) for other in graph),
        -graph.degree[node],
        node,
    ))
    physical_order = list(ring) + branch + remaining
    if len(physical_order) < env.logical_qubits or len(set(physical_order)) != len(physical_order):
        return None
    loads = Counter(qubit for gate in env.gates for qubit in gate)
    logical_order = sorted(
        range(env.logical_qubits),
        key=lambda qubit: (-loads[qubit], -sum(1 for gate in env.gates if qubit in gate), qubit),
    )
    candidate = list(physical_order[:env.logical_qubits])
    # ``base`` is only used to make the fallback deterministic if the graph
    # construction ever yields fewer physical positions than expected.
    if len(candidate) < env.logical_qubits:
        candidate.extend(physical for physical in base if physical not in candidate)
    mapping = [-1] * env.logical_qubits
    for logical, physical in zip(logical_order, candidate):
        mapping[logical] = physical
    if min(mapping, default=-1) < 0 or len(set(mapping)) != env.logical_qubits:
        return None
    return mapping


def _graph_walks(env, limit: int = 8) -> list[list[int]]:
    """Return bounded deterministic physical walks selected for the current topology."""
    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    profile = getattr(env, "topology_profile", "irregular_sparse")
    walks: list[list[int]] = []
    seen = set()

    def add(order) -> None:
        walk = list(order)[:env.logical_qubits]
        key = tuple(walk)
        if len(walk) == env.logical_qubits and len(set(walk)) == len(walk) and key not in seen:
            seen.add(key)
            walks.append(walk)

    if profile == "ring_like":
        start = min(graph)
        cycle = nx.cycle_basis(graph)
        if cycle:
            ring = cycle[0]
            while ring[0] != start:
                ring = ring[1:] + ring[:1]
            orientations = (ring, [ring[0], *reversed(ring[1:])])
            stride = max(1, len(ring) // max(1, limit // 2))
            for order in orientations:
                for offset in range(0, len(order), stride):
                    add(order[offset:] + order[:offset])
                    if len(walks) >= limit:
                        return walks
        return walks

    if profile == "path_like":
        endpoints = sorted(node for node, degree in graph.degree() if degree == 1)
        if endpoints:
            add(nx.shortest_path(graph, endpoints[0], endpoints[-1]))
            add(reversed(nx.shortest_path(graph, endpoints[0], endpoints[-1])))
        return walks

    starts = sorted(
        graph,
        key=lambda node: (
            -graph.degree[node] if profile in {"grid_like", "heavy_hex_like"} else graph.degree[node],
            sum(env.topology.distance(node, other) for other in graph),
            node,
        ),
    )[:max(4, limit)]
    for start in starts:
        path = [start]
        used = {start}
        while len(path) < env.logical_qubits:
            choices = [node for node in graph.neighbors(path[-1]) if node not in used]
            if not choices:
                break
            nxt = max(
                choices,
                key=lambda node: (
                    len([neighbor for neighbor in graph.neighbors(node) if neighbor not in used]),
                    -sum(env.topology.distance(node, other) for other in graph),
                    -node,
                ),
            )
            path.append(nxt)
            used.add(nxt)
        add(path)
        if len(walks) >= limit:
            break
    return walks


def _hub_topology_layouts(env, base: list[int], pattern: str, limit: int = 8) -> list[list[int]]:
    """Place BV/CC-like hub interactions on topology-appropriate walks/regions."""
    hub = getattr(env, "hub_logical", -1)
    if hub < 0:
        return []
    visits = []
    counts = Counter()
    seen_logical = {hub}
    for a, b in env.gates:
        if hub not in (a, b):
            continue
        leaf = b if a == hub else a
        counts[leaf] += 1
        if leaf not in seen_logical:
            seen_logical.add(leaf)
            visits.append(leaf)
    orders = [visits, list(reversed(visits))]
    if pattern == "cc_like":
        orders.insert(1, sorted(visits, key=lambda leaf: (-counts[leaf], visits.index(leaf))))
    logical_tail = [q for q in range(env.logical_qubits) if q not in seen_logical]
    layouts = []
    layout_seen = set()
    for physical_order in _graph_walks(env, limit=max(limit, 4)):
        for leaves in orders:
            candidate = list(base)
            for logical, physical in zip([hub, *leaves, *logical_tail], physical_order):
                candidate[logical] = physical
            key = tuple(candidate)
            if len(set(candidate)) == len(candidate) and key not in layout_seen:
                layout_seen.add(key)
                layouts.append(candidate)
                if len(layouts) >= limit:
                    return layouts
    return layouts


def _qft_topology_layouts(env, base: list[int], limit: int = 8) -> list[list[int]]:
    """Generate phase-, order-, and topology-aware seeds for complete symmetric interactions."""
    candidates = []
    seen = set()

    def add(candidate: list[int] | None) -> None:
        if candidate is None:
            return
        key = tuple(candidate)
        if len(set(candidate)) == len(candidate) and key not in seen:
            seen.add(key)
            candidates.append(candidate)

    phase_budget = max(2, limit // 2)
    for phase, fraction in ((0.0, 0.18), (0.20, 0.24), (0.50, 0.30), (0.80, 0.24), (1.0, 0.18)):
        weights = _windowed_interaction_weights(env.gates, phase, fraction)
        embedded = _weighted_embedding_layout(env, base, weights)
        add(embedded)
        if embedded is not None:
            add(_refine_layout_by_logical_swaps(env, embedded, weights, passes=3))
        if len(candidates) >= phase_budget:
            break

    first_seen = {logical: len(env.gates) for logical in range(env.logical_qubits)}
    for index, pair in enumerate(env.gates):
        for logical in pair:
            first_seen[logical] = min(first_seen[logical], index)
    logical_orders = [
        sorted(range(env.logical_qubits), key=lambda logical: (first_seen[logical], logical)),
        sorted(range(env.logical_qubits), key=lambda logical: (-first_seen[logical], logical)),
    ]
    for physical_order in _graph_walks(env, limit=4):
        for logical_order in logical_orders:
            candidate = list(base)
            for logical, physical in zip(logical_order, physical_order):
                candidate[logical] = physical
            add(candidate)
    return candidates[:limit]

def _topology_profile_layouts(
    env,
    base: list[int],
    routing_profile: str,
    limit: int = 4,
) -> list[list[int]]:
    """Generate candidates only when the hardware is not the established grid case."""
    topology_profile = getattr(env, "topology_profile", "grid_like")
    if topology_profile == "grid_like":
        return []
    weights = _interaction_weights(
        env.gates,
        "early" if routing_profile in {"local", "hub_sequence"} else "uniform",
    )
    layouts: list[list[int]] = []
    seen = set()

    def add(candidate: list[int] | None) -> None:
        if candidate is None:
            return
        key = tuple(candidate)
        if len(set(candidate)) == len(candidate) and key not in seen:
            seen.add(key)
            layouts.append(candidate)

    if topology_profile in {"ring_like", "path_like"}:
        first_seen = {logical: len(env.gates) for logical in range(env.logical_qubits)}
        weighted_degree = Counter()
        for index, (a, b) in enumerate(env.gates):
            first_seen[a] = min(first_seen[a], index)
            first_seen[b] = min(first_seen[b], index)
            pair = tuple(sorted((a, b)))
            weighted_degree[a] += weights[pair]
            weighted_degree[b] += weights[pair]
        orders = (
            sorted(range(env.logical_qubits), key=lambda logical: (first_seen[logical], logical)),
            sorted(range(env.logical_qubits), key=lambda logical: (-weighted_degree[logical], first_seen[logical], logical)),
        )
        for physical_order in _graph_walks(env, limit=max(4, limit)):
            for logical_order in orders:
                candidate = list(base)
                for logical, physical in zip(logical_order, physical_order):
                    candidate[logical] = physical
                add(candidate)
                if len(layouts) >= limit:
                    return layouts
        return layouts

    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    articulation = set(nx.articulation_points(graph))
    bridge_nodes = {node for edge in nx.bridges(graph) for node in edge}
    ranked_regions = []
    for region in _compact_physical_regions(env, env.logical_qubits, limit=max(4, limit * 2)):
        subgraph = graph.subgraph(region)
        diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) else env.physical_qubits
        score = (
            4 * len(set(region) & articulation)
            + 2 * len(set(region) & bridge_nodes)
            + diameter
            - 0.25 * subgraph.number_of_edges()
        )
        ranked_regions.append((score, tuple(region), region))
    for _, _, region in sorted(ranked_regions)[:limit]:
        assignment = _embed_component(list(range(env.logical_qubits)), region, env.topology, weights)
        add([assignment[logical] for logical in range(env.logical_qubits)])
    return layouts

def _hub_walk_layout(env, base: list[int]) -> list[int] | None:
    """Place a sequential hub and its leaves on a physical walk in visit order."""
    hub = getattr(env, "hub_logical", -1)
    if hub < 0:
        return None
    leaf_order = []
    seen = {hub}
    for a, b in env.gates:
        if hub not in (a, b):
            continue
        leaf = b if a == hub else a
        if leaf not in seen:
            seen.add(leaf)
            leaf_order.append(leaf)
    logical_order = [hub, *leaf_order, *[q for q in range(env.logical_qubits) if q not in seen]]

    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    starts = sorted(
        range(env.physical_qubits),
        key=lambda node: (graph.degree[node], sum(env.topology.distance(node, other) for other in graph), node),
    )[:16]
    best_walk: list[int] = []
    for start in starts:
        walk = [start]
        unused = set(range(env.physical_qubits)) - {start}
        while unused and len(walk) < env.logical_qubits:
            choices = [neighbor for neighbor in graph.neighbors(walk[-1]) if neighbor in unused]
            if not choices:
                break
            nxt = min(
                choices,
                key=lambda node: (
                    sum(neighbor in unused for neighbor in graph.neighbors(node)),
                    sum(env.topology.distance(node, other) for other in unused),
                    node,
                ),
            )
            walk.append(nxt)
            unused.remove(nxt)
        if len(walk) > len(best_walk):
            best_walk = walk
        if len(best_walk) >= env.logical_qubits:
            break
    if len(best_walk) < env.logical_qubits:
        return None
    candidate = list(base)
    for logical, physical in zip(logical_order, best_walk):
        candidate[logical] = physical
    return candidate if len(set(candidate)) == len(candidate) else None


def _small_hub_layouts(env, base: list[int], limit: int = 24) -> list[list[int]]:
    """Enumerate a bounded set of compact BFS embeddings for small hub circuits."""
    hub = getattr(env, "hub_logical", -1)
    if hub < 0:
        return []
    graph = nx.Graph()
    graph.add_nodes_from(range(env.physical_qubits))
    graph.add_edges_from(env.edges)
    centers = sorted(
        graph,
        key=lambda node: (
            -graph.degree[node],
            sum(env.topology.distance(node, other) for other in graph),
            node,
        ),
    )[:8]
    visits = []
    seen = {hub}
    counts = Counter()
    for a, b in env.gates:
        if hub not in (a, b):
            continue
        leaf = b if a == hub else a
        counts[leaf] += 1
        if leaf not in seen:
            seen.add(leaf)
            visits.append(leaf)
    orders = (
        visits,
        sorted(visits, key=lambda leaf: (-counts[leaf], visits.index(leaf))),
        list(reversed(visits)),
    )
    layouts: list[list[int]] = []
    layout_seen = set()
    for center in centers:
        physical_order = list(nx.bfs_tree(graph, center))
        if len(physical_order) < env.logical_qubits:
            continue
        for leaves in orders:
            logical_order = [hub, *leaves, *[q for q in range(env.logical_qubits) if q not in seen]]
            candidate = list(base)
            for logical, physical in zip(logical_order, physical_order):
                candidate[logical] = physical
            key = tuple(candidate)
            if len(set(candidate)) == len(candidate) and key not in layout_seen:
                layout_seen.add(key)
                layouts.append(candidate)
                if len(layouts) >= limit:
                    return layouts
    return layouts


def _paired_hub_walk_layout(env, base: list[int], hub_layout: list[int] | None) -> list[int] | None:
    """Interleave paired leaves along the same physical walk used by the hub layout."""
    hub = getattr(env, "hub_logical", -1)
    if hub < 0 or hub_layout is None:
        return None

    hub_leaves = []
    hub_seen = {hub}
    for a, b in env.gates:
        if hub not in (a, b):
            continue
        leaf = b if a == hub else a
        if leaf not in hub_seen:
            hub_seen.add(leaf)
            hub_leaves.append(leaf)
    source_order = [hub, *hub_leaves, *[q for q in range(env.logical_qubits) if q not in hub_seen]]
    physical_walk = [hub_layout[logical] for logical in source_order]

    module_order = []
    placed = {hub}
    for a, b in env.gates:
        if hub in (a, b) or a in placed or b in placed:
            continue
        placed.update((a, b))
        module_order.extend((a, b))
    if not module_order:
        return None
    logical_order = [hub, *module_order, *[q for q in range(env.logical_qubits) if q not in placed]]
    candidate = list(base)
    for logical, physical in zip(logical_order, physical_walk):
        candidate[logical] = physical
    return candidate if len(set(candidate)) == len(candidate) else None


def _rank0_metrics(env, initial: list[int]) -> object | None:
    """Route action 0 from one fixed layout and return its SWAP/depth baseline."""
    policy = copy.copy(env.policy)
    policy.decay_state = []
    ctx = RoutingContext(
        RoutingCircuit(env.logical_qubits, env.gates),
        env.topology,
        RoutingLayout(env.logical_qubits, env.physical_qubits, initial),
    )
    layers = [0] * env.physical_qubits
    remaining = len(env.gates)
    swaps = 0
    default_max_swaps = max(10_000, len(env.gates) * 20)
    max_swaps = min(default_max_swaps, max(0, int(getattr(env, "max_swaps", default_max_swaps))))
    while remaining:
        executed_ids = execute_routable_gates(ctx, layers)
        remaining -= len(executed_ids)
        if executed_ids:
            continue
        swap = policy.choose_best_swap(ctx)
        if swap is None or swaps >= max_swaps:
            return None
        ctx.layout().apply_swap(*swap)
        add_two_qubit_depth(layers, *swap)
        ctx._last_swap = swap
        ctx._swaps_since_progress += 1
        swaps += 1
    return RouteMetrics(swaps, max(layers, default=0))

def _rank0_trace(
    env, initial: list[int]
) -> tuple[object, tuple[tuple[str, object], ...], tuple[int, ...], tuple[int, ...]] | None:
    """Route LightSABRE rank 0 from one layout and retain an exportable trace."""
    policy = copy.copy(env.policy)
    policy.decay_state = []
    ctx = RoutingContext(
        RoutingCircuit(env.logical_qubits, env.gates),
        env.topology,
        RoutingLayout(env.logical_qubits, env.physical_qubits, initial),
    )
    layers = [0] * env.physical_qubits
    events: list[tuple[str, object]] = []
    remaining = len(env.gates)
    swaps = 0
    default_max_swaps = max(10_000, len(env.gates) * 20)
    max_swaps = min(default_max_swaps, max(0, int(getattr(env, "max_swaps", default_max_swaps))))
    while remaining:
        executed_ids = execute_routable_gates(ctx, layers)
        if executed_ids:
            events.extend(("gate", node_id) for node_id in executed_ids)
            remaining -= len(executed_ids)
            continue
        swap = policy.choose_best_swap(ctx)
        if swap is None or swaps >= max_swaps:
            return None
        ctx.layout().apply_swap(*swap)
        add_two_qubit_depth(layers, *swap)
        ctx._last_swap = swap
        ctx._swaps_since_progress += 1
        events.append(("swap", swap))
        swaps += 1
    return (
        RouteMetrics(swaps, max(layers, default=0)),
        tuple(events),
        tuple(initial),
        tuple(ctx.layout().logical_to_physical),
    )

def _beam_trace(
    env,
    initial: list[int],
    args: argparse.Namespace,
) -> tuple[object, tuple[tuple[str, object], ...], tuple[int, ...], tuple[int, ...], int, int] | None:
    """Route with model-free Beam search over LightSABRE-ranked SWAP candidates."""
    env.reset(seed=args.seed, options={"initial_mapping": initial})
    terminated = truncated = False
    final_info = {}
    beam_calls = 0
    beam_deviations = 0
    max_beam_calls = max(0, int(getattr(env, "max_beam_calls", 0)))
    swaps_since_beam = max(1, args.beam_interval)
    deadline = getattr(args, "_extra_deadline", None)
    while not (terminated or truncated):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        action = 0
        if (
            env.candidates
            and args.beam_width > 1
            and swaps_since_beam >= max(1, args.beam_interval)
            and beam_calls < max_beam_calls
            and env.no_improvement_beams < env.max_no_improvement_beams
        ):
            action = model_free_beam_action(
                env,
                0,
                args.beam_width,
                args.beam_depth,
                args.beam_branch,
                args.objective,
                args.depth_weight,
            )
            beam_calls += 1
            beam_deviations += int(action != 0)
            swaps_since_beam = 0
        _, _, terminated, truncated, final_info = env.step(action)
        swaps_since_beam += 1
    if not terminated:
        return None
    assert env.ctx is not None
    return (
        RouteMetrics(final_info["swap_count"], final_info["depth"]),
        tuple(env.route_events),
        tuple(initial),
        tuple(env.ctx.layout().logical_to_physical),
        beam_calls,
        beam_deviations,
    )

def _rank0_final_layout(env, initial: list[int], gates: list[tuple[int, int]]) -> list[int] | None:
    """Return the final mapping of one deterministic rank-0 routing pass."""
    policy = copy.copy(env.policy)
    policy.decay_state = []
    circuit = RoutingCircuit(env.logical_qubits, gates)
    ctx = RoutingContext(
        circuit,
        env.topology,
        RoutingLayout(env.logical_qubits, env.physical_qubits, initial),
    )
    layers = [0] * env.physical_qubits
    remaining = len(gates)
    swaps = 0
    max_swaps = max(10_000, len(gates) * 20)
    while remaining:
        executed_ids = execute_routable_gates(ctx, layers)
        remaining -= len(executed_ids)
        if executed_ids:
            continue
        swap = policy.choose_best_swap(ctx)
        if swap is None or swaps >= max_swaps:
            return None
        ctx.layout().apply_swap(*swap)
        add_two_qubit_depth(layers, *swap)
        ctx._last_swap = swap
        ctx._swaps_since_progress += 1
        swaps += 1
    return list(ctx.layout().logical_to_physical)


def _star_walk_route(env, initial: list[int]) -> tuple[object, tuple[tuple[str, object], ...], tuple[int, ...]] | None:
    """Route a pure star by moving its hub along shortest paths to each leaf in order."""
    hub = getattr(env, "hub_logical", -1)
    if hub < 0:
        return None
    ctx = RoutingContext(
        env.circuit,
        env.topology,
        RoutingLayout(env.logical_qubits, env.physical_qubits, initial),
    )
    layers = [0] * env.physical_qubits
    events: list[tuple[str, object]] = []
    swaps = 0
    remaining = len(env.gates)
    while remaining:
        executed_ids = execute_routable_gates(ctx, layers)
        if executed_ids:
            events.extend(("gate", node_id) for node_id in executed_ids)
            remaining -= len(executed_ids)
            continue
        next_node = min(ctx.front_ids, default=None)
        if next_node is None:
            return None
        pair = env.gates[next_node]
        if hub not in pair:
            return None
        leaf = pair[0] if pair[1] == hub else pair[1]
        hub_physical = ctx.layout().physical_of_logical(hub)
        leaf_physical = ctx.layout().physical_of_logical(leaf)
        swap = min(
            (
                tuple(sorted((hub_physical, neighbor)))
                for neighbor in env.topology.neighbors(hub_physical)
                if env.topology.distance(neighbor, leaf_physical)
                < env.topology.distance(hub_physical, leaf_physical)
            ),
            key=lambda edge: (env.topology.distance(edge[1] if edge[0] == hub_physical else edge[0], leaf_physical), edge),
            default=None,
        )
        if swap is None:
            return None
        ctx.layout().apply_swap(*swap)
        add_two_qubit_depth(layers, *swap)
        events.append(("swap", swap))
        swaps += 1
    metrics = RouteMetrics(swaps, max(layers, default=0))
    return metrics, tuple(events), tuple(ctx.layout().logical_to_physical)


def _reverse_rank0_layout(env, initial: list[int]) -> list[int] | None:
    """Use one reverse rank-0 pass to turn a structural seed into a forward layout."""
    return _rank0_final_layout(env, initial, list(reversed(env.gates)))


def generate_structure_aware_layouts(
    env,
    count: int,
    seed: int,
    max_perturbations: int,
) -> list[list[int]]:
    """Generate baseline, profile-specific, spectral and graph-embedded layouts."""
    base = env.policy.choose_best_initial_layout(
        RoutingContext(env.circuit, env.topology), random.Random(seed)
    )
    layout_limit = max(1, int(count)) + max(0, int(max_perturbations))
    layouts = [base]
    seen = {tuple(base)}
    perturbations = max(0, int(max_perturbations))
    if perturbations and env.logical_qubits >= 2:
        loads = Counter(qubit for gate in env.gates for qubit in gate)
        influential = [q for q, _ in loads.most_common(min(env.logical_qubits, 8))]
        for offset in range(perturbations):
            if len(influential) < 2:
                break
            candidate = list(base)
            a = influential[offset % len(influential)]
            b = influential[(offset + 1) % len(influential)]
            candidate[a], candidate[b] = candidate[b], candidate[a]
            key = tuple(candidate)
            if key not in seen:
                seen.add(key)
                layouts.append(candidate)

    def add(candidate: list[int] | None) -> None:
        if candidate is None or len(layouts) >= layout_limit:
            return
        key = tuple(candidate)
        if len(set(candidate)) == len(candidate) and key not in seen:
            seen.add(key)
            layouts.append(candidate)

    features = circuit_features((env.logical_qubits, env.gates))
    profile = getattr(env, "routing_profile", "local")
    pattern = getattr(env, "circuit_pattern", classify_circuit_pattern(features))
    specializations = _medium_specializations(features, profile, env.physical_qubits)
    paired_hub = _is_paired_hub(features)
    hub_layout = _hub_walk_layout(env, base) if profile == "hub_sequence" or paired_hub else None
    if profile == "hub_sequence":
        add(hub_layout)
    if getattr(env, "topology_profile", "") == "ring_like":
        for candidate in _ring_layouts(env, base, limit=6):
            add(candidate)

    uniform_layout = _spectral_layout(env, base, seed, temporal=False)
    temporal_layout = _spectral_layout(env, base, seed, temporal=True)
    add(uniform_layout)
    add(temporal_layout)

    components = env.circuit.logical_interaction_components()
    used = {logical for component in components for logical in component}
    components.extend([[logical] for logical in range(env.logical_qubits) if logical not in used])
    for weight_profile in ("early", "repeat", "front", "unique", "uniform"):
        weights = _interaction_weights(env.gates, weight_profile)
        candidate = list(base)
        for component in components:
            physicals = [base[logical] for logical in component]
            assignment = _embed_component(component, physicals, env.topology, weights)
            for logical, physical in assignment.items():
                candidate[logical] = physical
        add(candidate)
        if len(layouts) >= max(1, count):
            break

    medium_extras = []
    if specializations["repeat_temporal"]:
        medium_extras.append(_weighted_embedding_layout(
            env, base, _medium_interaction_weights(env.gates, "repeat_temporal")
        ))
    if specializations["long_temporal"]:
        medium_extras.append(_weighted_embedding_layout(
            env, base, _medium_interaction_weights(env.gates, "long_temporal")
        ))
    if specializations["hub_refine"]:
        medium_extras.extend((
            _weighted_embedding_layout(env, base, _interaction_weights(env.gates, "early")),
            _weighted_embedding_layout(env, base, _interaction_weights(env.gates, "repeat")),
        ))
    if specializations["dense_refine"]:
        medium_extras.extend((
            _weighted_embedding_layout(env, base, _medium_interaction_weights(env.gates, "long_temporal")),
            _spectral_layout(env, base, seed, temporal=True),
        ))
    for candidate in medium_extras:
        if candidate is not None and tuple(candidate) not in seen:
            seen.add(tuple(candidate))
            layouts.append(candidate)

    active_qubits = int(features["active_qubits"])
    refine_star = _is_star_hub(features) and active_qubits >= 64
    refine_paired = (
        (paired_hub or profile == "hub_sequence")
        and not _is_star_hub(features)
        and (active_qubits <= 67 or active_qubits >= 128)
    )
    refine_symmetric = _is_global_symmetric(features)
    refine = refine_star or refine_paired or refine_symmetric
    seed_layout = temporal_layout if refine_symmetric else hub_layout or temporal_layout
    refined = _reverse_rank0_layout(env, seed_layout or base) if refine and count >= 6 else None
    for extra in (
        refined,
        _paired_hub_walk_layout(env, base, hub_layout) if refine_paired else None,
    ):
        if extra is not None and tuple(extra) not in seen:
            seen.add(tuple(extra))
            layouts.append(extra)

    if refine_symmetric and refined is not None:
        forward_final = _rank0_final_layout(env, refined, env.gates)
        second_refined = _reverse_rank0_layout(env, forward_final or refined)
        if second_refined is not None and tuple(second_refined) not in seen:
            seen.add(tuple(second_refined))
            layouts.append(second_refined)
    if specializations["symmetric_refine"]:
        medium_refined = _reverse_rank0_layout(env, uniform_layout or base)
        medium_forward = None if medium_refined is None else _rank0_final_layout(
            env, medium_refined, env.gates
        )
        medium_second = _reverse_rank0_layout(env, medium_forward or medium_refined) \
            if medium_refined is not None else None
        for candidate in (medium_refined, medium_second):
            if candidate is not None and tuple(candidate) not in seen:
                seen.add(tuple(candidate))
                layouts.append(candidate)

    # Preserve every established candidate and place new ideas in a separate,
    # bounded pool so they cannot evict an established layout.
    if pattern in {"bv_like", "cc_like"}:
        for candidate in _hub_topology_layouts(env, base, pattern, limit=6):
            if candidate is not None and tuple(candidate) not in seen:
                seen.add(tuple(candidate))
                layouts.append(candidate)
    elif pattern == "qft_like" and not (getattr(env, "topology_profile", "grid_like") == "grid_like" and env.logical_qubits > 49):
        for candidate in _qft_topology_layouts(env, base, limit=8):
            if candidate is not None and tuple(candidate) not in seen:
                seen.add(tuple(candidate))
                layouts.append(candidate)
    for candidate in _topology_profile_layouts(env, base, profile, limit=4):
        if candidate is not None and tuple(candidate) not in seen:
            seen.add(tuple(candidate))
            layouts.append(candidate)

    env.sage_core_layout_count = len(layouts)
    extras: list[list[int]] = []
    weights = _interaction_weights(env.gates, "uniform")

    def add_extra(candidate: list[int] | None) -> None:
        if candidate is None:
            return
        key = tuple(candidate)
        if len(set(candidate)) == len(candidate) and key not in seen:
            seen.add(key)
            extras.append(candidate)

    active_qubits = int(features["active_qubits"])
    if (
        profile == "hub_sequence"
        and active_qubits <= 80
        and int(features["two_qubit_gates"]) <= 200
    ):
        for candidate in _small_hub_layouts(env, base, limit=2):
            add_extra(candidate)
    # Heavy-hex additions are appended after the established candidate pool.
    if getattr(env, "topology_profile", "") == "heavy_hex_like":
        graph = nx.Graph()
        graph.add_nodes_from(range(env.physical_qubits))
        graph.add_edges_from(env.edges)
        centrality = nx.closeness_centrality(graph)
        centers = sorted(graph, key=lambda node: (-centrality[node], -graph.degree[node], node))[:3]
        for center in centers:
            region = {center}
            frontier = set(graph.neighbors(center))
            while len(region) < env.logical_qubits and frontier:
                node = min(frontier, key=lambda item: (-graph.degree[item], sum(env.topology.distance(item, other) for other in region), item))
                region.add(node)
                frontier.update(graph.neighbors(node))
                frontier.difference_update(region)
            if len(region) == env.logical_qubits:
                assignment = _embed_component(list(range(env.logical_qubits)), sorted(region), env.topology, weights)
                candidate = list(base)
                for logical, physical in assignment.items():
                    candidate[logical] = physical
                add_extra(candidate)
    if profile == "symmetric":
        for phase in (0.10, 0.50, 0.90):
            add_extra(_weighted_embedding_layout(
                env, base, _windowed_interaction_weights(env.gates, phase, 0.28)
            ))
    if (
        profile in {"local", "dense_temporal"}
        and int(features["two_qubit_gates"]) >= 200
        and float(features["temporal_locality"]) >= 0.45
        and float(features["consecutive_overlap_ratio"]) >= 0.25
        and float(features["interaction_density"]) >= 0.01
    ):
        add_extra(_weighted_embedding_layout(env, base, _front_exponential_weights(env.gates)))
    layouts.extend(extras)
    return layouts


def _medium_sabre_refined_layouts(
    env, layouts: list[list[int]], lookaheads: tuple[int, ...] = (8, 20, 32, 48, 64)
) -> list[tuple[list[int], int]]:
    """Return the best SABRE-style refined seed for each Medium lookahead."""
    if env.physical_qubits > 49 or env.logical_qubits > 49:
        return []
    refined = []
    original_lookahead = env.policy.lookahead_size
    try:
        for lookahead in lookaheads:
            env.policy.lookahead_size = lookahead
            candidates = []
            seen = set()
            for seed in layouts:
                current = seed
                for _ in range(3):
                    reverse = _reverse_rank0_layout(env, current)
                    if reverse is None:
                        break
                    key = tuple(reverse)
                    if key not in seen:
                        seen.add(key)
                        metrics = _rank0_metrics(env, reverse)
                        if metrics is not None:
                            candidates.append((metric_key(metrics, "swap", 0.0), key, reverse))
                    forward = _rank0_final_layout(env, reverse, env.gates)
                    if forward is None:
                        break
                    current = forward
            if candidates:
                refined.append((min(candidates, key=lambda item: (item[0], item[1]))[2], lookahead))
    finally:
        env.policy.lookahead_size = original_lookahead
    return refined

def classify_case(case: LoadedCase) -> Classification:
    """Combine name-independent circuit and physical-topology classifications."""
    classification = classify_circuit(case.circuit)
    physical_qubits, edges = case.topology
    return replace(
        classification,
        topology_profile=classify_topology(physical_qubits, edges).profile,
    )


def write_analysis(
    train_cases: list[LoadedCase],
    test_cases: list[LoadedCase],
    output_path: Path,
) -> dict[str, dict[str, Classification]]:
    analyses: dict[str, dict[str, Classification]] = {"train": {}, "test": {}}
    serializable: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
    for split, cases in (("train", train_cases), ("test", test_cases)):
        by_name = {case.name: case for case in cases}
        for case in cases:
            classification = classify_case(case)
            if case.name.endswith("_transpiled"):
                original = by_name.get(case.name.removesuffix("_transpiled"))
                if original is not None:
                    stable = classify_case(original)
                    classification = replace(
                        classification,
                        expert=stable.expert,
                        routing_profile=stable.routing_profile,
                    )
            analyses[split][case.name] = classification
            serializable[split].append({
                "case": case.name,
                "qasm_path": str(case.qasm_path),
                "topology": case.topology_name,
                "expert": classification.expert,
                "routing_profile": classification.routing_profile,
                "circuit_pattern": classification.circuit_pattern,
                "topology_profile": classification.topology_profile,
                "interaction_fingerprint": interaction_fingerprint(case.qasm_path),
                "confidence": round(classification.confidence, 6),
                "scores": {key: round(value, 6) for key, value in classification.scores.items()},
                "features": classification.features,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return analyses


class DenseHybridRoutingEnv(HybridRoutingEnv):
    """LightSABRE environment with profile-specific candidates and Beam scoring."""

    def __init__(
        self,
        *args,
        lookahead_size: int = 64,
        baseline_lookahead_size: int = 20,
        routing_profile: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, lookahead_size=baseline_lookahead_size, **kwargs)
        features = circuit_features((self.logical_qubits, self.gates))
        inferred = classify_circuit((self.logical_qubits, self.gates)).routing_profile
        self.routing_profile = routing_profile or inferred
        self.policy.topology_profile = getattr(self, "topology_profile", "unknown")
        if self.routing_profile not in {"local", "dense_temporal", "hub_sequence", "symmetric"}:
            raise ValueError(f"unknown routing profile: {self.routing_profile}")
        self.policy.lookahead_size = lookahead_size
        self.hub_routing = self.routing_profile == "hub_sequence"
        self.hub_logical = int(np.argmax(self.logical_interaction_load)) if self.logical_interaction_load.size else -1
        self.potential_horizon = 64
        self.potential_lambda = 0.85
        route_work = max(1, self.logical_qubits * len(self.gates))
        self.max_beam_calls = min(32, max(3, 2_000_000 // route_work))
        self.max_no_improvement_beams = 8
        self.beam_budget_calls = 0
        self.no_improvement_beams = 0
        self.hub_candidate_index: int | None = None
        self.medium_symmetric = _medium_specializations(
            features, self.routing_profile, self.physical_qubits
        )["symmetric_refine"]
        if self.medium_symmetric:
            self.max_beam_calls = min(self.max_beam_calls, 12)
            self.max_no_improvement_beams = min(self.max_no_improvement_beams, 4)

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.beam_budget_calls = 0
        self.no_improvement_beams = 0
        self.hub_candidate_index = None
        # reset() computed candidates before resetting the index; recover it deterministically.
        self._refresh_hub_candidate_index()
        return result

    def _future_nodes(self, horizon: int | None = None) -> list[tuple[int, int]]:
        assert self.ctx is not None
        limit = self.potential_horizon if horizon is None else horizon
        queue = deque((node_id, 0) for node_id in sorted(self.ctx.front_ids))
        seen = set()
        nodes = []
        while queue and len(nodes) < limit:
            node_id, level = queue.popleft()
            if node_id in seen or self.ctx.done[node_id]:
                continue
            seen.add(node_id)
            nodes.append((node_id, level))
            for successor in sorted(self.circuit.node(node_id).successors()):
                if successor not in seen:
                    queue.append((successor, level + 1))
        return nodes

    def _pair_distance(self, pair: tuple[int, int], swap: tuple[int, int] | None = None) -> int:
        assert self.ctx is not None
        layout = self.ctx.layout()
        physicals = []
        for logical in pair:
            physical = layout.logical_to_physical[logical]
            if swap is not None:
                if physical == swap[0]:
                    physical = swap[1]
                elif physical == swap[1]:
                    physical = swap[0]
            physicals.append(physical)
        return self.topology.distance(*physicals)

    def _hub_queue_cost(self, swap: tuple[int, int] | None = None, horizon: int = 16) -> float:
        if self.hub_logical < 0:
            return 0.0
        total = weight_sum = 0.0
        queue_index = 0
        for node_id, level in self._future_nodes(self.potential_horizon):
            pair = self.gates[node_id]
            if self.hub_logical not in pair:
                continue
            weight = (self.potential_lambda ** level) * (0.90 ** queue_index)
            total += weight * max(0, self._pair_distance(pair, swap) - 1)
            weight_sum += weight
            queue_index += 1
            if queue_index >= horizon:
                break
        return total / max(1e-9, weight_sum)

    def _leaf_swap_penalty(self, swap: tuple[int, int], score_scale: float) -> float:
        assert self.ctx is not None
        logical_a, logical_b = (self.ctx.layout().physical_to_logical[p] for p in swap)
        if logical_a < 0 or logical_b < 0 or self.hub_logical in (logical_a, logical_b):
            return 0.0
        if self._hub_queue_cost(swap) + 1e-9 < self._hub_queue_cost():
            return 0.0
        return 0.25 * score_scale

    def _next_hub_candidate(self) -> tuple[int, int] | None:
        if not self.hub_routing or self.hub_logical < 0 or self.ctx is None:
            return None
        next_pair = next(
            (self.gates[node_id] for node_id, _ in self._future_nodes() if self.hub_logical in self.gates[node_id]),
            None,
        )
        if next_pair is None:
            return None
        leaf = next_pair[0] if next_pair[1] == self.hub_logical else next_pair[1]
        layout = self.ctx.layout()
        hub_physical = layout.logical_to_physical[self.hub_logical]
        leaf_physical = layout.logical_to_physical[leaf]
        candidates = []
        for source, target in ((hub_physical, leaf_physical), (leaf_physical, hub_physical)):
            for neighbor in self.topology.neighbors(source):
                if self.topology.distance(neighbor, target) < self.topology.distance(source, target):
                    candidates.append(tuple(sorted((source, neighbor))))
        return min(set(candidates), key=lambda swap: (self._hub_queue_cost(swap), swap), default=None)

    def _refresh_hub_candidate_index(self) -> None:
        hub_candidate = self._next_hub_candidate()
        self.hub_candidate_index = (
            self.candidates.index(hub_candidate) if hub_candidate in self.candidates else None
        )

    def _ranked_candidates(self):
        ranked = self.policy.rank_swaps(self.ctx) if self.ctx is not None else []
        if self.ctx is None or not ranked:
            self.hub_candidate_index = None
            return [], []
        ranked = ranked[:self.top_k]
        rank0 = ranked[0]
        scale = max(1.0, abs(float(rank0[1])))
        if self.hub_routing:
            adjusted = [rank0]
            adjusted.extend(sorted(
                (
                    (swap, float(score) + self._leaf_swap_penalty(swap, scale))
                    for swap, score in ranked[1:]
                ),
                key=lambda item: (item[1], item[0]),
            ))
            ranked = adjusted[:self.top_k]
            hub_candidate = self._next_hub_candidate()
            if hub_candidate is not None and hub_candidate not in [swap for swap, _ in ranked]:
                injected = (hub_candidate, float(rank0[1]) + 0.01 * scale)
                if len(ranked) < self.top_k:
                    ranked.append(injected)
                else:
                    ranked[-1] = injected
        self.hub_candidate_index = next(
            (index for index, (swap, _) in enumerate(ranked) if swap == self._next_hub_candidate()),
            None,
        )
        return [swap for swap, _ in ranked], [float(score) for _, score in ranked]

    def _future_pressure(self) -> float:
        assert self.ctx is not None
        front = FrontLayerScores.from_ctx(self.ctx)
        extended = build_extended_set(self.ctx, self.policy.lookahead_size)
        excess = (
            front.total_score(self.topology) - len(front)
            + 0.75 * (extended.total_score(self.topology) - len(extended))
        )
        return excess / max(1.0, len(front) + 0.75 * len(extended))


    def routing_potential(self) -> float:
        """Return a normalized potential where larger values mean easier remaining routing."""
        if self.ctx is None:
            return 0.0
        if self.routing_profile == "local":
            return -self._future_pressure()
        if self.routing_profile == "hub_sequence":
            return -self._hub_queue_cost(horizon=24)

        total = weight_sum = 0.0
        for index, (node_id, level) in enumerate(self._future_nodes()):
            weight = self.potential_lambda ** level
            if self.routing_profile == "symmetric" and index % 8 == 0:
                weight += 0.25
            total += weight * max(0, self._pair_distance(self.gates[node_id]) - 1)
            weight_sum += weight
        return -total / max(1e-9, weight_sum)

def _path_interaction_layout(env, base: list[int]) -> list[int] | None:
    """Embed a logical path directly onto a physical path, yielding zero SWAP when possible."""
    pairs = {tuple(sorted(pair)) for pair in env.gates}
    adjacency = {q: set() for q in range(env.logical_qubits)}
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    active = [q for q, neighbors in adjacency.items() if neighbors]
    if len(active) < 2 or len(pairs) != len(active) - 1 or max(map(lambda q: len(adjacency[q]), active)) > 2:
        return None
    endpoints = [q for q in active if len(adjacency[q]) == 1]
    if len(endpoints) != 2:
        return None
    logical_path = []
    previous = -1
    current = min(endpoints)
    while current >= 0:
        logical_path.append(current)
        nxt = next((q for q in sorted(adjacency[current]) if q != previous), -1)
        previous, current = current, nxt
    if len(logical_path) != len(active):
        return None

    physical_path = None
    for start in sorted(range(env.physical_qubits), key=lambda q: (-len(env.topology.neighbors(q)), q)):
        path = [start]
        used = {start}
        while len(path) < len(logical_path):
            choices = [q for q in env.topology.neighbors(path[-1]) if q not in used]
            if not choices:
                break
            nxt = max(choices, key=lambda q: (len([n for n in env.topology.neighbors(q) if n not in used]), -q))
            path.append(nxt)
            used.add(nxt)
        if len(path) == len(logical_path):
            physical_path = path
            break
    if physical_path is None:
        return None
    candidate = list(base)
    for logical, physical in zip(logical_path, physical_path):
        candidate[logical] = physical
    return candidate if len(set(candidate)) == len(candidate) else None


def _add_path_layout_candidate(result: HybridEvaluation, case: LoadedCase, args: argparse.Namespace) -> HybridEvaluation:
    logical_qubits, gates = case.circuit
    physical_qubits, edges = case.topology
    topology = RoutingTopology(physical_qubits, edges)
    circuit = RoutingCircuit(logical_qubits, gates)
    env = argparse.Namespace(
        logical_qubits=logical_qubits,
        physical_qubits=physical_qubits,
        gates=gates,
        topology=topology,
        policy=CandidatePolicy(lookahead_size=args.lookahead_size),
    )
    base = env.policy.choose_best_initial_layout(RoutingContext(circuit, topology), random.Random(args.seed))
    initial = _path_interaction_layout(env, base)
    if initial is None:
        return result
    candidate = _rank0_trace(env, initial)
    if candidate is None:
        return result
    metrics, events, initial_mapping, final_mapping = candidate
    if result.metrics is not None and metric_key(result.metrics, args.objective, args.depth_weight) <= metric_key(metrics, args.objective, args.depth_weight):
        return result
    return replace(result, metrics=metrics, method="path_embed", layouts_tried=result.layouts_tried + 1, route_events=events, initial_mapping=initial_mapping, final_mapping=final_mapping)

def _extra_search_target(classification: Classification, env) -> bool:
    """Identify structurally hard routes worth bounded layout search."""
    features = classification.features
    gates = int(features["two_qubit_gates"])
    active = int(features["active_qubits"])
    return (
        (classification.routing_profile in {"dense_temporal", "symmetric"} and gates >= 400)
        or (classification.routing_profile == "hub_sequence" and active >= 64)
        or (classification.expert == "arithmetic" and gates >= 1500)
        or (gates >= 5000 and active >= 32)
    )


def _self_improvement_target(
    classification: Classification,
    env,
    result: HybridEvaluation,
) -> int | None:
    """Return a one-SWAP-better target for bounded, structurally hard routes."""
    if result.metrics is None or result.metrics.swap_count == 0:
        return None
    # Keep bounded self-improvement enabled on large-grid QFTs.
    features = classification.features
    front_loaded = _is_front_loaded_repeated(features)
    gates = int(features["two_qubit_gates"])
    active = int(features["active_qubits"])
    if result.metrics.swap_count < 24 and not front_loaded:
        return None
    route_work = active * max(1, gates)
    medium = env.physical_qubits <= 49 and active <= 49
    bounded_mid = active <= 127 and route_work <= 600_000
    bounded_hub = (
        classification.routing_profile == "hub_sequence"
        and active <= 361
        and gates <= 2000
        and route_work <= 650_000
    )
    if gates < 200 or not (medium or bounded_mid or bounded_hub):
        return None
    return max(0, result.metrics.swap_count - 1)


def _adaptive_extra_policy(
    classification: Classification,
    env,
    result: HybridEvaluation,
    reference_target: int | None,
    args: argparse.Namespace,
) -> bool:
    """Enable bounded extra search only for a material reference gap."""
    if result.metrics is None:
        return False
    if reference_target is None:
        reference_target = result.baseline.swap_count
    gap = result.metrics.swap_count - reference_target
    ratio = gap / max(1, reference_target)
    features = circuit_features((env.logical_qubits, env.gates))
    gates = int(features["two_qubit_gates"])
    active = int(features["active_qubits"])
    repeated = float(features["repeated_interaction_ratio"])
    dense = float(features["interaction_density"])
    medium = env.physical_qubits <= 49 and env.logical_qubits <= 49
    topology_profile = getattr(env, "topology_profile", "grid_like")
    # Heavy-hex only: preserve the established Grid behavior exactly.
    if topology_profile == "heavy_hex_like" and gates < 50:
        return False
    tie_breaker = (
        topology_profile == "heavy_hex_like"
        and result.metrics.swap_count == result.baseline.swap_count
        and result.baseline.swap_count > 0
        and (gates >= 50 or active >= 10)
    )
    if tie_breaker:
        args._tie_breaker = True
        args._adaptive_reason = f"tie_breaker,gates={gates},active={active}"
    min_gap = max(2 if medium else 8, int(getattr(args, "medium_adaptive_min_gap" if medium else "adaptive_min_gap", 2 if medium else 8)))
    if topology_profile == "heavy_hex_like":
        min_gap = max(1, min_gap // 2)
    min_ratio = float(getattr(args, "medium_adaptive_min_ratio" if medium else "adaptive_min_ratio", 0.03 if medium else 0.05))
    if topology_profile == "heavy_hex_like":
        min_ratio = min(min_ratio, 0.02)
    self_improve = bool(getattr(args, "_self_improve", False))
    if not self_improve and not tie_breaker and (gap < min_gap or ratio < min_ratio):
        return False

    # Bounded tiers: the largest circuits never receive evolutionary or full
    # rollout search because one complete probe is already expensive.
    if medium:
        layouts, search, evolution, timeout = 4, 20, 64, 15
    elif gates >= 10000 or active >= 300:
        layouts, search, evolution, timeout = 1, 4, 0, 5
    elif gates >= 5000 or active >= 180:
        layouts, search, evolution, timeout = 2, 8, 0, 25
    elif gates >= 1500 or active >= 90:
        layouts, search, evolution, timeout = 2, 12, 16, 30
    else:
        layouts, search, evolution, timeout = 3, 16, 32, 30

    if classification.circuit_pattern == "qft_like":
        layouts = max(layouts, 5 if active <= 80 else 3)
        search = max(search, 32 if active <= 80 else 12)
        evolution = max(evolution, 64 if active <= 80 else 0)
        timeout = max(timeout, 30 if active <= 80 else 20)
    elif classification.circuit_pattern in {"bv_like", "cc_like"}:
        layouts = max(layouts, 4)
        search = max(search, 12)
        timeout = max(timeout, 12)
    if topology_profile in {"ring_like", "heavy_hex_like", "irregular_sparse"}:
        layouts = min(max(layouts + 1, 4), 6)
        search = min(max(search, 16), 40)

    if repeated >= 0.80 and gates >= 1500:
        layouts = min(layouts + 1, 5)
    if dense >= 0.25 and classification.routing_profile in {"symmetric", "dense_temporal"}:
        search = min(search + 12, 48)
    if ratio >= 0.20:
        layouts = min(layouts + 1, 6)
    if _is_front_loaded_repeated(features):
        layouts = max(layouts, 5)
        search = max(search, 28)
        evolution = max(evolution, 64)
        timeout = max(timeout, 30)

    if topology_profile == "heavy_hex_like":
        layouts = min(layouts + 2, 8)
        search = min(search + 16, 56)
        evolution = min(max(evolution, 24) + 16, 96)
    # Only Heavy-hex receives the bounded budget increase. Grid keeps the
    # pre-XV cap so its existing benchmark remains reproducible.
    if topology_profile == "heavy_hex_like":
        args.targeted_layouts = min(max(int(args.targeted_layouts), layouts), 8)
        args.layout_search_budget = min(max(int(args.layout_search_budget), search), 56)
        args.evolution_budget = min(max(int(args.evolution_budget), evolution), 96)
        args.layout_search_seeds = min(max(int(args.layout_search_seeds), layouts), 8)
    else:
        args.targeted_layouts = min(int(args.targeted_layouts), layouts)
        args.layout_search_budget = min(int(args.layout_search_budget), search)
        args.evolution_budget = min(int(args.evolution_budget), evolution)
        args.layout_search_seeds = min(int(args.layout_search_seeds), layouts)
    args._extra_deadline = time.perf_counter() + min(float(args.adaptive_timeout_s), timeout)
    mode = "self" if self_improve else "reference"
    prefix = "tie_breaker," if tie_breaker else ""
    args._adaptive_reason = f"{prefix}mode={mode},target={reference_target},gap={gap},gates={gates},active={active}"
    return True

def _extra_budget_available(args: argparse.Namespace) -> bool:
    deadline = getattr(args, "_extra_deadline", None)
    return deadline is None or time.perf_counter() < deadline


def _add_targeted_layout_candidates(
    result: HybridEvaluation,
    env,
    seed_layouts: list[list[int]],
    args: argparse.Namespace,
    external_target: int | None,
    targeted_layouts: list[list[int]] | None = None,
) -> HybridEvaluation:
    """Try SAGE-only phase-aware layouts when the current route trails a reference."""
    if result.metrics is None:
        needs_search = True
    elif external_target is not None:
        tie_breaker = bool(getattr(args, "_tie_breaker", False))
        needs_search = result.metrics.swap_count >= external_target if tie_breaker else result.metrics.swap_count > external_target
    else:
        features = circuit_features((env.logical_qubits, env.gates))
        needs_search = _extra_search_target(classify_circuit((env.logical_qubits, env.gates)), env) and int(features["two_qubit_gates"]) >= 400
    if not needs_search:
        return result

    candidates = targeted_layouts if targeted_layouts is not None else generate_targeted_layouts(env, seed_layouts)
    limit = max(1, int(args.targeted_layouts))
    lookaheads = (8, 12, 20, 32, 48, 64) if env.physical_qubits <= 49 else (20, 32, 64)
    best = result.metrics
    best_payload = None
    tried = 0
    original_lookahead = env.policy.lookahead_size
    try:
        for initial in candidates[:limit]:
            if not _extra_budget_available(args):
                break
            for lookahead in lookaheads:
                if not _extra_budget_available(args):
                    break
                env.policy.lookahead_size = lookahead
                candidate = _rank0_trace(env, initial)
                tried += 1
                if candidate is None:
                    continue
                metrics, events, initial_mapping, final_mapping = candidate
                if best is None or metric_key(metrics, args.objective, args.depth_weight) < metric_key(best, args.objective, args.depth_weight):
                    best = metrics
                    best_payload = (events, initial_mapping, final_mapping)
                if external_target is not None and best is not None and best.swap_count <= external_target:
                    break
            if external_target is not None and best is not None and best.swap_count <= external_target:
                break
    finally:
        env.policy.lookahead_size = original_lookahead
    if best_payload is None:
        return result
    events, initial_mapping, final_mapping = best_payload
    return replace(
        result,
        metrics=best,
        method="targeted_layout+rank0",
        layouts_tried=result.layouts_tried + tried,
        route_events=events,
        initial_mapping=initial_mapping,
        final_mapping=final_mapping,
    )

def _add_routing_scored_layout_candidate(
    result: HybridEvaluation,
    env,
    seed_layouts: list[list[int]],
    args: argparse.Namespace,
    external_target: int | None,
    targeted_layouts: list[list[int]] | None = None,
) -> HybridEvaluation:
    """Use only SAGE routing to score and improve mappings for trailing circuits."""
    tie_breaker = bool(getattr(args, "_tie_breaker", False))
    if result.metrics is not None and external_target is not None:
        if result.metrics.swap_count < external_target or (result.metrics.swap_count == external_target and not tie_breaker):
            return result
    if external_target is None and not _extra_search_target(classify_circuit((env.logical_qubits, env.gates)), env):
        return result
    best = result.metrics
    best_payload = None
    evaluated = 0
    seeds = list(seed_layouts)
    candidates = targeted_layouts if targeted_layouts is not None else generate_targeted_layouts(env, seed_layouts)
    seeds.extend(candidates[: max(1, int(args.layout_search_seeds))])
    seen = set()
    for initial in seeds:
        if not _extra_budget_available(args):
            break
        key = tuple(initial)
        if key in seen:
            continue
        seen.add(key)
        candidate, count = _search_layout_by_routing(env, initial, args, external_target)
        evaluated += count
        if candidate is None:
            continue
        metrics, events, initial_mapping, final_mapping = candidate
        if best is None or metric_key(metrics, args.objective, args.depth_weight) < metric_key(best, args.objective, args.depth_weight):
            best = metrics
            best_payload = (events, initial_mapping, final_mapping)
        if external_target is not None and best is not None and best.swap_count <= external_target:
            break
    if best_payload is None:
        return result
    events, initial_mapping, final_mapping = best_payload
    return replace(
        result,
        metrics=best,
        method="routing_scored_layout",
        layouts_tried=result.layouts_tried + evaluated,
        route_events=events,
        initial_mapping=initial_mapping,
        final_mapping=final_mapping,
    )

def _add_evolutionary_layout_candidate(
    result: HybridEvaluation,
    env,
    seed_layouts: list[list[int]],
    args: argparse.Namespace,
    external_target: int | None,
    targeted_layouts: list[list[int]] | None = None,
) -> HybridEvaluation:
    """Use extra compute only for circuits that still trail a known reference."""
    if (
        external_target is None
        or result.metrics is None
        or result.metrics.swap_count <= external_target
        or int(args.evolution_budget) <= 0
        or bool(getattr(args, "_tie_breaker", False))
    ):
        return result
    seeds = []
    if result.initial_mapping:
        seeds.append(list(result.initial_mapping))
    seeds.extend(seed_layouts)
    if targeted_layouts:
        seeds.extend(targeted_layouts)
    candidate, evaluated = _evolve_layout_by_routing(env, seeds, args, external_target)
    if candidate is None:
        return result
    metrics, events, initial_mapping, final_mapping = candidate
    if metric_key(result.metrics, args.objective, args.depth_weight) <= metric_key(
        metrics, args.objective, args.depth_weight
    ):
        return result
    return replace(
        result,
        metrics=metrics,
        method="evolved_own_layout",
        layouts_tried=result.layouts_tried + evaluated,
        route_events=events,
        initial_mapping=initial_mapping,
        final_mapping=final_mapping,
    )

def clone_dense_env(env):
    return clone_env(env)


def _beam_state_key(
    env,
    terminated: bool,
    objective: str,
    depth_weight: float,
) -> tuple[float, ...]:
    if terminated:
        depth = float(max(env.layers, default=0))
        if objective == "depth":
            return 0.0, depth, float(env.swap_count)
        if objective == "swap-depth":
            return 0.0, float(env.swap_count) + depth_weight * depth, float(env.swap_count), depth
        return 0.0, float(env.swap_count), depth
    depth = float(max(env.layers, default=0))
    potential = float(env.routing_potential()) if hasattr(env, "routing_potential") else 0.0
    if objective == "depth":
        return 1.0, depth, -potential, float(env.swap_count), float(-env.executed_gates)
    projected_swaps = float(env.swap_count) - potential
    if objective == "swap-depth":
        return 1.0, projected_swaps + depth_weight * depth, projected_swaps, depth, float(-env.executed_gates)
    return 1.0, projected_swaps, depth, float(-env.executed_gates)


def model_free_beam_action(
    env,
    root_action: int,
    width: int,
    depth: int,
    branch: int,
    objective: str = "swap",
    depth_weight: float = 0.05,
) -> int:
    before = float(env.routing_potential()) if hasattr(env, "routing_potential") else 0.0
    frontier = [(clone_dense_env(env), None, False)]
    for level in range(max(1, depth)):
        expanded = []
        for state, first_action, terminated in frontier:
            if terminated:
                expanded.append((state, first_action, True))
                continue
            if not state.candidates:
                continue
            preferred = root_action
            hub_index = getattr(state, "hub_candidate_index", None)
            order = list(dict.fromkeys([preferred, 0, hub_index, *range(len(state.candidates))]))
            state_expanded = 0
            for action in order:
                if action is None or action >= len(state.candidates):
                    continue
                child = clone_dense_env(state)
                _, _, child_terminated, child_truncated, _ = child.step(action)
                if not child_truncated or child_terminated:
                    expanded.append((child, action if first_action is None else first_action, child_terminated))
                    state_expanded += 1
                if state_expanded >= max(1, branch):
                    break
        if not expanded:
            return 0 if root_action >= len(env.candidates) else root_action
        expanded.sort(key=lambda item: _beam_state_key(item[0], item[2], objective, depth_weight))
        frontier = expanded[:max(1, width)]
    best = min(frontier, key=lambda item: _beam_state_key(item[0], item[2], objective, depth_weight))
    after = float(best[0].routing_potential()) if hasattr(best[0], "routing_potential") else before
    env.beam_budget_calls = getattr(env, "beam_budget_calls", 0) + 1
    env.no_improvement_beams = 0 if after > before + 1e-9 else getattr(env, "no_improvement_beams", 0) + 1
    return int(best[1] if best[1] is not None else root_action)


def _add_adaptive_targeted_beam_candidate(
    result: HybridEvaluation,
    env,
    args: argparse.Namespace,
    external_target: int | None,
    targeted_layouts: list[list[int]] | None,
) -> HybridEvaluation:
    """Run a small Beam portfolio on adaptive layouts for genuinely trailing routes."""
    if (
        external_target is None
        or result.metrics is None
        or result.metrics.swap_count < external_target
        or (result.metrics.swap_count == external_target and not getattr(args, "_tie_breaker", False))
        or not targeted_layouts
        or not _extra_budget_available(args)
    ):
        return result
    features = circuit_features((env.logical_qubits, env.gates))
    gates = int(features["two_qubit_gates"])
    active = int(features["active_qubits"])
    # This is a route-quality portfolio, not a global Beam expansion.
    limit = 2 if gates >= 5000 or active >= 180 else 3
    beam_args = copy.copy(args)
    beam_args.beam_width = 4
    beam_args.beam_depth = 3 if (env.physical_qubits <= 49 and active <= 49) else min(max(2, int(args.beam_depth)), 3)
    beam_args.beam_branch = 3 if (env.physical_qubits <= 49 and active <= 49) else min(max(2, int(args.beam_branch)), 3)
    beam_args.beam_interval = 4 if (env.physical_qubits <= 49 and active <= 49) else min(max(4, int(args.beam_interval)), 8)
    best = result.metrics
    best_payload = None
    calls = 0
    deviations = 0
    routes_tried = 0
    original_lookahead = env.policy.lookahead_size
    original_max_beams = env.max_beam_calls
    original_no_improvement = env.max_no_improvement_beams
    env.max_beam_calls = min(original_max_beams, 6 if limit == 2 else 10)
    env.max_no_improvement_beams = min(original_no_improvement, 3)
    beam_layouts = list(targeted_layouts)
    try:
        for initial in beam_layouts[:limit]:
            if not _extra_budget_available(args):
                break
            env.policy.lookahead_size = max(int(args.lookahead_size), int(args.dense_lookahead_size))
            candidate = _beam_trace(env, initial, beam_args)
            if candidate is None:
                continue
            metrics, events, initial_mapping, final_mapping, beam_calls, beam_deviations = candidate
            routes_tried += 1
            calls += beam_calls
            deviations += beam_deviations
            if metric_key(metrics, args.objective, args.depth_weight) < metric_key(best, args.objective, args.depth_weight):
                best = metrics
                best_payload = (events, initial_mapping, final_mapping)
            if best.swap_count <= external_target:
                break
    finally:
        env.policy.lookahead_size = original_lookahead
        env.max_beam_calls = original_max_beams
        env.max_no_improvement_beams = original_no_improvement
    if best_payload is None:
        return replace(
            result,
            beam_calls=result.beam_calls + calls,
            beam_tried=result.beam_tried + calls,
            beam_changed=result.beam_changed + deviations,
            layouts_tried=result.layouts_tried + routes_tried,
        )
    events, initial_mapping, final_mapping = best_payload
    return replace(
        result,
        metrics=best,
        method="adaptive_layout+beam",
        beam_calls=result.beam_calls + calls,
        beam_tried=result.beam_tried + calls,
        beam_changed=result.beam_changed + deviations,
        beam_improved=result.beam_improved + 1,
        layouts_tried=result.layouts_tried + routes_tried,
        route_events=events,
        initial_mapping=initial_mapping,
        final_mapping=final_mapping,
    )


def _add_heavy_hex_center_candidate(result: HybridEvaluation, env, initial_layouts: list[tuple[list[int], int]], args: argparse.Namespace) -> HybridEvaluation:
    """Try a center-biased Heavy-hex candidate without changing baseline ranking."""
    if getattr(env, "topology_profile", "") != "heavy_hex_like" or result.metrics is None:
        return result
    old_bias, old_profile = env.policy.heavy_hex_center_bias, env.policy.topology_profile
    env.policy.topology_profile, env.policy.heavy_hex_center_bias = "heavy_hex_like", 0.1
    best = None; tried = 0
    try:
        for initial, lookahead in initial_layouts[:3]:
            env.policy.lookahead_size = lookahead
            candidate = _rank0_trace(env, initial); tried += 1
            if candidate is not None and (best is None or metric_key(candidate[0], args.objective, args.depth_weight) < metric_key(best[0], args.objective, args.depth_weight)):
                best = candidate
    finally:
        env.policy.heavy_hex_center_bias, env.policy.topology_profile = old_bias, old_profile
    if best is None or metric_key(result.metrics, args.objective, args.depth_weight) <= metric_key(best[0], args.objective, args.depth_weight):
        return result
    metrics, events, initial_mapping, final_mapping = best
    return replace(result, metrics=metrics, method="heavy_hex_center_rank0", layouts_tried=result.layouts_tried + tried, route_events=events, initial_mapping=initial_mapping, final_mapping=final_mapping)
def _add_star_walk_candidate(
    result: HybridEvaluation,
    case: LoadedCase,
    classification: Classification,
    args: argparse.Namespace,
) -> HybridEvaluation:
    topology_profile = classification.topology_profile
    if not _is_star_hub(classification.features) or (
        int(classification.features["active_qubits"]) < 32
        and topology_profile != "heavy_hex_like"
    ):
        return result
    logical_qubits, gates = case.circuit
    physical_qubits, edges = case.topology
    topology = RoutingTopology(physical_qubits, edges)
    circuit = RoutingCircuit(logical_qubits, gates)
    loads = Counter(qubit for gate in gates for qubit in gate)
    env = argparse.Namespace(
        logical_qubits=logical_qubits,
        physical_qubits=physical_qubits,
        gates=gates,
        edges=edges,
        topology=topology,
        circuit=circuit,
        policy=CandidatePolicy(lookahead_size=args.lookahead_size),
        hub_logical=max(range(logical_qubits), key=lambda qubit: (loads[qubit], -qubit)),
    )
    base = env.policy.choose_best_initial_layout(
        RoutingContext(circuit, topology), random.Random(args.seed)
    )
    initial = _hub_walk_layout(env, base)
    candidate = None if initial is None else _star_walk_route(env, initial)
    if candidate is None:
        return result
    metrics, events, final = candidate
    if result.metrics is not None and metric_key(
        result.metrics, args.objective, args.depth_weight
    ) <= metric_key(metrics, args.objective, args.depth_weight):
        return result
    return replace(
        result,
        metrics=metrics,
        rescues=0,
        method="star_walk",
        layouts_tried=result.layouts_tried + 1,
        route_events=events,
        initial_mapping=tuple(initial),
        final_mapping=final,
    )

def _add_heavy_hex_lookahead_rescue(
    result: HybridEvaluation,
    env,
    baseline: RouteMetrics,
    args: argparse.Namespace,
) -> HybridEvaluation:
    """Try three cheap lookaheads only for a tied Heavy-hex route."""
    if (
        getattr(env, "topology_profile", "") != "heavy_hex_like"
        or result.metrics is None
        or not result.initial_mapping
        or result.metrics.swap_count < baseline.swap_count
        or baseline.swap_count <= 0
        or env.logical_qubits > 49
    ):
        return result
    original = env.policy.lookahead_size
    best = result
    tried = 0
    try:
        for lookahead in (8, 20, 40):
            if lookahead == original:
                continue
            env.policy.lookahead_size = lookahead
            candidate = _rank0_trace(env, list(result.initial_mapping))
            tried += 1
            if candidate is None:
                continue
            metrics, events, initial_mapping, final_mapping = candidate
            if metrics.swap_count < best.metrics.swap_count:
                best = replace(
                    best,
                    metrics=metrics,
                    method="heavy_hex_lookahead_rescue",
                    route_events=events,
                    initial_mapping=initial_mapping,
                    final_mapping=final_mapping,
                )
    finally:
        env.policy.lookahead_size = original
    return replace(best, layouts_tried=best.layouts_tried + tried) if tried else best


def route_with_experts(
    test_cases: list[LoadedCase],
    analyses: dict[str, Classification],
    args: argparse.Namespace,
    references: ExternalReferences | None = None,
) -> None:
    evaluation_started = time.perf_counter()
    total_swaps = 0
    total_baseline_swaps = 0
    total_full_swaps = 0
    total_full_baseline_swaps = 0
    full_wins = full_ties = full_losses = 0
    print("\\n=== SAGE XV routing ===")
    print("selection: SAGE XV circuit x topology layouts + bounded Beam/evolution + Qiskit LightSABRE baseline")
    print(
        "circuit\tfingerprint\tpredicted_expert\trouting_profile\tselected_model\tfinal_swaps\tlightsabre_swaps"
        "\toptimization\tfinal_depth\tlightsabre_depth\tmethod\tbeam_tried\tbeam_changed\tbeam_improved"
        "\tlayouts\tvalidated\truntime_s\trouted_qasm"
    )

    cache_arg_key = tuple(sorted(
        (name, repr(value))
        for name, value in vars(args).items()
        if not name.startswith("_") and name not in {
            "dataset_dir", "case_manifest", "analysis_output", "output_dir",
            "split", "eval_only", "analyze_only", "self_check", "no_qasm",
            "mapping_only", "runtime_baseline_s", "reference_results",
        }
    ))

    def canonical_cache_key(case: LoadedCase, classification: Classification, fingerprint: str):
        reference_target = None if references is None else references.best(case.name, fingerprint)
        return (
            case.circuit[0],
            tuple(case.circuit[1]),
            case.topology[0],
            tuple(sorted(tuple(sorted(edge)) for edge in case.topology[1])),
            classification.expert,
            classification.routing_profile,
            classification.circuit_pattern,
            classification.topology_profile,
            reference_target,
            cache_arg_key,
        )

    route_keys = {
        case.name: canonical_cache_key(
            case, analyses[case.name], interaction_fingerprint(case.qasm_path)
        )
        for case in test_cases
    }
    duplicate_keys = {
        key for key, count in Counter(route_keys.values()).items() if count > 1
    }
    route_cache: dict[tuple, tuple[HybridEvaluation, int, int]] = {}
    cache_hits = 0

    def emit_case(
        case: LoadedCase,
        classification: Classification,
        result: HybridEvaluation,
        rank0_layout_count: int,
        star_walk_tried: int,
        fingerprint: str,
        started: float,
    ) -> None:
        nonlocal total_swaps, total_baseline_swaps
        nonlocal total_full_swaps, total_full_baseline_swaps
        nonlocal full_wins, full_ties, full_losses
        baseline = result.baseline
        reference_free = bool(getattr(args, "reference_free", False))
        use_baseline = result.metrics is None if reference_free else (
            result.metrics is None or metric_key(
                baseline, args.objective, args.depth_weight
            ) <= metric_key(result.metrics, args.objective, args.depth_weight)
        )
        metrics = baseline if use_baseline else result.metrics
        method = "qiskit_lightsabre_fallback" if use_baseline else result.method
        validate_route_trace(case, result, use_baseline)
        routed_path: object = "disabled"
        if args.output_dir is not None:
            write_mapping_json(case, result, use_baseline=use_baseline, output_dir=args.output_dir)
            if not getattr(args, "mapping_only", False):
                try:
                    routed_path, _, full_metrics, full_baseline = write_routed_qasm(
                        case,
                        result,
                        use_baseline=use_baseline,
                        output_dir=args.output_dir,
                        seed=args.seed,
                        restarts=args.lightsabre_restarts,
                        layout_candidates=args.full_qasm_layout_candidates,
                    )
                    total_full_swaps += full_metrics.swap_count
                    total_full_baseline_swaps += full_baseline.swap_count
                    full_wins += int(full_metrics.swap_count < full_baseline.swap_count)
                    full_ties += int(full_metrics.swap_count == full_baseline.swap_count)
                    full_losses += int(full_metrics.swap_count > full_baseline.swap_count)
                except ValueError as error:
                    if "QASM does not support multiple conditions" not in str(error):
                        raise
                    routed_path = "EXPORT_UNSUPPORTED"
        optimization = 0.0 if baseline.swap_count == 0 else (
            100.0 * (baseline.swap_count - metrics.swap_count) / baseline.swap_count
        )
        runtime_seconds = time.perf_counter() - started
        total_swaps += metrics.swap_count
        total_baseline_swaps += baseline.swap_count
        print(
            f"{case.name}\t{fingerprint}\t{classification.expert}\t{classification.routing_profile}/{classification.circuit_pattern}@{classification.topology_profile}\tlightsabre"
            f"\t{metrics.swap_count}\t{baseline.swap_count}\t{optimization:.2f}%\t{metrics.depth}\t{baseline.depth}"
            f"\t{method}\t{result.beam_tried}\t{result.beam_changed}\t{result.beam_improved}"
            f"\t{rank0_layout_count + star_walk_tried}\tPASS\t{runtime_seconds:.3f}\t{routed_path}",
            flush=True,
        )

    for case in test_cases:
        started = time.perf_counter()
        classification = analyses[case.name]
        fingerprint = interaction_fingerprint(case.qasm_path)
        route_key = route_keys[case.name]
        cached = route_cache.get(route_key)
        if cached is not None:
            cache_hits += 1
            result, rank0_layout_count, star_walk_tried = cached
            print(
                f"START {case.name}: gates={len(case.circuit[1])} active={case.circuit[0]} "
                f"lightsabre_restarts={args.lightsabre_restarts}",
                flush=True,
            )
            print(f"canonical route cache hit: {case.name}", flush=True)
            emit_case(
                case, classification, result, rank0_layout_count,
                star_walk_tried, fingerprint, started,
            )
            continue
        eval_args = copy.copy(args)
        medium_case = int(case.route_case[0]) <= 49 and int(circuit_features(case.circuit)["active_qubits"]) <= 49
        if medium_case:
            eval_args.adaptive_depth_slack = min(float(eval_args.adaptive_depth_slack), 0.02)
        eval_args.lookahead_size = (
            args.lookahead_size if classification.routing_profile == "local"
            else max(args.lookahead_size, args.dense_lookahead_size)
        )
        env = DenseHybridRoutingEnv(
            *case.route_case,
            lookahead_size=eval_args.lookahead_size,
            baseline_lookahead_size=args.lookahead_size,
            top_k=args.top_k,
            objective=args.objective,
            depth_weight=args.depth_weight,
            routing_profile=classification.routing_profile,
        )
        env.circuit_pattern = classification.circuit_pattern
        env.topology_profile = classification.topology_profile
        env.routing_policy = f"{classification.routing_profile}x{classification.topology_profile}"
        env.policy.topology_profile = classification.topology_profile
        if classification.topology_profile == "heavy_hex_like":
            eval_args.beam_interval = min(int(eval_args.beam_interval), 4)
            env.max_beam_calls = min(64, max(env.max_beam_calls, 6))
        print(
            f"START {case.name}: gates={len(case.circuit[1])} active={case.circuit[0]} "
            f"lightsabre_restarts={args.lightsabre_restarts}",
            flush=True,
        )
        baseline, baseline_events, baseline_initial, baseline_final, baseline_circuit = route_lightsabre_trace(
            case.route_case, args.lookahead_size, args.seed, args.lightsabre_restarts, case.qasm_path
        )
        env.lightsabre_metrics = baseline
        env.lightsabre_swaps = baseline.swap_count
        env.max_swaps = (
            max(10_000, len(env.gates) * 20)
            if getattr(eval_args, "reference_free", False)
            else baseline.swap_count if args.objective == "swap" else max(200, 2 * baseline.swap_count)
        )
        medium_case = int(case.route_case[0]) <= 49 and int(circuit_features(case.circuit)["active_qubits"]) <= 49
        layout_starts = args.multi_starts + (2 if medium_case else 0)
        layout_perturbations = args.layout_perturbations + (2 if medium_case else 0)
        layouts = generate_structure_aware_layouts(env, layout_starts, args.seed, layout_perturbations)
        core_layout_count = int(getattr(env, "sage_core_layout_count", len(layouts)))
        core_layouts = layouts[:core_layout_count]
        extra_layouts = layouts[core_layout_count:]
        medium_refined_layouts = _medium_sabre_refined_layouts(env, core_layouts)
        priority_candidates: list[tuple[list[int], int, str]] = []
        priority_seen = set()

        def add_priority(layout: list[int] | None, method: str) -> None:
            if layout is None:
                return
            key = tuple(layout)
            if key not in priority_seen:
                priority_seen.add(key)
                priority_candidates.append((layout, eval_args.lookahead_size, method))

        if (
            int(classification.features["components"]) == 1
            and (int(classification.features["max_degree"]) <= 2 or classification.topology_profile == "heavy_hex_like")
        ):
            add_priority(_path_interaction_layout(env, core_layouts[0]), "path_embed")
        if classification.routing_profile == "hub_sequence":
            add_priority(_hub_walk_layout(env, core_layouts[0]), "hub_priority")
        if classification.topology_profile == "heavy_hex_like":
            add_priority(_heavy_hex_center_ring_layout(env, core_layouts[0]), "heavy_hex_center_ring")
            for candidate in _heavy_hex_ring_layouts(env, core_layouts[0], limit=5):
                add_priority(candidate, "heavy_hex_ring")
        if classification.circuit_pattern in {"bv_like", "cc_like"}:
            for candidate in _hub_topology_layouts(env, core_layouts[0], classification.circuit_pattern, limit=3):
                add_priority(candidate, f"{classification.circuit_pattern}_topology")
        elif classification.circuit_pattern == "qft_like" and not (classification.topology_profile == "grid_like" and env.logical_qubits > 49):
            for candidate in _qft_topology_layouts(env, core_layouts[0], limit=3):
                add_priority(candidate, "qft_topology")

        priority_methods = {tuple(layout): method for layout, _, method in priority_candidates}
        core_candidates = [
            *((layout, lookahead) for layout, lookahead, _ in priority_candidates),
            *((layout, eval_args.lookahead_size) for layout in core_layouts if tuple(layout) not in priority_seen),
            *((layout, lookahead) for layout, lookahead in medium_refined_layouts if tuple(layout) not in priority_seen),
        ]
        priority_count = len(priority_candidates)
        equal_budget = bool(getattr(eval_args, "equal_budget", False))
        route_work = max(1, env.logical_qubits * len(env.gates))
        if equal_budget:
            # The paired spectral ablation uses the same predeclared number of
            # complete rank-0 routes. Deterministic nonspectral mappings fill
            # any short candidate pool without consulting a reference result.
            requested_budget = max(1, int(getattr(
                eval_args, "equal_candidate_budget", args.min_complete_route_candidates
            )))
            seen_candidates = {tuple(layout) for layout, _ in core_candidates}
            base_layout = list(core_candidates[0][0])
            for left in range(env.logical_qubits):
                for right in range(left + 1, env.logical_qubits):
                    if len(core_candidates) >= requested_budget:
                        break
                    candidate = list(base_layout)
                    candidate[left], candidate[right] = candidate[right], candidate[left]
                    key = tuple(candidate)
                    if key not in seen_candidates:
                        seen_candidates.add(key)
                        core_candidates.append((candidate, eval_args.lookahead_size))
                if len(core_candidates) >= requested_budget:
                    break
            used_physicals = set(base_layout)
            unused_physicals = [
                physical for physical in range(env.physical_qubits)
                if physical not in used_physicals
            ]
            for logical in range(env.logical_qubits):
                for physical in unused_physicals:
                    if len(core_candidates) >= requested_budget:
                        break
                    candidate = list(base_layout)
                    candidate[logical] = physical
                    key = tuple(candidate)
                    if key not in seen_candidates:
                        seen_candidates.add(key)
                        core_candidates.append((candidate, eval_args.lookahead_size))
                if len(core_candidates) >= requested_budget:
                    break
            layout_route_budget = min(len(core_candidates), requested_budget)
        else:
            # Preserve the dynamic budget for ordinary runs, while ensuring
            # expensive large routes still evaluate the requested minimum.
            dynamic_route_budget = max(1, min(len(core_candidates), 1_200_000 // route_work))
            pattern_minimum = {
                "qft_like": 5 if env.logical_qubits <= 80 else 3,
                "bv_like": 4,
                "cc_like": 4,
            }.get(classification.circuit_pattern, 1)
            priority_minimum = min(priority_count, 8) if classification.topology_profile == "heavy_hex_like" else 0
            layout_route_budget = min(
                len(core_candidates),
                max(dynamic_route_budget, args.min_complete_route_candidates, pattern_minimum, priority_minimum),
            )
        if len(core_candidates) > layout_route_budget:
            prescore_weights = _interaction_weights(env.gates, "uniform")
            ranked_layouts = sorted(
                enumerate(core_candidates),
                key=lambda item: (
                    _layout_pair_cost(item[1][0], env.topology, prescore_weights),
                    item[0],
                ),
            )
            priority_budget = min(priority_count, layout_route_budget)
            pinned = sorted(
                list(enumerate(core_candidates[:priority_count])),
                key=lambda item: (
                    _layout_pair_cost(item[1][0], env.topology, prescore_weights),
                    item[0],
                ),
            )[:priority_budget]
            pinned_indices = {index for index, _ in pinned}
            selected = pinned + [
                item for item in ranked_layouts if item[0] not in pinned_indices
            ][: max(0, layout_route_budget - len(pinned))]
            if priority_count == 0 and all(index != 0 for index, _ in selected):
                selected[-1] = (0, core_candidates[0])
            core_candidates = [candidate for _, candidate in sorted(selected, key=lambda item: item[0])]
            print(
                f"layout prefilter: {case.name} {len(core_layouts) + len(medium_refined_layouts)}"
                f"->{len(core_candidates)} core routes",
                flush=True,
            )
        extra_cap = 6 if classification.topology_profile == "heavy_hex_like" else 3
        extra_route_budget = (
            0 if equal_budget
            else min(len(extra_layouts), extra_cap, max(0, 300_000 // route_work))
        )
        candidate_layouts = [
            *core_candidates,
            *((layout, eval_args.lookahead_size) for layout in extra_layouts[:extra_route_budget]),
        ]
        best = None
        best_events: tuple[tuple[str, object], ...] = ()
        best_initial: tuple[int, ...] = ()
        best_final: tuple[int, ...] = ()
        best_layout_index = 0
        best_method = "rank0"
        best_beam_calls = 0
        beam_tried = 0
        beam_changed = 0
        beam_improved = 0
        rank0_candidates = []
        rank0_evaluations = 0
        route_deadline = (
            started + max(0.0, float(getattr(eval_args, "wall_clock_cap_s", 0.0)))
            if equal_budget and float(getattr(eval_args, "wall_clock_cap_s", 0.0)) > 0.0
            else None
        )
        for layout_index, (initial, candidate_lookahead) in enumerate(candidate_layouts):
            if route_deadline is not None and time.perf_counter() >= route_deadline:
                print(f"equal-budget wall-clock cap reached: {case.name}", flush=True)
                break
            rank0_evaluations += 1
            env.policy.lookahead_size = candidate_lookahead
            rank0_candidate = _rank0_trace(env, initial)
            if rank0_candidate is None:
                continue
            metrics, events, initial_mapping, final_mapping = rank0_candidate
            rank0_candidates.append((
                metric_key(metrics, args.objective, args.depth_weight),
                layout_index,
                initial,
                candidate_lookahead,
            ))
            if best is None or metric_key(
                metrics, args.objective, args.depth_weight
            ) < metric_key(best, args.objective, args.depth_weight):
                best = metrics
                best_events = events
                best_initial = initial_mapping
                best_final = final_mapping
                best_layout_index = layout_index
                best_method = priority_methods.get(
                    tuple(initial), "rank0" if layout_index == 0 else "multistart+rank0"
                )
                best_beam_calls = 0
            if best is not None and best.swap_count == 0 and not equal_budget:
                break
        exact_zero = best is not None and best.swap_count == 0

        # ponytail: Beam only the strongest rank-0 seeds; full-layout Beam scales
        # poorly and repeatedly explores layouts already shown to be inferior.
        route_work = max(1, env.logical_qubits * len(env.gates))
        disable_qft_ring_beam = (
            classification.circuit_pattern == "qft_like"
            and classification.topology_profile == "ring_like"
        )
        if equal_budget:
            beam_layout_budget = max(0, int(getattr(eval_args, "equal_beam_budget", 0)))
            env.max_beam_calls = beam_layout_budget
            eval_args._extra_deadline = route_deadline
        else:
            beam_layout_budget = 0 if exact_zero or disable_qft_ring_beam else max(0, min(4, 600_000 // route_work))
        beam_seeds = [
            candidate for candidate in sorted(rank0_candidates, key=lambda item: (item[0], item[1]))
            if candidate[1] < len(layouts)
        ][:beam_layout_budget]
        ranked_candidate_layouts = [
            (initial, lookahead) for _, _, initial, lookahead
            in sorted(rank0_candidates, key=lambda item: (item[0], item[1]))
        ]
        for _, layout_index, initial, candidate_lookahead in beam_seeds:
            env.policy.lookahead_size = candidate_lookahead
            beam_candidate = _beam_trace(env, initial, eval_args)
            if beam_candidate is None:
                continue
            metrics, events, initial_mapping, final_mapping, beam_calls, beam_deviations = beam_candidate
            beam_tried += beam_calls
            beam_changed += beam_deviations
            if best is None or metric_key(
                metrics, args.objective, args.depth_weight
            ) < metric_key(best, args.objective, args.depth_weight):
                best = metrics
                best_events = events
                best_initial = initial_mapping
                best_final = final_mapping
                best_layout_index = layout_index
                best_method = "beam" if layout_index == 0 else "multistart+beam"
                beam_improved += 1
                best_beam_calls = beam_calls

        result = HybridEvaluation(
            best,
            0,
            baseline,
            best_method,
            best_beam_calls,
            rank0_evaluations,
            best_events,
            best_initial,
            best_final,
            baseline_events,
            baseline_initial,
            baseline_final,
            time.perf_counter() - started,
            baseline_circuit,
            beam_tried=beam_tried,
            beam_changed=beam_changed,
            beam_improved=beam_improved,
        )
        rank0_layout_count = result.layouts_tried
        if not exact_zero and not equal_budget:
            result = _add_path_layout_candidate(result, case, eval_args)
            result = _add_star_walk_candidate(result, case, classification, eval_args)

            result = _add_heavy_hex_center_candidate(result, env, candidate_layouts, eval_args)
        star_walk_tried = int(result.layouts_tried > rank0_layout_count)
        reference_free = bool(getattr(eval_args, "reference_free", False))
        external_target = None if reference_free or references is None else references.best(case.name, fingerprint)
        large_grid_qft = (
            classification.circuit_pattern == "qft_like"
            and classification.topology_profile == "grid_like"
            and env.logical_qubits > 49
        )
        pattern_target = (
            max(0, baseline.swap_count - 1)
            if not reference_free
            and baseline.swap_count > 0
            and not large_grid_qft
            and (result.metrics is None or result.metrics.swap_count >= baseline.swap_count)
            and classification.circuit_pattern in {"qft_like", "bv_like", "cc_like"}
            else None
        )
        self_target = None if reference_free or external_target is not None else (
            pattern_target if pattern_target is not None else _self_improvement_target(
                classification, env, result
            )
        )
        search_target = (
            min(external_target, baseline.swap_count)
            if external_target is not None
            else self_target if self_target is not None else baseline.swap_count
        )
        eval_args._self_improve = self_target is not None
        # Reference gaps and bounded self-improvement share the same verified
        # search stages; candidates are adopted only when SWAP count decreases.
        pre_adaptive_result = result
        adaptive = (
            False if reference_free
            else _adaptive_extra_policy(classification, env, result, search_target, eval_args)
        )
        targeted_layouts = generate_targeted_layouts(env, layouts) if adaptive and _extra_budget_available(eval_args) else None
        if adaptive:
            print(f"adaptive extra search: {case.name} {eval_args._adaptive_reason}", flush=True)
        # Evolution found the only material Medium improvement in prior runs,
        # so self-improvement tries it before cheaper stages consume the timer.
        if adaptive and eval_args._self_improve and _extra_budget_available(eval_args):
            result = _add_evolutionary_layout_candidate(
                result, env, layouts, eval_args, search_target, targeted_layouts
            )
        if adaptive and _extra_budget_available(eval_args):
            result = _add_targeted_layout_candidates(result, env, layouts, eval_args, search_target, targeted_layouts)
        if adaptive and _extra_budget_available(eval_args):
            result = _add_adaptive_targeted_beam_candidate(
                result, env, eval_args, search_target, targeted_layouts
            )
        if adaptive and _extra_budget_available(eval_args):
            result = _add_routing_scored_layout_candidate(result, env, layouts, eval_args, search_target, targeted_layouts)
        if adaptive and not eval_args._self_improve and _extra_budget_available(eval_args):
            result = _add_evolutionary_layout_candidate(result, env, layouts, eval_args, search_target, targeted_layouts)
        if getattr(eval_args, "_tie_breaker", False) and _extra_budget_available(eval_args):
            result = _add_heavy_hex_lookahead_rescue(result, env, baseline, eval_args)
        if adaptive and pre_adaptive_result.metrics is not None and result.metrics is not None:
            swap_gain = pre_adaptive_result.metrics.swap_count - result.metrics.swap_count
            depth_regression = result.metrics.depth - pre_adaptive_result.metrics.depth
            depth_slack = max(8, int(pre_adaptive_result.metrics.depth * 0.50))
            min_gain = 5
            if depth_regression > depth_slack and swap_gain < min_gain:
                searched = result
                result = replace(
                    pre_adaptive_result,
                    beam_calls=searched.beam_calls,
                    beam_tried=searched.beam_tried,
                    beam_changed=searched.beam_changed,
                    beam_improved=searched.beam_improved,
                )
                print(f"adaptive depth guard: {case.name} kept pre-search route", flush=True)
        if route_key in duplicate_keys:
            route_cache[route_key] = (result, rank0_layout_count, star_walk_tried)
        emit_case(
            case, classification, result, rank0_layout_count,
            star_walk_tried, fingerprint, started,
        )

    wall_runtime = time.perf_counter() - evaluation_started
    print("\\n=== totals ===")
    print("total_swaps:", total_swaps)
    print("lightsabre_swaps:", total_baseline_swaps)
    print(
        "swap_optimization:",
        f"{100 * (total_baseline_swaps - total_swaps) / max(1, total_baseline_swaps):.2f}%",
    )
    print("wall_runtime_s:", round(wall_runtime, 3))
    print("canonical_route_cache_hits:", cache_hits)
    if total_full_baseline_swaps:
        print("\n=== complete QASM totals ===")
        print("full_export_swaps:", total_full_swaps)
        print("full_lightsabre_swaps:", total_full_baseline_swaps)
        print(
            "full_swap_optimization:",
            f"{100 * (total_full_baseline_swaps - total_full_swaps) / total_full_baseline_swaps:.2f}%",
        )
        print("full_win_tie_loss:", f"{full_wins}/{full_ties}/{full_losses}")
    if args.runtime_baseline_s is not None:
        reduction = 100.0 * (args.runtime_baseline_s - wall_runtime) / args.runtime_baseline_s
        print("runtime_reduction:", f"{reduction:.2f}%")
        print("runtime_target_30pct:", "PASS" if reduction >= 30.0 else "FAIL")

def self_check() -> None:
    chain = (8, [(index, index + 1) for index in range(7)])
    qft_like = (8, [(a, b) for a in range(8) for b in range(a + 1, 8)])
    hub_sequence = (12, [(0, leaf) for leaf in range(1, 12)])
    paired_hub = (
        41,
        [
            gate
            for leaf in range(1, 21)
            for gate in [
                (leaf, leaf + 20),
                (0, leaf),
                (leaf, leaf + 20),
                (0, leaf),
                (leaf, leaf + 20),
                (0, leaf),
                (leaf, leaf + 20),
                (0, leaf),
                (leaf, leaf + 20),
            ]
        ],
    )
    large_paired_hub = (
        161,
        [
            gate
            for leaf in range(1, 81)
            for gate in [
                (leaf, leaf + 80),
                (0, leaf),
                (leaf, leaf + 80),
                (0, leaf),
                (leaf, leaf + 80),
                (0, leaf),
                (leaf, leaf + 80),
                (0, leaf),
                (leaf, leaf + 80),
            ]
        ],
    )
    module = (6, [(0, 1), (1, 2), (2, 3), (0, 1), (4, 5), (3, 4)] * 20)
    symmetric = (12, [(a, b) for a in range(12) for b in range(a + 1, 12)] * 10)

    qft_pattern = classify_circuit((8, [(a, b) for a in range(8) for b in range(a + 1, 8)] * 2))
    bv_pattern = classify_circuit((12, [(0, leaf) for leaf in range(1, 12)]))
    cc_pattern = classify_circuit((12, [(0, leaf) for leaf in range(1, 12)] + [(0, 1)]))
    assert qft_pattern.circuit_pattern == "qft_like"
    assert bv_pattern.circuit_pattern == "bv_like"
    assert cc_pattern.circuit_pattern == "cc_like"

    grid_edges = make_grid_topology(49)[1]
    ring_edges = [(node, (node + 1) % 12) for node in range(12)]
    path_edges = [(node, node + 1) for node in range(11)]
    assert classify_topology(49, grid_edges).profile == "grid_like"
    assert classify_topology(12, ring_edges).profile == "ring_like"
    assert classify_topology(12, path_edges).profile == "path_like"
    assert classify_circuit(chain).routing_profile == "local"
    assert classify_circuit(qft_like).routing_profile != "hub_sequence"
    assert classify_circuit(hub_sequence).routing_profile == "hub_sequence"
    paired_result = classify_circuit(paired_hub)
    assert paired_result.expert == "arithmetic"
    assert paired_result.routing_profile == "local"
    assert not _is_large_paired_hub(paired_result.features)
    large_paired_result = classify_circuit(large_paired_hub)
    assert large_paired_result.expert == "dense_global"
    assert large_paired_result.routing_profile == "hub_sequence"
    assert _is_large_paired_hub(large_paired_result.features)
    assert classify_circuit(module).expert == "arithmetic"
    assert classify_circuit(symmetric).routing_profile in {"dense_temporal", "symmetric"}
    assert all(classify_circuit(case).expert in EXPERTS for case in (chain, qft_like, hub_sequence, paired_hub, large_paired_hub, module))

    rank0_env = argparse.Namespace(
        logical_qubits=4,
        physical_qubits=4,
        gates=[(0, 3), (1, 2), (0, 3)],
        topology=RoutingTopology(4, [(0, 1), (1, 2), (2, 3)]),
        policy=CandidatePolicy(lookahead_size=4),
    )
    rank0_first = _rank0_metrics(rank0_env, [0, 1, 2, 3])
    rank0_second = _rank0_metrics(rank0_env, [0, 1, 2, 3])
    assert rank0_first is not None and rank0_second is not None
    assert (rank0_first.swap_count, rank0_first.depth) == (

        rank0_second.swap_count,

        rank0_second.depth,

    )
    assert _evolution_rng(42, [0, 1, 2]).random() == _evolution_rng(42, [0, 1, 2]).random()
    assert _evolution_budget(
        argparse.Namespace(physical_qubits=49, logical_qubits=13, gates=[(0, 1)] * 700),
        argparse.Namespace(evolution_budget=2400),
    ) == 32
    assert _evolution_budget(
        argparse.Namespace(physical_qubits=49, logical_qubits=15, gates=[(0, 1)] * 200),
        argparse.Namespace(evolution_budget=2400),
    ) == 64




    beam_env = DenseHybridRoutingEnv(
        4,
        [(0, 3), (1, 2), (0, 3)],
        4,
        [(0, 1), (1, 2), (2, 3)],
        lookahead_size=4,
        baseline_lookahead_size=4,
        top_k=3,
        objective="swap",
        routing_profile="local",
    )
    mutation_rng = random.Random(42)
    moved = _mutate_layout(beam_env, [0, 1, 2, 3], mutation_rng)
    assert len(moved) == 4 and len(set(moved)) == 4 and all(0 <= physical < 4 for physical in moved)
    beam_args = argparse.Namespace(
        seed=42,
        beam_width=3,
        beam_depth=2,
        beam_branch=2,
        beam_interval=1,
        objective="swap",
        depth_weight=0.05,
    )
    beam_route = _beam_trace(beam_env, [0, 1, 2, 3], beam_args)
    assert beam_route is not None and beam_route[4] > 0 and 0 <= beam_route[5] <= beam_route[4]
    assert sum(kind == "gate" for kind, _ in beam_route[1]) == 3

    gcm_pairs = [(index % 11, (index % 11) + 1) for index in range(22)]
    gcm_like = circuit_features((13, [pair for pair in gcm_pairs for _ in range(35)][:762]))
    square_pairs = [(index % 17, (index % 17) + 1) for index in range(36)]
    square_like = circuit_features((18, [pair for pair in square_pairs for _ in range(33)][:1158]))
    medium_refine_env = argparse.Namespace(
        physical_qubits=49,
        logical_qubits=4,
        policy=CandidatePolicy(lookahead_size=20),
        gates=[(0, 3), (1, 2), (0, 3)],
        topology=RoutingTopology(49, [
            (row * 7 + column, row * 7 + column + 1)
            for row in range(7) for column in range(6)
        ] + [
            (row * 7 + column, (row + 1) * 7 + column)
            for row in range(6) for column in range(7)
        ]),
        circuit=RoutingCircuit(4, [(0, 3), (1, 2), (0, 3)]),
    )
    medium_refined = _medium_sabre_refined_layouts(
        medium_refine_env, [[0, 1, 2, 3]], lookaheads=(20, 32)
    )
    assert medium_refined and {lookahead for _, lookahead in medium_refined} <= {20, 32}
    large_refine_env = copy.copy(medium_refine_env)
    large_refine_env.physical_qubits = 441
    assert not _medium_sabre_refined_layouts(large_refine_env, [[0, 1, 2, 3]])
    qft_medium = circuit_features((18, [(a, b) for a in range(18) for b in range(a + 1, 18)] * 2))
    assert _medium_specializations(gcm_like, "local", 49)["repeat_temporal"]
    assert _medium_specializations(square_like, "local", 49)["long_temporal"]
    assert _medium_specializations(qft_medium, "symmetric", 49)["symmetric_refine"]
    assert not any(_medium_specializations(gcm_like, "local", 441).values())
    assert not any(_medium_specializations(square_like, "local", 441).values())
    assert not any(_medium_specializations(qft_medium, "symmetric", 441).values())

    class Grid:
        @staticmethod
        def distance(a, b):
            return abs(a // 3 - b // 3) + abs(a % 3 - b % 3)

        @staticmethod
        def neighbors(qubit):
            row, column = divmod(qubit, 3)
            return [
                r * 3 + c
                for r, c in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
                if 0 <= r < 3 and 0 <= c < 3
            ]

    class SpectralEnv:
        logical_qubits = physical_qubits = 6
        gates = module[1]
        edges = [(0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0)]
        topology = Grid()
        circuit = RoutingCircuit(logical_qubits, gates)

    uniform = _spectral_layout(SpectralEnv(), list(range(6)), 42, temporal=False)
    temporal = _spectral_layout(SpectralEnv(), list(range(6)), 42, temporal=True)
    assert uniform is not None and sorted(uniform) == list(range(6))
    assert temporal is not None and sorted(temporal) == list(range(6))

    class ReverseEnv:
        logical_qubits = 3
        physical_qubits = 4
        gates = [(0, 1), (1, 2)]
        topology = RoutingTopology(4, [(0, 1), (1, 2), (2, 3)])
        policy = CandidatePolicy(lookahead_size=4)

    reverse_layout = _reverse_rank0_layout(ReverseEnv(), [0, 3, 2])
    assert reverse_layout is not None and len(set(reverse_layout)) == 3
    assert all(0 <= physical < 4 for physical in reverse_layout)

    class StarEnv:
        logical_qubits = physical_qubits = 4
        gates = [(0, 1), (0, 2), (0, 3)]
        topology = RoutingTopology(4, [(0, 1), (1, 2), (2, 3)])
        circuit = RoutingCircuit(logical_qubits, gates)
        hub_logical = 0

    star_route = _star_walk_route(StarEnv(), [0, 1, 2, 3])
    assert star_route is not None and star_route[0].swap_count == 2
    assert sum(kind == "gate" for kind, _ in star_route[1]) == 3
    class DistanceTopology:
        @staticmethod
        def distance(a, b):
            return abs(a - b)

        @staticmethod
        def neighbors(node):
            return [other for other in (node - 1, node + 1) if 0 <= other < 4]

    class Layout:
        logical_to_physical = [0, 3, 2]
        physical_to_logical = [0, -1, 2, 1]

    class Context:
        done = [False, False]
        front_ids = {0: None}

        @staticmethod
        def layout():
            return Layout()

    potential_env = object.__new__(DenseHybridRoutingEnv)
    potential_env.ctx = Context()
    potential_env.topology = DistanceTopology()
    potential_env.circuit = RoutingCircuit(3, [(0, 1), (0, 2)])
    potential_env.gates = [(0, 1), (0, 2)]
    potential_env.potential_horizon = 64
    potential_env.potential_lambda = 0.85
    potential_env.hub_logical = 0
    potential_env.hub_routing = True
    near_weight = potential_env.potential_lambda ** potential_env._future_nodes()[0][1]
    far_weight = potential_env.potential_lambda ** potential_env._future_nodes()[1][1]
    assert near_weight > far_weight
    assert potential_env._next_hub_candidate() in {(0, 1), (2, 3)}
    assert potential_env._hub_queue_cost((2, 3)) < potential_env._hub_queue_cost()
    assert potential_env._leaf_swap_penalty((2, 3), 1.0) == 0.0

    validation_case = LoadedCase(
        "validation",
        Path("validation.qasm"),
        (2, [(0, 1)]),
        (2, [(0, 1)]),
        "validation",
    )
    validation_metrics = RouteMetrics(0, 1)
    validation_result = HybridEvaluation(
        validation_metrics,
        0,
        validation_metrics,
        "rank0",
        0,
        1,
        (("gate", 0),),
        (0, 1),
        (0, 1),
        (("gate", 0),),
        (0, 1),
        (0, 1),
        0.0,
    )
    validate_route_trace(validation_case, validation_result, use_baseline=False)

    references = ExternalReferences()
    references.add("case", "abc", 7)
    references.add("case", "abc", 5)
    assert references.best("case", "abc") == 5
    path_env = argparse.Namespace(
        logical_qubits=4,
        physical_qubits=4,
        gates=[(0, 1), (1, 2), (2, 3)],
        topology=RoutingTopology(4, [(0, 1), (1, 2), (2, 3)]),
    )
    path_layout = _path_interaction_layout(path_env, [0, 1, 2, 3])
    assert path_layout is not None
    path_metrics = _rank0_metrics(argparse.Namespace(**vars(path_env), policy=CandidatePolicy()), path_layout)
    assert path_metrics is not None and path_metrics.swap_count == 0

    ring_env = argparse.Namespace(
        logical_qubits=8,
        physical_qubits=12,
        gates=[pair for a in range(8) for b in range(a + 1, 8) for pair in ((a, b), (a, b))],
        edges=ring_edges,
        topology=RoutingTopology(12, ring_edges),
        circuit=RoutingCircuit(8, [pair for a in range(8) for b in range(a + 1, 8) for pair in ((a, b), (a, b))]),
        hub_logical=0,
        topology_profile="ring_like",
    )
    qft_layouts = _qft_topology_layouts(ring_env, list(range(8)), limit=4)
    hub_layouts_ring = _hub_topology_layouts(ring_env, list(range(8)), "bv_like", limit=4)
    generic_ring_layouts = _topology_profile_layouts(ring_env, list(range(8)), "symmetric", limit=4)
    assert qft_layouts and all(len(set(layout)) == 8 for layout in qft_layouts)
    assert hub_layouts_ring and all(len(set(layout)) == 8 for layout in hub_layouts_ring)
    assert generic_ring_layouts and all(len(set(layout)) == 8 for layout in generic_ring_layouts)
    center_ring_layout = _heavy_hex_center_ring_layout(ring_env, list(range(8)))
    assert center_ring_layout is not None and len(set(center_ring_layout)) == 8
    tie_args = argparse.Namespace(
        adaptive_min_gap=8, adaptive_min_ratio=0.05,
        medium_adaptive_min_gap=2, medium_adaptive_min_ratio=0.03,
        adaptive_timeout_s=10, targeted_layouts=12, layout_search_budget=40,
        evolution_budget=2400, layout_search_seeds=4,
    )
    tie_env = argparse.Namespace(
        topology_profile="heavy_hex_like", physical_qubits=57, logical_qubits=18,
        gates=[(index % 18, (index + 1) % 18) for index in range(50)],
    )
    tie_metrics = RouteMetrics(10, 20)
    tie_result = HybridEvaluation(tie_metrics, 0, tie_metrics, "rank0", 0, 1, (), (), (), (), (), (), 0.0)
    tie_class = classify_circuit((18, tie_env.gates))
    assert _adaptive_extra_policy(tie_class, tie_env, tie_result, 10, tie_args)
    heavy_env = argparse.Namespace(
        logical_qubits=18,
        physical_qubits=57,
        gates=[(index % 18, (index + 1) % 18) for index in range(80)],
        edges=[(node, node + 1) for node in range(56)],
        topology_profile="heavy_hex_like",
    )
    heavy_env.topology = RoutingTopology(57, heavy_env.edges)
    heavy_layouts = _heavy_hex_ring_layouts(heavy_env, list(range(18)), limit=5)
    assert all(len(layout) == 18 and len(set(layout)) == 18 for layout in heavy_layouts)
    phase_weights = [_windowed_interaction_weights(qft_like[1], phase, 0.28) for phase in (0.10, 0.50, 0.90)]
    assert len({tuple(sorted(weights.items())) for weights in phase_weights}) == 3
    hub_env = argparse.Namespace(
        logical_qubits=4,
        physical_qubits=9,
        gates=[(0, 1), (0, 2), (0, 3)],
        edges=[
            (row * 3 + column, row * 3 + column + 1)
            for row in range(3) for column in range(2)
        ] + [
            (row * 3 + column, (row + 1) * 3 + column)
            for row in range(2) for column in range(3)
        ],
        topology=RoutingTopology(9, [
            (row * 3 + column, row * 3 + column + 1)
            for row in range(3) for column in range(2)
        ] + [
            (row * 3 + column, (row + 1) * 3 + column)
            for row in range(2) for column in range(3)
        ]),
        hub_logical=0,
    )
    hub_layouts = _small_hub_layouts(hub_env, [0, 1, 2, 3], limit=2)
    assert 1 <= len(hub_layouts) <= 2 and all(len(set(layout)) == 4 for layout in hub_layouts)
    self_result = replace(validation_result, metrics=RouteMetrics(30, 10))
    self_classification = classify_circuit((18, [(index % 17, (index % 17) + 1) for index in range(300)]))
    self_env = argparse.Namespace(physical_qubits=49)
    assert _self_improvement_target(self_classification, self_env, self_result) == 29
    huge_env = argparse.Namespace(physical_qubits=441)
    huge_classification = classify_circuit((433, [(index % 432, (index % 432) + 1) for index in range(2000)]))
    assert _self_improvement_target(huge_classification, huge_env, self_result) is None

    parser = build_parser()
    default_args = parser.parse_args(["--self-check"])
    legacy_args = parser.parse_args(["--self-check", "--complete-route-candidates", "4"])
    minimum_args = parser.parse_args(["--self-check", "--min-complete-route-candidates", "3"])
    assert legacy_args.min_complete_route_candidates == 4
    assert minimum_args.min_complete_route_candidates == 3

    print("SAGE XV circuit/topology/layout/search/validation self-check passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Circuit-and-topology-aware LightSABRE routing with model-free Beam search"
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "large")
    parser.add_argument("--topology", type=Path, default=None)
    parser.add_argument("--case-manifest", type=Path, default=None)
    parser.add_argument("--analysis-output", type=Path, default=PROJECT_ROOT / "sage_xv_analysis.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "routed_output" / "sage_xv")
    parser.add_argument("--split", choices=("train", "test"), default="test", help="route the train or test split")
    parser.add_argument("--shard-count", type=lambda value: max(1, int(value)), default=1)
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based contiguous case shard")
    parser.add_argument(
        "--no-qasm",
        action="store_true",
        help="disable routed QASM export but keep mapping JSON export",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    mode.add_argument("--self-check", action="store_true")
    parser.add_argument("--dense-lookahead-size", type=int, default=64)
    parser.add_argument("--lookahead-size", type=int, default=20)
    parser.add_argument("--lightsabre-restarts", type=int, default=6, help="canonical LightSABRE random restarts")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-baseline-s", type=float, default=None)
    parser.add_argument("--multi-starts", type=int, default=6)
    parser.add_argument("--layout-perturbations", type=int, default=4)
    parser.add_argument(
        "--min-complete-route-candidates",
        "--complete-route-candidates",
        dest="min_complete_route_candidates",
        type=lambda value: max(1, int(value)),
        default=3,
        help="minimum core initial-layout candidates evaluated by complete routing",
    )
    parser.add_argument(
        "--full-qasm-layout-candidates",
        type=lambda value: max(1, int(value)),
        default=6,
        help="real-CX SAGE layouts evaluated when exporting a complete QASM",
    )
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--beam-depth", type=int, default=4)
    parser.add_argument("--beam-branch", type=int, default=3)
    parser.add_argument("--beam-interval", type=int, default=12)
    parser.add_argument("--objective", choices=("swap", "swap-depth", "depth"), default="swap")
    parser.add_argument("--depth-weight", type=float, default=0.05)
    parser.add_argument("--targeted-layouts", type=int, default=12)
    parser.add_argument("--layout-search-seeds", type=int, default=4)
    parser.add_argument("--layout-search-budget", type=int, default=40)
    parser.add_argument("--layout-search-rounds", type=int, default=3)
    parser.add_argument("--evolution-budget", type=int, default=2400)
    parser.add_argument("--evolution-population", type=int, default=10)
    parser.add_argument("--evolution-seeds", type=int, default=12)
    parser.add_argument("--layout-search-logicals", type=int, default=8)
    parser.add_argument("--layout-search-empty", type=int, default=2)
    parser.add_argument("--reference-results", type=Path, action="append", default=[])
    parser.add_argument(
        "--reference-free",
        action="store_true",
        help="disable reference targets, adaptive extra search, and final baseline fallback",
    )
    parser.add_argument("--adaptive-timeout-s", type=float, default=180.0, help="per-circuit cap for reference-triggered extra search")
    parser.add_argument("--adaptive-min-gap", type=int, default=8)
    parser.add_argument("--adaptive-min-ratio", type=float, default=0.05)
    parser.add_argument("--medium-adaptive-min-gap", type=int, default=2)
    parser.add_argument("--medium-adaptive-min-ratio", type=float, default=0.03)
    parser.add_argument("--adaptive-depth-slack", type=float, default=0.03)
    parser.add_argument("--adaptive-min-gain-ratio", type=float, default=0.01)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    if args.self_check:
        self_check()
        return
    # Keep the output directory for mapping-only runs; --no-qasm only skips
    # replaying the event trace into a routed QASM file.
    args.mapping_only = bool(args.no_qasm)
    args.output_dir = None if args.output_dir is None else args.output_dir.resolve()
    train_cases, test_cases, source = load_dataset_cases(args)
    analyses = write_analysis(train_cases, test_cases, args.analysis_output.resolve())
    profile_counts = Counter(item.routing_profile for item in analyses["train"].values())
    print("case source:", source)
    print("routing profiles:", dict(profile_counts))
    topology_counts = Counter(item.topology_profile for item in analyses["train"].values())
    pattern_counts = Counter(item.circuit_pattern for item in analyses["train"].values())
    print("topology profiles:", dict(topology_counts))
    print("circuit patterns:", dict(pattern_counts))
    print("analysis:", args.analysis_output.resolve())
    if args.analyze_only:
        return
    target_cases = train_cases if args.split == "train" else test_cases
    target_analyses = analyses[args.split]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < --shard-count")
    chunk = (len(target_cases) + args.shard_count - 1) // args.shard_count
    start = args.shard_index * chunk
    target_cases = target_cases[start:min(start + chunk, len(target_cases))]
    target_analyses = {case.name: target_analyses[case.name] for case in target_cases}
    print(
        "routing shard:",
        f"{args.shard_index + 1}/{args.shard_count}",
        f"range=[{start},{start + len(target_cases)})",
    )
    print("routing split:", args.split, "cases:", len(target_cases))
    references = load_external_references(args.reference_results) if args.reference_results else None
    route_with_experts(target_cases, target_analyses, args, references)

if __name__ == "__main__":
    main()


