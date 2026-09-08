#!/usr/bin/env python3
"""Canonical routing-only input shared by Ver.X, Qiskit SABRE and TKET."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm


def _quantum_qasm(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("creg ", "measure ", "barrier ")):
            continue
        lines.append(re.sub(r"^\s*if\s*\([^)]*\)\s*", "", line))
    if not any(line.strip().upper().startswith("OPENQASM ") for line in lines):
        lines.insert(0, "OPENQASM 2.0;")
    return "\n".join(lines) + "\n"


def canonical_interactions(path: Path) -> tuple[int, list[tuple[int, int]]]:
    """Decompose once with Cirq and retain the ordered two-qubit interactions."""
    operations, logical_of = canonical_operations(path)
    interactions = [
        (logical_of[operation.qubits[0]], logical_of[operation.qubits[1]])
        for operation in operations
        if len(operation.qubits) == 2 and not cirq.is_measurement(operation)
    ]
    return len(logical_of), interactions


@lru_cache(maxsize=None)
def canonical_operations(path: Path):
    """Return the exact Cirq decomposition and logical index used by routing."""
    path = path.resolve()
    circuit = circuit_from_qasm(_quantum_qasm(path))
    operations = tuple(cirq.decompose(circuit.all_operations(), keep=lambda op: len(op.qubits) <= 2))
    unsupported = [operation for operation in operations if len(operation.qubits) > 2]
    if unsupported:
        raise ValueError(f"Cirq could not decompose operation: {unsupported[0]!r}")
    two_qubit = [
        operation for operation in operations
        if len(operation.qubits) == 2 and not cirq.is_measurement(operation)
    ]
    qubits = sorted({qubit for operation in two_qubit for qubit in operation.qubits}, key=str)
    index = {qubit: number for number, qubit in enumerate(qubits)}
    return operations, index


def canonical_qubit_names(path: Path) -> tuple[str, ...]:
    """Return canonical logical-qubit names in the same order as interactions."""
    _, logical_of = canonical_operations(path)
    return tuple(str(qubit) for qubit, _ in sorted(logical_of.items(), key=lambda item: item[1]))


def canonical_qasm(path: Path) -> str:
    logical_qubits, interactions = canonical_interactions(path)
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{logical_qubits}];"]
    lines.extend(f"cx q[{a}],q[{b}];" for a, b in interactions)
    return "\n".join(lines) + "\n"


def interaction_fingerprint(path: Path) -> str:
    logical_qubits, interactions = canonical_interactions(path)
    payload = f"{logical_qubits}|" + ";".join(f"{a},{b}" for a, b in interactions)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]
