# SAGE-XV: Spectral-Aware Quantum Circuit Routing

This repository contains a minimal, reproducible release of **SAGE-XV** for quantum-circuit initial placement and routing on connectivity-constrained quantum hardware.

## Repository contents

- `src/`: SAGE-XV routing implementation and canonical QASM input utilities.
- `topologies/`: hardware connectivity graphs used in the benchmark release.
- `circuits/benchmark_40/`: 40 small benchmark circuits.
- `circuits/benchmark_46/`: 46 medium benchmark circuits.
- `circuits/benchmark_87/`: 87 large benchmark circuits.
- `LICENSE`: MIT License.

The numbers 40, 46, and 87 denote the **number of circuit files in each benchmark set**, not the number of qubits in one circuit. The topology files are named by their physical-qubit counts: 16, 49, and 441 qubits.

## Requirements

Python 3.10 or newer is recommended. Install the pinned dependencies with:

```bash
python -m pip install -r requirements.txt
```

The release was checked with Cirq 1.5.0, NetworkX 3.4.2, NumPy 2.2.6, and Qiskit 0.46.3.

## Quick start

Run the built-in self-check:

```bash
python src/SAGE-XV.py --self-check
```

For a benchmark directory containing `train/` and `test/` subdirectories, route the test split on a supplied topology:

```bash
python src/SAGE-XV.py --dataset-dir path/to/dataset --topology topologies/grid_7x7_49q.json --eval-only
```

Outputs are written to the default `src/routed_output/sage_xv` directory unless `--output-dir` is supplied. The command-line interface exposes additional options for benchmark-scale experiments; run `python src/SAGE-XV.py --help` for details.

## Input formats

- Circuits: OpenQASM 2.0 files (`.qasm`).
- Topologies: JSON files with the fields `num_qubits` and `edges`, where each edge is a pair of physical-qubit indices.

## Notes on the benchmark sets

The benchmark directories preserve the original QASM filenames, including both original and transpiled variants where present. The 40-circuit set uses a 4x4 grid (16 physical qubits), the 46-circuit set uses a 7x7 grid (49 physical qubits), and the 87-circuit set uses a 21x21 grid (441 physical qubits) in the corresponding experiments.

## Citation

Please cite the accompanying paper when using SAGE-XV in academic work.
