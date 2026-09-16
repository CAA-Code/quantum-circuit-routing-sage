# SAGE: Structure-Aware Gate Embedding for Quantum Circuit Routing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**SAGE** (Structure-Aware Gate Embedding) is a transparent, training-free quantum circuit routing framework that bridges the gap between global spectral placement and local dynamic routing. Unlike learned policies (e.g., reinforcement learning) or purely local search heuristics, SAGE utilizes a novel **Double-Spectrum Expansion** to explicitly couple the logical interaction graph with the physical hardware Laplacian, providing a rigorous structural prior for SWAP insertion.

This repository contains the reference implementation (SAGE-XV), benchmark circuits, device topologies, and reproducibility scripts for the paper *"Quantum Circuit Routing Beyond Learned Policies: Spectral Placement and Structural SWAP Lower Bounds."*

## 💡 Key Innovations

1. **Double-Spectrum Expansion**: A joint functional that couples the time-weighted logical Laplacian ($L_l^{(\rho)}$) with the hardware Laplacian pseudoinverse ($L_p^+$):

$$
\mathcal{R}_{\rho}(\pi) = \mathrm{tr}(L_p^+ P_\pi L_l^{(\rho)} P_\pi^\top) = \sum_{r=2}^N \sum_{s=2}^n \frac{\mu_s}{\lambda_r} (u_r^\top P_\pi v_s)^2
$$

This separates logical communication modes ($\mu_s$), physical transport bottlenecks ($1/\lambda_r$), and their placement-dependent projection.

2. **Structural SWAP Certificates**: Proves a single-layer SWAP lower bound for disjoint-gate layers:

$$
S_M^*(\pi) \ge \frac{1}{2} \bigl[ \mathcal{R}_M(\pi) - \lvert M \rvert \bigr]_+
$$

and a temporal multilayer extension with an explicit discount factor. These act as auditable structural guarantees for routing difficulty.

3. **Training-Free Workflow**: SAGE eliminates the need for reward design, training corpora, or learned-policy inference. It generates explicit hardware-aware candidate placements via low-frequency spectral coordinates, rounds them to injective mappings, and evaluates them using a bounded dynamic router (LightSABRE-style).

## 📁 Repository Structure

*   `src/`: Core implementation of the SAGE-XV routing engine.
    *   `SAGE-XV.py`: Main entry point for the routing algorithm.
*   `topologies/`: Hardware coupling graphs used in experiments.
    *   `grid_4x4.json`, `grid_7x7.json`, `grid_21x21.json`
    *   `ring_49.json`, `heavy_hex_57.json`
*   `circuits/`: QASM benchmark circuits.
    *   `benchmark_40/`: Small/test circuits (40 files).
    *   `benchmark_46/`: Medium/train circuits (46 files).
    *   `benchmark_87/`: Large/train circuits (87 files).
*   `requirements.txt`: Pinned dependencies (cirq, networkx, numpy, qiskit).
*   `LICENSE`: MIT License.

## 🚀 Installation

SAGE requires **Python 3.10 or newer**.

```bash
# Clone the repository
git clone https://github.com/CAA-Code/quantum-circuit-routing-sage.git
cd quantum-circuit-routing-sage

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Usage

SAGE-XV operates as a command-line tool. You can run a self-check, or route a specific circuit using a hardware topology.

**Self-Check (Verify Installation):**
```bash
python src/SAGE-XV.py --self-check
```

**Run Routing on a Benchmark:**
```bash
python src/SAGE-XV.py \
  --dataset-dir circuits/benchmark_46 \
  --topology topologies/grid_7x7.json \
  --eval-only
```

**Available Arguments:**
*   `--dataset-dir`: Path to the directory containing QASM circuit files.
*   `--topology`: Path to the JSON file defining the hardware coupling graph.
*   `--eval-only`: Runs the full placement and routing evaluation pipeline.
*   `--seed`: Random seed for reproducibility (default: 42).

## 📊 Benchmark Highlights

SAGE demonstrates competitive or superior performance against established baselines (SABRE, TKET, LightSABRE, IBM AIRouting) without any reinforcement learning.

*   **Reference-Free Production (G1/G2)**: Reduces aggregate explicit SWAPs by **18.35%** on Medium circuits and **28.73%** on Large circuits compared to the default candidate source.
*   **Equal-Envelope Control (G3/G4)**: At a fixed budget of 12 candidates, spectral candidates provide an additional **1.93%** (Medium) and **3.34%** (Large) reduction.
*   **Common-Valid Intersection (vs. AIRouting)**: Accepted SAGE reduces SWAPs by **22.22%** (Small), **45.58%** (Medium), and **39.46%** (Large).
*   **Topologies**: Evaluated on 4x4, 7x7, 21x21 grids, Ring-49, and Heavy-Hex-57. SAGE achieves a **28.09%** reduction on Ring-49 and **14.16%** on Heavy-Hex-57 compared to LightSABRE.

## 📖 Theoretical Background

The core theoretical contribution is the **Double-Spectrum Expansion**, which converts the placement problem into a trace form. This allows the algorithm to:
1.  Compute the effective-resistance transport energy of a placement.
2.  Audit the logical/physical mode projection before routing begins.
3.  Derive rigorous lower bounds on the number of SWAPs required for disjoint-gate layers (Theorem 2, Corollary 1) and multilayer circuits (Theorem 3).

The algorithm is designed around a non-RL division of labor: **spectral analysis supplies the global placement prior, while finite search resolves local path, congestion, and gate-order decisions.**


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

The authors acknowledge the institutional and computational support provided for the experiments reported in the paper. This work received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
