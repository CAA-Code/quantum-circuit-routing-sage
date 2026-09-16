# SAGE: Structure-Aware Gate Embedding for Quantum Circuit Routing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**SAGE** (Structure-Aware Gate Embedding) is a transparent, training-free quantum circuit routing framework that bridges the gap between global spectral placement and local dynamic routing. Unlike learned policies (e.g., reinforcement learning) or purely local search heuristics, SAGE utilizes a novel **Double-Spectrum Expansion** to explicitly couple the logical interaction graph with the physical hardware Laplacian, providing a rigorous structural prior for SWAP insertion.

This repository contains the reference implementation (SAGE-XV), benchmark circuits, device topologies, and reproducibility scripts for the paper *"Quantum Circuit Routing Beyond Learned Policies: Spectral Placement and Structural SWAP Lower Bounds."*

## 💡 Key Innovations

1. **Double-Spectrum Expansion**: A joint functional that couples the time-weighted logical Laplacian (\(L_l^{(\rho)}\)) with the hardware Laplacian pseudoinverse (\(L_p^+\)):
   \[
   \mathcal{R}_{\rho}(\pi) = \operatorname{tr}\left(L_p^+ P_\pi L_l^{(\rho)} P_\pi^\top\right) = \sum_{r=2}^N \sum_{s=2}^n \frac{\mu_s}{\lambda_r} \left(u_r^\top P_\pi v_s\right)^2
   \]
   This separates logical communication modes (\(\mu_s\)), physical transport bottlenecks (\(1/\lambda_r\)), and their placement-dependent projection.

2. **Structural SWAP Certificates**: Proves a single-layer SWAP lower bound for disjoint-gate layers (\(S_M^*(\pi) \ge \frac{1}{2}[\mathcal{R}_M(\pi) - |M|]_+\)) and a temporal multilayer extension with an explicit discount factor. These act as auditable structural guarantees for routing difficulty.

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
