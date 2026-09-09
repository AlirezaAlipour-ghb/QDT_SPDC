# QDT_SPDC

**Quantum digital twin and analytical benchmarks for multimode spontaneous parametric down-conversion (SPDC) in thermal-loss channels.**

> **Suggested GitHub “About” description:**  
> QuTiP-based quantum digital twin for multimode SPDC with thermal-loss channels, exact Gaussian purity/fidelity benchmarks, polarization-resolved counting and CHSH analysis, and finite-Fock convergence studies.

## Overview

This repository contains the numerical and analytical calculations associated with **“Analytical Benchmarks for a Multimode SPDC Digital Twin”** by Alireza Alipour and Jonathan L. Habif. The implementation combines a finite-Fock density-matrix digital twin with cutoff-independent analytical benchmarks for noisy, lossy, polarization–spectral SPDC systems.

The code supports two complementary benchmark configurations:

1. **State-level benchmark:** identical thermal-loss channels act directly on the SPDC modes, and the resulting state is evaluated using global purity and squared Uhlmann–Jozsa fidelity.
2. **Measurement-level benchmark:** the SPDC state is first transformed by a lossless nonpolarizing beam splitter (NPBS); independent thermal-loss channels then act on the idler and signal output arms before polarization-resolved singles, coincidences, visibility, and conditional CHSH quantities are evaluated.

The numerical model is implemented with **QuTiP** and finite local Fock dimension $d$. Analytical Gaussian formulas provide an infinite-dimensional reference and extend the model to multimode regimes that become impractical for direct density-matrix propagation.

## Physical model

### Polarization–spectral mode structure

The discrete SPDC basis is

$$
(H,f_0),(V,f_0),
\left\{
(H,f_k^+),(V,f_k^-),(H,f_k^-),(V,f_k^+)
\right\}_{k=1}^{N_k}.
$$

The number of physical optical modes and independent conjugate SPDC pairs are

$$
N_{\mathrm{mode}}=2+4N_k,
\qquad
M=1+2N_k.
$$

Here $N_k$ is the integer sideband-pair count of the discrete digital-twin Hamiltonian. It is **not** the spectral Schmidt number. Noninteger $N_k$ values appear only as interpolation coordinates in the analytical ternary visualizations.

For equal pair gain $g$, each conjugate pair is a two-mode squeezed-vacuum (TMSV) state with

$$
P_n=\operatorname{sech}^2(g)\tanh^{2n}(g),
\qquad
\bar n_0=\sinh^2(g).
$$

### Thermal-loss channel

Each optical mode is coupled to an independent thermal environment through a virtual beam splitter,

$$
\hat a_{\mathrm{out}}
=
\sqrt{T}\,\hat a_{\mathrm{in}}
+
\sqrt{1-T}\,\hat e,
\qquad
\langle \hat e^\dagger \hat e\rangle=\bar n_{\mathrm{th}},
$$

where $T$ is the channel transmissivity and $\bar n_{\mathrm{th}}$ is the mean thermal occupation.

For one SPDC pair, define

$$
A=T\cosh(2g)+(1-T)(2\bar n_{\mathrm{th}}+1),
\qquad
C=T\sinh(2g).
$$

The exact Gaussian pair purity and squared target-state fidelity are

$$
\mathcal P_{\mathrm{pair}}=\frac{1}{A^2-C^2},
$$

$$
\mathcal F_{\mathrm{pair}}
=
\frac{4}{
[\cosh(2g)+A]^2-[\sinh(2g)+C]^2
}.
$$

For $M=1+2N_k$ identical independent pairs,

$$
\mathcal P_{N_k}
=
\mathcal P_{\mathrm{pair}}^{\,M},
\qquad
\mathcal F_{N_k}
=
\mathcal F_{\mathrm{pair}}^{\,M}.
$$

These expressions contain no finite-Fock cutoff.

### Reduced polarization-count model

For the post-NPBS measurement branch,

$$
\bar n_0=\sinh^2 g,
\qquad
b=(1-T)\bar n_{\mathrm{th}},
$$

$$
Q=T^2\eta(1-\eta)\bar n_0,
\qquad
B_{\mathrm{th}}=M^2b(T\bar n_0+b),
$$

where $\eta$ is the NPBS transmissivity.

The same- and cross-polarization coincidence moments are

$$
C_{\mathrm{same}}
=
B_{\mathrm{th}}+MQ\sin^2(\alpha-\beta),
$$

$$
C_{\mathrm{cross}}
=
B_{\mathrm{th}}+MQ\cos^2(\alpha-\beta).
$$

The reduced-model visibility and conditional CHSH quantity are

$$
V_{\mathrm{pol}}
=
\frac{MQ}{MQ+2B_{\mathrm{th}}},
\qquad
S_{\mathrm{CHSH}}^{(1)}
=
2\sqrt 2\,V_{\mathrm{pol}}.
$$

The density-matrix calculation instead evaluates the Horodecki maximum

$$
S_{\max}=2\sqrt{u_1+u_2},
$$

using the two largest eigenvalues of the conditioned polarization correlation tensor $\Gamma^{T}\Gamma$. This CHSH quantity is evaluated after one-photon-per-arm conditioning and should not be interpreted as a loophole-free Bell-test statistic without an explicit detector-efficiency and postselection model.

## Numerical truncation and convergence

Each bosonic mode is represented in the local Fock basis

$$
\mathcal H_d=\mathrm{span}\{|0\rangle,\ldots,|d-1\rangle\}.
$$

Before the NPBS,

$$
D_{\mathrm{sys}}=d^{\,2+4N_k},
$$

so direct density-matrix propagation becomes exponentially expensive with $N_k$.

Useful a-priori truncation diagnostics are

$$
\epsilon_{\mathrm{SPDC}}(d)=\tanh^{2d}(g),
\qquad
\epsilon_{\mathrm{th}}(d)
=
\left(
\frac{\bar n_{\mathrm{th}}}{1+\bar n_{\mathrm{th}}}
\right)^d.
$$

The manuscript uses explicit successive-cutoff tests rather than relying only on these tail estimates:

- **State-level benchmark:** $d=2,\ldots,6$; $d=6$ is the first tested cutoff satisfying the adopted $10^{-2}$ full-grid theory-agreement and successive-cutoff criteria.
- **Measurement-level benchmark:** $d=2,\ldots,6$; $d=4$ is numerically sufficient under the normalized $10^{-3}$ successive-cutoff criterion, with $d=5,6$ providing further stability checks.

## Repository contents

```text
QDT_SPDC/
├── README.md
├── QDT_SPDC_reproducible.ipynb
├── main.py
├── matrixes_representation.py
├── requirements.txt
├── references/
│   ├── reference_main.bib
│   └── reference_SI.bib
└── manuscript/
    ├── main.tex
    └── SI.tex
```

`QDT_SPDC_reproducible.ipynb` is self-contained: it includes the helper routines from `main.py` and `matrixes_representation.py` directly in the notebook, followed by the analytical and numerical figure-generation cells.

## Installation

```bash
git clone https://github.com/AlirezaAlipour-ghb/QDT_SPDC.git
cd QDT_SPDC

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab QDT_SPDC_reproducible.ipynb
```

The implementation uses `qutip` for quantum states/operators and thermal density matrices, and `qutip-qip` for subsystem operator expansion.

## Notebook workflow

Run the notebook from top to bottom. The main sections are:

1. environment and imports;
2. plotting and output configuration;
3. symbolic/density-matrix visualization helpers;
4. SPDC Hamiltonian, NPBS, thermal-loss channel, fidelity, and photon-statistics primitives;
5. complete finite-Fock state-evolution routine;
6. exact Gaussian purity/fidelity theory;
7. state-level simulation-versus-theory validation;
8. reduced polarization-count theory;
9. Stokes operators, one-photon-per-arm projection, visibility, and Horodecki CHSH estimator;
10. measurement-level theory-versus-simulation comparison;
11. Cartesian and ternary coincidence/CHSH phase maps;
12. state-level and measurement-level finite-Fock convergence studies.

Generated figures are written to `images/`, and numerical fields/contours are written to `csv_data/`.

## Reproducibility notes

- The analytical $N_k>0$ purity/fidelity continuation follows exactly from the factorized independent-pair model; it is not a brute-force density-matrix extrapolation.
- Ternary plots use a constrained simplex and continuously interpolate $N_k$ only for visualization.
- The reduced polarization-count model is a weak-gain model. Its finite-$g$ phase maps should be interpreted as analytical extrapolations when used outside the weak-gain validation point.
- Expected counts use a deterministic normalization $N_{\mathrm{meas}}$; no stochastic detector sampling is performed.
- The final convergence cells are computationally expensive, especially at $d=5,6$.

## Citation

If this repository contributes to published work, please cite the accompanying manuscript and the software repository:

```bibtex
@misc{AlipourQDTSPDC2026,
  author       = {Alipour, Alireza},
  title        = {{QDT_SPDC}: Quantum Digital Twin Simulation Code for Multimode SPDC},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/AlirezaAlipour-ghb/QDT_SPDC}
}
```

The complete manuscript and Supplemental Material bibliographies are included under `references/`.

## License

This repository uses the MIT License.
