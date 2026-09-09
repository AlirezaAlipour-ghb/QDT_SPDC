

import numpy as np
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import qutip as qt
from qutip_qip.operations import expand_operator
import os


def compute_precision(Fk_dim, g, N_k):
    D = Fk_dim ** (2 + 4 * N_k)
    max_n = np.sqrt(D) - 1
    p_min = np.tanh(g) ** (2 * max_n)
    epsilon = 1e-100
    precision = min(12, max(4, int(-np.log10(p_min + epsilon) + 2)))
    return precision


#Hamiltonian SPDC: all pair correlations between modes with orthogonal polarization only
def hamiltonian_spdc(N_k, Fk_dim, g_spdc):
    '''Mode structure:
    - Index 0: H, f0
    - Index 1: V, f0
    - For k in 0 to N_k-1:
        Index 2 + 4k:     H, f_k^+
        Index 2 + 4k + 1: V, f_k^-
        Index 2 + 4k + 2: H, f_k^-
        Index 2 + 4k + 3: V, f_k^+ '''
    mode_labels = [("H", "f0"), ("V", "f0")]
    for k in range(N_k):
        mode_labels += [("H", f"f{k+1}+"), ("V", f"f{k+1}-"), ("H", f"f{k+1}-"), ("V", f"f{k+1}+")]
    total_modes = len(mode_labels)
    # Annihilation operators:
    def a(m):
        ops = [qt.qeye(Fk_dim) for _ in range(total_modes)]
        ops[m] = qt.destroy(Fk_dim)
        return qt.tensor(*ops)
    a_ops = [a(i) for i in range(total_modes)]
    # Energy conservation (e.g., fk+ + fk− =const)
    # Momentum conservation (phase-matching with conjugate frequencies/ signal and idler are created in opposite k modes, or at least k-conjugate pairs)
    H_spdc = 0
    def match(f1, f2):
        return (f1 == f2 == "f0") or (f1[:-1] == f2[:-1] and {f1[-1], f2[-1]} == {"+", "-"})
    # Add SPDC terms only for orthogonal polarization pairs
    for i in range(total_modes):
        for j in range(i + 1, total_modes):  # i < j to ensure Mode symmetry (e.g., indistinguishability, bosonic symmetrization)
            pol_i, freq_i = mode_labels[i]
            pol_j, freq_j = mode_labels[j]
            if pol_i != pol_j and match(freq_i, freq_j):  # Polarization orthogonality (H-V pairs) and momentum-Energy conservation
                H_pair = g_spdc * (a_ops[i].dag() @ a_ops[j].dag())
                H_spdc += H_pair + H_pair.dag()

    return H_spdc


# Hamiltonian non polarizing beam splitter (NPBS)
def hamiltonian_np_beamsplitter(Fk_dim,T_bs,phi_bs):
    theta = np.arccos(np.sqrt(T_bs))
    a = qt.destroy(Fk_dim) & qt.qeye(Fk_dim)
    b = qt.qeye(Fk_dim) & qt.destroy(Fk_dim)
    H_bs = -1j * np.exp(1j * phi_bs) * a.dag() @ b
    H_bs += H_bs.dag()
    U_bs = (-1j * theta * H_bs).expm()
    return U_bs

# Non Polarizing Beam Spliter (NPBS)
def apply_NPBS(rho_in, Fk_dim, T_bs, nth_bs, phi_bs, N_k):
    total_modes = 2 + 4 * int(N_k)
    total_system_modes = 2 * total_modes
    rho_th_bs = qt.tensor(*[qt.thermal_dm(Fk_dim, nth_bs) for _ in range(total_modes)])
    rho_before_bs = rho_in & rho_th_bs
    U_full = qt.tensor(*[qt.qeye(Fk_dim) for _ in range(total_system_modes)])
    dims = [Fk_dim] * total_system_modes
    for i in range(total_modes):
        U_bs = hamiltonian_np_beamsplitter(Fk_dim, T_bs, phi_bs)
        U_i = expand_operator(U_bs, dims=dims, targets=[i, i + total_modes])
        U_full = U_i @ U_full
    rho_out = U_full @ rho_before_bs @ U_full.dag()
    return U_full,rho_out


# Hamiltonian Digital Twin: apply loss and thermal noise using virtual beamsplitters to a system of multiple modes
def apply_digital_twin_model_single_mode(rho_in, mode_idx, Fk_dim, T, nth, phi):
    U_dt = U_bs = hamiltonian_np_beamsplitter(Fk_dim,T,phi)
    total_modes = len(rho_in.dims[0])
    rho_th = qt.thermal_dm(Fk_dim, nth)
    # Step 1: Add thermal mode
    rho_ext = rho_th & rho_in  # thermal comes first temporarily
    # Step 2: Permute to bring thermal to correct place
    # Thermal at position 0, signal at position mode_idx+1
    perm = [0, mode_idx + 1] + [j for j in range(1, total_modes + 1) if j - 1 != mode_idx]
    rho_perm = rho_ext.permute(perm)
    # Step 3: Build full operator
    ops = [qt.qeye(Fk_dim)] * (total_modes)
    ops[0] = U_dt
    U_full = qt.tensor(*ops)
    # Step 4: Apply unitary and trace out thermal (index 0)
    rho_after_dt = U_full @ rho_perm @ U_full.dag()
    rho_out = rho_after_dt.ptrace([j for j in range(total_modes + 1) if j != 0])
    return rho_out



def Fidelity(rho1, rho2):
    M = rho1.full() @ rho2.full()
    lambdas = eigvals(M)
    s = np.sqrt(lambdas)
    return np.real(np.sum(s) ** 2)



# Phototn Statisctics
def count_photon_pairs(rho, basis_dims, pair_indices):
    prob_pairs = {}
    total_dim = np.prod(basis_dims)
    rho_diag = np.real(np.diag(rho.full()))
    for flat_idx in range(total_dim):
        prob = rho_diag[flat_idx]
        if prob == 0:
            continue
        idx_tuple = np.unravel_index(flat_idx, basis_dims)
        pair_count = sum(min(int(idx_tuple[i]), int(idx_tuple[j])) for i, j in pair_indices)
        prob_pairs[pair_count] = prob_pairs.get(pair_count, 0) + prob
    return prob_pairs



def compute_mode_statistics(rho, Fk_dim, total_modes):
    i = total_modes - 1
    identity_ops = [qt.qeye(Fk_dim) for _ in range(total_modes)]
    number_ops = [qt.tensor(*(identity_ops[:i] + [qt.num(Fk_dim)] + identity_ops[i+1:])) for i in range(total_modes)]
    n_avg = qt.expect(number_ops[i], rho)
    n_sq_avg = qt.expect(number_ops[i] ** 2, rho)
    std = np.sqrt(n_sq_avg - n_avg ** 2)
    distribution = rho.ptrace(i).diag().real
    return n_avg, std, distribution


# single-mode second-order autocorrelation at zero delay
def compute_g2_auto_mode(rho, Fk_dim, modes):
    rho_mode = rho.ptrace(modes)
    a = qt.destroy(Fk_dim)
    n = np.real(qt.expect(a.dag() @ a, rho_mode))
    numerator = np.real(qt.expect(a.dag() @ a.dag() @ a @ a, rho_mode))
    return np.nan if n < 1e-12 else float(numerator / n**2)
    
# correlation between the two generated photons rather than thermal statistics within one mode
def compute_g2_cross(rho, Fk_dim, mode_s=0, mode_i=1):
    rho_si = rho.ptrace([mode_s, mode_i])
    a_s = qt.tensor(qt.destroy(Fk_dim), qt.qeye(Fk_dim))
    a_i = qt.tensor(qt.qeye(Fk_dim), qt.destroy(Fk_dim))
    n_s = np.real(qt.expect(a_s.dag() @ a_s, rho_si))
    n_i = np.real(qt.expect(a_i.dag() @ a_i, rho_si))
    num = np.real(qt.expect(a_s.dag() @ a_i.dag() @ a_i @ a_s, rho_si))
    return np.nan if n_s*n_i < 1e-12 else float(num/(n_s*n_i))
    

def plot_wigner(rho, Fk_dim, save_path=None):
    xmax = max(5, np.sqrt(Fk_dim) * 1.5)
    xvec = np.linspace(-xmax, xmax, 200)
    W = qt.wigner(rho, xvec, xvec)
    fig, ax = plt.subplots(figsize=(3,3))
    im = ax.contourf(xvec, xvec, W, levels=80, cmap="jet")
    ax.set_xlabel(r"$x_H$")
    ax.set_ylabel(r"$x_V$")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
