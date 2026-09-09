
import re
import itertools
from collections import defaultdict

import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

'''--------State and density matrixes representations -----------------------------'''

def to_sympy_rounded(matrix, precision):
    return sp.Matrix(np.round(matrix, precision))

def generate_symbolic_state_or_rho(state, Fk_dim, precision, N_k, n_sp, mode_type):
    def extract_ket_bra(label):
        ket_match = re.search(r'\\left\|(.*?)\\right', label)
        bra_match = re.search(r'\\left\\langle(.*?)\\right', label)
        if ket_match and bra_match:
            return ket_match.group(1), bra_match.group(1)
        else:
            return None, None
    coeff_map = defaultdict(list)
    n_modes = 2 + int(N_k) * 4 + int(n_sp)
    mode_labels = []
    if mode_type == "polarized+spectural":
        mode_labels = ["H,f0", "V,f0"]
        for k in range(1, N_k + 1):
            mode_labels += [f"H,f{k}+", f"V,f{k}-", f"H,f{k}-", f"V,f{k}+"]
    elif mode_type == "polarized+spectural+spatial":
        mode_labels = ["H_i,f0", "V_i,f0", "H_s,f0", "V_s,f0"]
        for k in range(1, N_k + 1):
            mode_labels += [f"H_i,f{k}+", f"V_i,f{k}-", f"H_i,f{k}-", f"V_i,f{k}+",
                            f"H_s,f{k}+", f"V_s,f{k}-", f"H_s,f{k}-", f"V_s,f{k}+"]
    elif mode_type == "polarized+spectural+idler":
        mode_labels = ["H_i,f0", "V_i,f0"]
        for k in range(1, N_k + 1):
            mode_labels += [f"H_i,f{k}+", f"V_i,f{k}-", f"H_i,f{k}-", f"V_i,f{k}+"]
    elif mode_type == "polarized+spectural+signal":
        mode_labels = ["H_s,f0", "V_s,f0"]
        for k in range(1, N_k + 1):
            mode_labels += [f"H_s,f{k}+", f"V_s,f{k}-", f"H_s,f{k}-", f"V_s,f{k}+"]

    is_density_matrix = not (state.isket or state.isbra)
    for i, coeff in np.ndenumerate(state.full()):
        if abs(coeff) >= 10 ** (-2*precision):
            r = round(coeff.real, precision)
            im = round(coeff.imag, precision)
            coeff_key = (r, im)
            if is_density_matrix:
                row_basis = [(i[0] // Fk_dim ** k) % Fk_dim for k in range(n_modes)]
                col_basis = [(i[1] // Fk_dim ** k) % Fk_dim for k in range(n_modes)]
                row_label = ",".join([f"{row_basis[j]}_{{{mode_labels[j]}}}" for j in range(n_modes)])
                col_label = ",".join([f"{col_basis[j]}_{{{mode_labels[j]}}}" for j in range(n_modes)])
                label_str = f"\\left|{row_label}\\right\\rangle\\left\\langle{col_label}\\right|"
            else:
                basis_state = [(i[0] // Fk_dim ** k) % Fk_dim for k in range(n_modes)]
                label_str = ",".join([f"{basis_state[j]}_{{{mode_labels[j]}}}" for j in range(n_modes)])
                label_str = f"\\left|{label_str}\\right\\rangle"
            coeff_map[coeff_key].append(label_str)

    symbolic_terms = []
    used = set()
    coeff_items = list(coeff_map.items())

    for (r, im), basis_list in coeff_items:
        key = (r, im)
        if key in used:
            continue

        # Check for conjugate terms only if it's complex (not real or purely imaginary)
        is_conjugate_pair = (r, im) != (-r, -im) and (-r, -im) in coeff_map
        if is_conjugate_pair:
            basis1 = coeff_map[key]
            basis2 = coeff_map[(-r, -im)]
            paired = []
            used_pairs = set()

            for b1 in basis1:
                for b2 in basis2:
                    if (b1, b2) in used_pairs or (b2, b1) in used_pairs:
                        continue
                    b1_ket, b1_bra = extract_ket_bra(b1)
                    b2_ket, b2_bra = extract_ket_bra(b2)
                    if b1_ket is None or b2_ket is None:
                        continue
                    if b1_bra == b2_ket and b1_ket == b2_bra:
                        ordered = sorted([(b1_ket, b1_bra), (b2_ket, b2_bra)], reverse=True)
                        term1 = f"\\left|{ordered[0][0]}\\right\\rangle\\left\\langle{ordered[0][1]}\\right|"
                        term2 = f"\\left|{ordered[1][0]}\\right\\rangle\\left\\langle{ordered[1][1]}\\right|"
                        sign = "-" if im != 0 else "+"
                        combined = f"{term1} {sign} {term2}"
                        paired.append(combined)
                        used_pairs.add((b1, b2))

            if paired:
                coeff_value = f"{abs(im):.{precision}f}i" if im != 0 else f"{abs(r):.{precision}f}"
                term_body = " + \\\\\n".join(paired)
                term_str = f" + {coeff_value}\\left(\\begin{{aligned}}{term_body}\\end{{aligned}}\\right)"
                symbolic_terms.append(term_str)
                used.add(key)
                used.add((-r, -im))
            continue

        # Otherwise, this is a lone term (either real or non-paired complex)
        coeff_value = ""
        if im == 0:
            coeff_value = f"{r:+.{precision}f}"
        elif r == 0:
            coeff_value = f"{im:+.{precision}f}i"
        else:
            coeff_value = f"({r:+.{precision}f}{im:+.{precision}f}i)"

        prefix = "" if len(symbolic_terms) == 0 else ("  " if not coeff_value.startswith("-") else " ")
        if len(basis_list) == 1:
            term_str = f"{prefix}{coeff_value}{basis_list[0]}"
        else:
            grouped = " + \\\\\n".join(basis_list)
            term_str = f"{prefix}{coeff_value}\\left(\\begin{{aligned}}{grouped}\\end{{aligned}}\\right)"
        symbolic_terms.append(term_str)
        used.add(key)
    return "".join(symbolic_terms)



# Density matrix represenation
def plot_sparse_density_matrix(rho, total_modes, Fk_dim, N_k=0, threshold=1e-26,cmap_name = 'jet',save_path=None):
    dim = rho.shape[0]
    if N_k == 0:
        fig = plt.figure(figsize=(7, 9))
        fontsize_xticks, fontsize_yticks, fontsize_tick_params = 9, 9, 12
        pad_x, pad_y, pad_z = 6, 0, 3
        fig_title_size, colorbar_pad, colorbar_size = 9, 0.12, 11
        shrink_size, aspect_size, colorbar_labelsize = 0.25, 15, 12
    else:
        fig = plt.figure(figsize=(20, 24))
        fontsize_xticks, fontsize_yticks, fontsize_tick_params = 15, 15, 15
        pad_x, pad_y, pad_z = 27, 4, 5
        fig_title_size, colorbar_pad, colorbar_size = 15, 0.15, 20
        shrink_size, aspect_size, colorbar_labelsize = 0.25, 23, 15

    ax = fig.add_subplot(111, projection='3d')
    xpos, ypos, zpos, dx, dy, dz = [], [], [], [], [], []
    for i in range(dim):
        for j in range(dim):
            xpos.append(j)
            ypos.append(i)
            zpos.append(0)
            dx.append(0)
            dy.append(0)
            dz.append(np.abs(rho[i, j]))
    if not dz:
        dz = [0]

    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    colors = plt.colormaps[cmap_name](norm(dz))
    dx = dy = 0.6
    xpos = [x + (1 - dx) / 2 for x in xpos]
    ypos = [y + (1 - dy) / 2 for y in ypos]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=1)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, shrink=shrink_size, aspect=aspect_size, pad=1.0*colorbar_pad, ax=ax)
    cbar.set_label(r"$|\rho_{ij}|$", fontsize=colorbar_size)
    cbar.ax.tick_params(labelsize=colorbar_labelsize)

    basis_states = list(itertools.product(range(Fk_dim), repeat=total_modes))
    full_labels = [rf"$\left|{','.join(map(str, b))}\right\rangle$" for b in basis_states]

    if N_k == 0:
        xtick_labels = ytick_labels = full_labels
        xtick_positions = [x + 0.5 for x in range(len(full_labels))]
        ytick_positions = [y + 0.5 for y in range(len(full_labels))]
    else:
        xtick_labels, ytick_labels, xtick_positions, ytick_positions = [], [], [], []
        for i in range(dim):
            if np.any(np.abs(rho[i, :]) > threshold) or np.any(np.abs(rho[:, i]) > threshold):
                xtick_positions.append(i + 0.5)
                ytick_positions.append(i + 0.5)
                xtick_labels.append(full_labels[i])
                ytick_labels.append(full_labels[i])

    ax.set_xticks(xtick_positions)
    ax.set_yticks(ytick_positions)
    ax.set_xticklabels(xtick_labels, rotation=65, fontsize=fontsize_xticks, ha='center', va='center')
    ax.tick_params(axis='x', labelsize=fontsize_tick_params, pad=pad_x)
    ax.set_xlim(0, dim)
    ax.set_yticklabels(ytick_labels, fontsize=fontsize_yticks, ha='left', va='center')
    ax.tick_params(axis='y', labelsize=fontsize_tick_params, pad=pad_y)
    ax.set_ylim(0, dim)
    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_zticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax.tick_params(axis='z', labelsize=fontsize_tick_params, pad=pad_z)
    #ax.set_zlabel("Probability", fontsize=fontsize_tick_params, labelpad=1 * pad_z)
    ax.set_title("Mixed-state Density Matrix", fontsize=2 * fig_title_size)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)