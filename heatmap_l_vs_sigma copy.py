import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros físicos fixos
# ----------------------------------------------------------------------
gamma = 4.5          # gamma_1 = gamma_2 (fixo)
Delta = 0.0           # Delta_1 = Delta_2 (convenção)
omega0 = 1.1
omega_c = Delta + omega0
chi = 5.0             # fixo

def Gamma(w):
    return gamma/2 + 1j*(Delta - w)

def phi(w):
    """Fase de espalhamento de um único átomo (|phi|=1)."""
    return np.conj(Gamma(w)) / Gamma(w)

def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0/(2*np.pi*sigma**2))**0.25 * np.exp(-(w-omega_c)**2/(4*sigma**2))

def Kfac(oa, ob):
    return 1.0/(1.0 + 2j*chi/(Gamma(oa)+Gamma(ob)))

# ----------------------------------------------------------------------
# Integral tripla para o termo de "Cross" (interferência target x interação)
# ----------------------------------------------------------------------
def compute_overlap(sigma, L, n=10, k_sigma=6.0, k_gamma=8.0):
    """
    Overlap = 1 + Cross(sigma, L)

    Cross = -i*chi*gamma^2/pi *
            ∫∫∫ dwa dwb dna  xi*(wa) xi*(wb) xi(na) xi(nb) *
                e^{-i(wa+wb)L} phi*(wa)^2 phi*(wb)^2 * Kfac(wa,wb) /
                (Gamma(wa)Gamma(wb)Gamma(na)Gamma(nb)) *
                [ e^{i(wa+nb)L} phi(wa) phi(nb) + e^{i(na+wb)L} phi(na) phi(wb) ]

    com nb = wa + wb - na (da delta).
    """
    half = max(k_sigma*sigma, k_gamma*gamma)
    grid = np.linspace(omega_c-half, omega_c+half, n)

    OA, OB, NA = np.meshgrid(grid, grid, grid, indexing='ij')
    NB = OA + OB - NA

    bracket = (np.exp(1j*(OA+NB)*L) * phi(OA) * phi(NB) +
               np.exp(1j*(NA+OB)*L) * phi(NA) * phi(OB))

    integrand = (np.conj(xi_in(OA, sigma)) * np.conj(xi_in(OB, sigma)) *
                 xi_in(NA, sigma) * xi_in(NB, sigma) *
                 np.exp(-1j*(OA+OB)*L) * np.conj(phi(OA))**2 * np.conj(phi(OB))**2 *
                 Kfac(OA, OB) /
                 (Gamma(OA)*Gamma(OB)*Gamma(NA)*Gamma(NB)) *
                 bracket)

    I_na = simpson(integrand, x=grid, axis=2)
    I_ob = simpson(I_na,      x=grid, axis=1)
    I    = simpson(I_ob,      x=grid, axis=0)

    Cross = -1j*chi*gamma**2/np.pi * I
    return 1.0 + Cross

# ----------------------------------------------------------------------
# Varredura no espaço de parâmetros (sigma em escala log, L em escala linear)
# ----------------------------------------------------------------------
sigma_vals = np.logspace(np.log10(0.1), np.log10(10), 200)   # eixo x -- log
L_vals     = np.linspace(0, 6.0, 200)                      # eixo y -- linear

overlap_grid = np.zeros((len(L_vals), len(sigma_vals)), dtype=complex)

for i, L in enumerate(L_vals):
    for j, sigma in enumerate(sigma_vals):
        overlap_grid[i, j] = compute_overlap(sigma, L, n=40)
    print(f"L={L:6.3f}  concluído ({i+1}/{len(L_vals)})")

abs_overlap = np.abs(overlap_grid)

if np.any(abs_overlap > 1.0 + 1e-3):
    print("AVISO: |overlap| > 1 em alguns pontos -- considere aumentar 'n' "
          "(resolução da malha) para melhorar a convergência numérica.")

# ----------------------------------------------------------------------
# Heatmap (x em escala log -> usar pcolormesh, não imshow com extent linear)
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
im = plt.pcolormesh(sigma_vals, L_vals, abs_overlap,
                     cmap='viridis', vmin=0, vmax=1, shading='auto')
plt.xscale('log')
plt.colorbar(im, label=r'$|\langle \mathrm{target}|\phi_{\mathrm{espalhado}}\rangle|$')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$L$')
plt.title(rf'Overlap alvo-espalhado  ($\gamma={gamma}$, $\chi={chi}$, $\omega_0={omega0}$)')
plt.tight_layout()
plt.savefig('heatmap_overlap_sigma_L.svg', dpi=300, bbox_inches='tight')
plt.show()