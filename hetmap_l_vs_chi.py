import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros físicos fixos
# ----------------------------------------------------------------------
gamma = 4.5          # gamma_1 = gamma_2 (fixo, conforme pedido)
Delta = 0.0           # Delta_1 = Delta_2
omega0 = 1.1
omega_c = Delta + omega0

# Largura do pacote de onda -- NÃO foi especificada; assumo sigma=1.0
# (próxima do pico de fidelidade observado nos gráficos anteriores).
# Ajuste livremente.
sigma = 1.0

def Gamma(w):
    return gamma/2 + 1j*(Delta - w)

def phi(w):
    """Fase de espalhamento de um único átomo (|phi|=1)."""
    return np.conj(Gamma(w)) / Gamma(w)

def xi_in(w):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0/(2*np.pi*sigma**2))**0.25 * np.exp(-(w-omega_c)**2/(4*sigma**2))

def Kfac(oa, ob, chi):
    return 1.0/(1.0 + 2j*chi/(Gamma(oa)+Gamma(ob)))

# ----------------------------------------------------------------------
# Integral tripla para o termo de "Cross" (interferência target x interação)
# ----------------------------------------------------------------------
def compute_overlap(chi, L, n=40, k_sigma=6.0, k_gamma=8.0):
    """
    Overlap = 1 + Cross(chi, L)

    Cross = -i*chi*gamma^2/pi *
            ∫∫∫ dwa dwb dna  xi*(wa) xi*(wb) xi(na) xi(nb) *
                e^{-i(wa+wb)L} phi*(wa)^2 phi*(wb)^2 * Kfac(wa,wb;chi) /
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

    integrand = (np.conj(xi_in(OA)) * np.conj(xi_in(OB)) *
                 xi_in(NA) * xi_in(NB) *
                 np.exp(-1j*(OA+OB)*L) * np.conj(phi(OA))**2 * np.conj(phi(OB))**2 *
                 Kfac(OA, OB, chi) /
                 (Gamma(OA)*Gamma(OB)*Gamma(NA)*Gamma(NB)) *
                 bracket)

    I_na = simpson(integrand, x=grid, axis=2)
    I_ob = simpson(I_na,      x=grid, axis=1)
    I    = simpson(I_ob,      x=grid, axis=0)

    Cross = -1j*chi*gamma**2/np.pi * I
    return 1.0 + Cross

# ----------------------------------------------------------------------
# Varredura no espaço de parâmetros (chi, L)
# ----------------------------------------------------------------------
chi_vals = np.linspace(0.1, 10, 40)     # eixo x -- ajuste o intervalo se necessário
L_vals   = np.linspace(0, 2.0, 40)   # eixo y -- ajuste o intervalo se necessário

overlap_grid = np.zeros((len(L_vals), len(chi_vals)), dtype=complex)

for i, L in enumerate(L_vals):
    for j, chi in enumerate(chi_vals):
        overlap_grid[i, j] = compute_overlap(chi, L, n=40)
    print(f"L={L:6.3f}  concluído ({i+1}/{len(L_vals)})")

abs_overlap = np.abs(overlap_grid)

# aviso de unitariedade (overlap nao pode exceder 1 em módulo)
if np.any(abs_overlap > 1.0 + 1e-3):
    print("AVISO: |overlap| > 1 em alguns pontos -- considere aumentar 'n' "
          "(resolução da malha) para melhorar a convergência numérica.")

# ----------------------------------------------------------------------
# Heatmap
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
extent = [chi_vals.min(), chi_vals.max(), L_vals.min(), L_vals.max()]
im = plt.imshow(abs_overlap, extent=extent, origin='lower', aspect='auto',
                 cmap='viridis', vmin=0, vmax=1)
plt.colorbar(im, label=r'$|\langle \mathrm{target}|\phi_{\mathrm{espalhado}}\rangle|$')
plt.xlabel(r'$\chi$')
plt.ylabel(r'$L$')
plt.title(rf'Overlap alvo-espalhado  ($\gamma={gamma}$, $\sigma={sigma}$, $\omega_0={omega0}$)')
plt.tight_layout()
plt.savefig('heatmap_overlap_chi_L.png', dpi=150)
plt.show()