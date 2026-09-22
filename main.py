import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros (Fig. 5 de Brod & Combes, dois sítios, contrapropagante)
# ----------------------------------------------------------------------
chi1 = chi2 = 5.0
gamma1 = gamma2 = 4.5
Delta1 = Delta2 = 0.0
omega0 = 1.1
omega_c = Delta1 + omega0
L = 0.0   # teste solicitado

def Gamma1(w): return gamma1/2 + 1j*(Delta1 - w)
def Gamma2(w): return gamma2/2 + 1j*(Delta2 - w)

def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0/(2*np.pi*sigma**2))**0.25 * np.exp(-(w-omega_c)**2/(4*sigma**2))

def phase1(w):
    """Fase de fóton único de dois sítios: Γ1*Γ2*/(Γ1 Γ2). |phase1|=1."""
    return (np.conj(Gamma1(w))*np.conj(Gamma2(w))) / (Gamma1(w)*Gamma2(w))

def correction(w):
    """conj(phase1(w)) = 1/phase1(w), usada para remover a deformação
    de fóton único: S -> S1†(ωa)S1†(ωb) S."""
    return np.conj(phase1(w))

def K1(oa, ob):
    return 1.0/(1.0 + 2j*chi1/(Gamma1(oa)+Gamma1(ob)))

def K2(oa, ob):
    return 1.0/(1.0 + 2j*chi2/(Gamma2(oa)+Gamma2(ob)))

def term2_bracket(oa, ob, na, nb):
    """Termo 2 da expressão de S (sem a delta), com a fase e^{i(oa+nb)L}."""
    phase_L = np.exp(1j*(oa+nb)*L)
    return -phase_L * (1j*chi1*gamma1**2/np.pi) \
           * (np.conj(Gamma2(oa))*np.conj(Gamma2(nb))/(Gamma2(oa)*Gamma2(nb))) \
           * K1(oa, ob) \
           / (Gamma1(oa)*Gamma1(ob)*Gamma1(na)*Gamma1(nb))

def term3_bracket(oa, ob, na, nb):
    """Termo 3 da expressão de S (sem a delta), com a fase e^{i(na+ob)L}."""
    phase_L = np.exp(1j*(na+ob)*L)
    return -phase_L * (1j*chi2*gamma2**2/np.pi) \
           * (np.conj(Gamma1(na))*np.conj(Gamma1(ob))/(Gamma1(na)*Gamma1(ob))) \
           * K2(oa, ob) \
           / (Gamma2(oa)*Gamma2(ob)*Gamma2(na)*Gamma2(nb))

# ----------------------------------------------------------------------
# Integral tripla (a delta δ(ωa+ωb-νa-νb) elimina ωb = νa+νb-ωa)
# ----------------------------------------------------------------------
def compute_G(sigma, n=110, k_sigma=6.0, k_gamma=8.0):
    """
    F = 1 + G, com
    G = ∫∫∫ dνa dνb dωa  ξ(νa)ξ(νb)ξ(ωa)ξ(ωb) *
                          correction(ωa) correction(ωb) *
                          [term2_bracket + term3_bracket](ωa,ωb,νa,νb)
    onde ωb = νa+νb-ωa (vem da delta).
    """
    half = max(k_sigma*sigma, k_gamma*max(gamma1, gamma2))
    grid = np.linspace(omega_c-half, omega_c+half, n)

    NA, NB, OA = np.meshgrid(grid, grid, grid, indexing='ij')
    OB = NA + NB - OA

    bracket = term2_bracket(OA, OB, NA, NB) + term3_bracket(OA, OB, NA, NB)

    integrand = (xi_in(NA, sigma) * xi_in(NB, sigma) *
                 xi_in(OA, sigma) * xi_in(OB, sigma) *
                 correction(OA) * correction(OB) *
                 bracket)

    I_oa = simpson(integrand, x=grid, axis=2)
    I_nb = simpson(I_oa,      x=grid, axis=1)
    I    = simpson(I_nb,      x=grid, axis=0)
    return I

def F1_fidelity(sigma, phi=np.pi, **kwargs):
    G = compute_G(sigma, **kwargs)
    F = 1.0 + G

    if np.abs(F) > 1.0 + 1e-6:
        print(f"AVISO: |F|={np.abs(F):.4f} > 1 em sigma={sigma:.3f} "
              f"(considere aumentar 'n' ou 'k_gamma')")

    F1_phi = (6 + 3*np.real(np.exp(1j*phi)*F) + np.abs(F)**2)/10.0
    F1_opt = (6 + 3*np.abs(F) + np.abs(F)**2)/10.0
    return F1_phi, F1_opt, F

# ----------------------------------------------------------------------
# Varredura em sigma e plot
# ----------------------------------------------------------------------
sigmas = np.logspace(np.log10(0.1), np.log10(100), 35)

F1_pi_list, F1_opt_list = [], []
for s in sigmas:
    f1_pi, f1_opt, F = F1_fidelity(s, phi=np.pi, n=110)
    F1_pi_list.append(f1_pi)
    F1_opt_list.append(f1_opt)
    print(f"sigma={s:8.3f}   |F|={np.abs(F):.4f}   F1(pi)={f1_pi:.4f}   F1(opt)={f1_opt:.4f}")

plt.figure(figsize=(7,5))
plt.semilogx(sigmas, F1_pi_list,  'o-',  color='purple',
             label=r'$F_1(\phi=\pi)$ — 2 sítios, contrapropagante')
plt.semilogx(sigmas, F1_opt_list, '--',  color='purple',
             label=r'$F_1(\phi_{opt})$ — 2 sítios, contrapropagante')
plt.xlabel(r'Largura do pacote de onda $\sigma$')
plt.ylabel('Fidelidade $F_1$')
plt.title(r'$\chi=5,\ \gamma=4.5,\ \omega_0=1.1,\ L=0$')
plt.ylim(0.3, 1.02)
plt.axhline(1.0, color='gray', lw=0.8, ls=':')
plt.legend()
# plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig('fidelity_counterprop_L0.png', dpi=150)
plt.show()