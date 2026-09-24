"""
Calcula o mapa de fidelidade no plano (sigma, L) para dois sítios cross-Kerr
idênticos, fótons contrapropagantes, agora COM um banho térmico de defasagem
pura acoplado a cada átomo, e salva os dados em data/heatmap para serem
plotados por heatmap_l_vs_sigma_plotar.py.

Modelo do banho
---------------
Cada átomo acopla a um banho térmico estacionário por sigma_z (b + b^dagger).
Na aproximação de Born-Markov (acoplamento fraco, k_B T >> hbar*gamma), o banho
entra apenas pela taxa de defasagem gamma_phi(T). Para um banho ôhmico,
J(w) = alpha*w, tem-se gamma_phi = 4*pi*alpha*k_B*T/hbar. Em todas as
expressões isso equivale a trocar gamma/2 -> gamma/2 + gamma_phi nos
denominadores; as taxas de acoplamento ao guia (fatores gamma no numerador)
não mudam.

O que é calculado
-----------------
Com o banho, o estado de saída dos fótons é misto:
    rho_out = |psi_coh><psi_coh| + rho_inc.
Este código calcula apenas a parte COERENTE, isto é, os elementos da matriz-S
média no banho, Tr_B[rho_B S]. Em particular:
  * t_site(w) deixa de ter módulo 1 (|t| < 1 para gamma_phi > 0);
  * f1 = sobreposição entre o fóton único espalhado (parte coerente) e o fóton
    único ideal (sem banho) após os dois sítios;
  * F  = sobreposição entre a parte coerente de dois fótons e o alvo ideal.
O alvo é sempre a evolução linear IDEAL (gamma_phi = 0): é a porta que se
quer implementar. Como a parte incoerente só pode somar termos não negativos
à fidelidade média, os valores F1_pi e F1_opt salvos são COTAS INFERIORES.

Método numérico
---------------
Usa as mesmas expressões e o mesmo método de integração de calcular.py: a
integral tripla é reduzida a convoluções 1D (o núcleo fatoriza, com K
dependendo só da energia total E = wa+wb), o que permite resolver sigma e
gamma simultaneamente.
"""
import os

import numpy as np
from scipy.integrate import simpson
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------
# Parâmetros físicos fixos
# ----------------------------------------------------------------------
gamma = 1.0           # gamma_1 = gamma_2 (fixo): acoplamento ao guia
chi = 1000.0          # chi_1 = chi_2 (fixo)
Delta = 0.0           # Delta_1 = Delta_2 (convenção)
omega0 = 0.0          # frequência central do pacote, relativa a Delta
phi = np.pi           # fase usada no cálculo de F1(phi)

# ----------------------------------------------------------------------
# Banho térmico de defasagem
# ----------------------------------------------------------------------
# Escolha UMA das opções:
#   (a) fixar gamma_phi diretamente (em unidades de gamma), ou
#   (b) dar alpha e T e deixar gamma_phi_de_T calcular (use_T = True).
use_T = False
gamma_phi = 0      # (a) taxa de defasagem, em unidades de gamma
alpha = 0.01          # (b) acoplamento ôhmico adimensional (precisa alpha << 1)
T_red = 1.0           # (b) temperatura reduzida k_B T / (hbar*gamma); precisa T_red >> 1


def gamma_phi_de_T(alpha, T_red):
    """Taxa de defasagem de um banho ôhmico: 4*pi*alpha*k_B*T/hbar, em unidades de gamma."""
    return 4 * np.pi * alpha * T_red * gamma


if use_T:
    gamma_phi = gamma_phi_de_T(alpha, T_red)

omega_c = Delta + omega0  # frequência central efetiva (derivada, não editar)

# ----------------------------------------------------------------------
# Varredura no plano (sigma, L)
# ----------------------------------------------------------------------
sigma_min = 0.01      # menor largura do pacote (eixo x, escala log)
sigma_max = 10.0      # maior largura do pacote
n_sigma = 200         # número de colunas do mapa

L_min = 0.0           # menor separação entre os sítios (eixo y, escala linear)
L_max = 6.0           # maior separação
n_L = 200             # número de linhas do mapa

# ----------------------------------------------------------------------
# Parâmetros numéricos da integração (mesma convenção de calcular.py)
# ----------------------------------------------------------------------
k_sigma = 9.0         # metade da janela de integração, em unidades de sigma
pts_per_width = 14    # pontos por menor escala relevante, min(sigma, gamma)
n_min = 2001          # número mínimo de pontos na grade
n_max = 600001        # teto de pontos (proteção de tempo/memória)

# ----------------------------------------------------------------------
# Parâmetros de saída
# ----------------------------------------------------------------------
outdir = os.path.join('data', 'heatmap')          # pasta onde os dados serão salvos
outfile = f'heatmap_chi{chi:g}_gamma{gamma:g}_omega{omega0:g}_gphi{gamma_phi:g}_thermal.npz'
quiet = False         # se True, não imprime o progresso da varredura


def Gamma(w, gphi=None):
    """Denominador de um sítio: meia largura total (radiativa + defasagem)
    mais a dessintonia. Com banho: gamma/2 + gamma_phi + i(Delta - w)."""
    if gphi is None:
        gphi = gamma_phi
    return gamma / 2 + gphi + 1j * (Delta - w)


def t_site(w, gphi=None):
    """Amplitude COERENTE de espalhamento de um sítio.

    t = gamma/Gamma - 1 = (gamma/2 - gamma_phi - i(Delta-w)) / (gamma/2 + gamma_phi + i(Delta-w)).
    Para gamma_phi = 0 isso coincide com conj(Gamma)/Gamma (|t| = 1). Para
    gamma_phi > 0, |t| < 1: a fração 1-|t|^2 do fóton sai incoerente
    (não se perde, mas não interfere com o alvo).
    """
    return gamma / Gamma(w, gphi) - 1.0


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0 / (2 * np.pi * sigma ** 2)) ** 0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def correction(w, L):
    """Conjugado do alvo de fóton único: a amplitude IDEAL (sem banho) de dois
    sítios, t_0^2, incluindo a fase de propagação e^{iwL} entre os sítios.

    O alvo é a evolução linear sem defasagem, porque é a operação que a porta
    deveria realizar. Sem banho, esse fator cancela exatamente a amplitude de
    fóton único (|t_0| = 1); com banho, sobra |t|^2 e um pequeno desvio de fase,
    que aparecem em f1."""
    return np.conj(t_site(w, 0.0)) ** 2 * np.exp(-1j * w * L)


def grade(sigma):
    """Grade de frequências e seu espaçamento.

    As quatro gaussianas confinam o integrando a |w-w_c| <~ k_sigma*sigma (o
    núcleo decai como 1/w⁴), então a janela é fixada só por sigma. Já o
    espaçamento precisa resolver a menor escala presente, min(sigma, gamma):
    é o que garante convergência tanto para sigma << gamma quanto sigma >> gamma.
    A defasagem só alarga as ressonâncias (gamma/2 -> gamma/2 + gamma_phi),
    então essa escolha continua suficiente.
    """
    half = k_sigma * sigma
    n = int(2 * half / (min(sigma, gamma) / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1   # ímpar: Simpson com nº par de intervalos
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


def compute_f1(sigma):
    """Sobreposição de fóton único com o alvo ideal, após os dois sítios:
    f1 = ∫ dw |xi(w)|^2 t(w)^2 conj(t_0(w))^2.
    Não depende de L (a fase de propagação cancela com a do alvo).
    |f1|^2 <= 1; a diferença mede a parte incoerente mais a distorção."""
    w, _ = grade(sigma)
    xi = xi_in(w, sigma)
    return simpson(np.abs(xi) ** 2 * t_site(w) ** 2 * np.conj(t_site(w, 0.0)) ** 2, x=w)


def compute_F(sigma, L):
    """
    F = <alvo|parte coerente de dois fótons> = f1^2 + G, com

    G = ∫∫∫ dwa dwb dna  xi(wa)xi(wb)xi(na)xi(nb) *
                          correction(wa) correction(wb) *
                          [termo 2 + termo 3](wa,wb,na,nb)
    onde nb = wa+wb-na (vem da delta).

    O termo linear f1^2 substitui o "1" do caso sem banho: cada fóton, mesmo
    sem interagir com o outro, já perde coerência nos dois sítios.

    Nos termos não lineares, o banho entra (i) em cada t_site, (ii) nos
    denominadores Gamma e (iii) no núcleo K, pela largura das duas excitações
    simultâneas: gamma -> gamma + 2*gamma_phi.

    Com E = wa+wb = na+nb, K depende só de E e cada fator restante depende de
    uma única frequência: a integral tripla vira uma integral 1D em E sobre o
    produto de duas convoluções.
    """
    w, h = grade(sigma)

    xi = xi_in(w, sigma)
    c = correction(w, L)
    eL = np.exp(1j * w * L)
    G = Gamma(w)
    t = t_site(w)

    E = np.linspace(2 * w[0], 2 * w[-1], 2 * len(w) - 1)   # energia total
    K = 1.0 / (1.0 + 2j * chi / (gamma + 2 * gamma_phi + 1j * (2 * Delta - E)))

    def conv(a, b):
        return fftconvolve(a, b) * h

    # Termo 2 (não linearidade no sítio 1). Com sítios idênticos (inclusive o
    # mesmo gamma_phi) o termo 3 (não linearidade no sítio 2) dá exatamente o
    # mesmo valor — as duas convoluções apenas trocam de ordem —, daí o fator 2.
    termo = -(1j * chi * gamma ** 2 / np.pi) * simpson(
        K * conv(xi * c * eL * t / G, xi * c / G)
          * conv(xi / G, xi * eL * t / G), x=E)

    f1 = compute_f1(sigma)
    return f1 ** 2 + 2 * termo


def fidelidades(f1, F, phi):
    """Fidelidade média da porta (d = 4) usando só a parte coerente, com
    amplitudes diagonais a = (1, f1, f1, e^{i phi} F):
        F_avg = (sum|a_i|^2 + |sum a_i|^2) / 20.
    Para f1 = 1 isso reduz a (6 + 3 Re(e^{i phi}F) + |F|^2)/10, a fórmula do caso
    sem banho. A parte incoerente só soma termos não negativos, então estes
    valores são cotas inferiores."""
    soma_mod = 1 + 2 * np.abs(f1) ** 2 + np.abs(F) ** 2
    F1_phi = (soma_mod + np.abs(1 + 2 * f1 + np.exp(1j * phi) * F) ** 2) / 20.0
    F1_opt = (soma_mod + (np.abs(1 + 2 * f1) + np.abs(F)) ** 2) / 20.0
    return F1_phi, F1_opt


def main():
    sigma_vals = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    L_vals = np.linspace(L_min, L_max, n_L)

    if not quiet:
        print(f"gamma_phi = {gamma_phi:g} (unidades de gamma)")

    f1_vals = np.array([compute_f1(s) for s in sigma_vals])   # não depende de L

    F_grid = np.zeros((len(L_vals), len(sigma_vals)), dtype=complex)
    for i, L in enumerate(L_vals):
        for j, sigma in enumerate(sigma_vals):
            F_grid[i, j] = compute_F(sigma, L)
        if not quiet:
            print(f"L={L:6.3f}  concluído ({i+1}/{len(L_vals)})")

    F_abs = np.abs(F_grid)
    if np.any(F_abs > 1.0 + 1e-6) and not quiet:
        print("AVISO: |F| > 1 em alguns pontos -- considere aumentar "
              "'pts_per_width' ou 'k_sigma'.")

    f1_grid = np.broadcast_to(f1_vals, F_grid.shape)
    F1_pi, F1_opt = fidelidades(f1_grid, F_grid, phi)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile)
    np.savez(
        outpath,
        sigmas=sigma_vals,
        Ls=L_vals,
        F_abs=F_abs,
        F_re=np.real(F_grid),
        F_im=np.imag(F_grid),
        f1_re=np.real(f1_vals),
        f1_im=np.imag(f1_vals),
        F1_pi=F1_pi,          # cota inferior (só parte coerente)
        F1_opt=F1_opt,        # cota inferior, fase condicional otimizada
        chi=chi, gamma=gamma, Delta=Delta, omega0=omega0, phi=phi,
        gamma_phi=gamma_phi, use_T=use_T, alpha=alpha, T_red=T_red,
        k_sigma=k_sigma, pts_per_width=pts_per_width,
    )
    if not quiet:
        print(f"Dados salvos em {outpath}")


if __name__ == '__main__':
    main()