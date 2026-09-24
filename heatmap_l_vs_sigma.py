"""
Heatmap da fidelidade média da porta CZ no plano (sigma, L) para dois sítios
cross-Kerr idênticos, fótons contrapropagantes.

O produto interno F = <target|espalhado> = 1 + G é só um passo intermediário;
a fidelidade é

    F1(phi) = [6 + 3 Re(e^{i phi} F) + |F|^2] / 10

(README, secao 1: [|Tr(U*A)|^2 + Tr(A*A)]/[d(d+1)] com d = 4 e
A = diag(1,1,1,F)). O termo Tr(A*A) = 3 + |F|^2, em vez de d = 4, é o que
trata o vazamento para fora do subespaço computacional quando |F| < 1.

Usa as mesmas expressões e o mesmo método de integração de calcular.py:
a integral tripla é reduzida a convoluções 1D (o núcleo fatoriza, com K
dependendo só da energia total E = wa+wb), o que permite resolver sigma e
gamma simultaneamente.
"""
import numpy as np
from scipy.integrate import simpson
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros físicos fixos
# ----------------------------------------------------------------------
gamma = 1.0           # gamma_1 = gamma_2 (fixo)
chi = 1000.0          # chi_1 = chi_2 (fixo)
Delta = 0.0           # Delta_1 = Delta_2 (convenção)
omega0 = 0.0          # frequência central do pacote, relativa a Delta
phi = np.pi           # fase da porta CZ usada em F1(phi)

omega_c = Delta + omega0  # frequência central efetiva (derivada, não editar)

# ----------------------------------------------------------------------
# Varredura no plano (sigma, L)
# ----------------------------------------------------------------------
sigma_min = 0.01       # menor largura do pacote (eixo x, escala log)
sigma_max = 10.0      # maior largura do pacote
n_sigma = 200         # número de colunas do heatmap

L_min = 0.0           # menor separação entre os sítios (eixo y, escala linear)
L_max = 6.0           # maior separação
n_L = 200             # número de linhas do heatmap

# ----------------------------------------------------------------------
# Parâmetros numéricos da integração (mesma convenção de calcular.py)
# ----------------------------------------------------------------------
k_sigma = 9.0         # metade da janela de integração, em unidades de sigma
pts_per_width = 14    # pontos por menor escala relevante, min(sigma, gamma)
n_min = 2001          # número mínimo de pontos na grade
n_max = 600001        # teto de pontos (proteção de tempo/memória)

# ----------------------------------------------------------------------
# O que colorir
# ----------------------------------------------------------------------
field = 'F1_pi'       # 'F1_pi'  = fidelidade da porta com a fase `phi` acima
                      # 'F1_opt' = fidelidade com a fase ótima (a fase phi que
                      #            maximiza F1, isto é Re(e^{i phi} F) = |F|)
                      # 'F_abs'  = só o módulo do produto interno |F|, que era
                      #            o que este arquivo plotava antes
vmin, vmax = 0.4, 1.0 # faixa da barra de cores. Para F1 o piso é 0.4 (F = 1,
                      # nenhuma fase condicional, ou seja, sem porta) e o teto
                      # é 1.0 (CZ perfeito). Para 'F_abs' use 0.0, 1.0.

# ----------------------------------------------------------------------
# Parâmetros da figura
# ----------------------------------------------------------------------
outfile = fr'heatmap_{field}_sigma_L.svg'  # arquivo de imagem de saída
dpi = 300
figsize = (8, 6)
cmap = 'viridis'
show = True                               # abre a janela interativa além de salvar
quiet = False                             # se True, não imprime o progresso


def Gamma(w):
    return gamma / 2 + 1j * (Delta - w)


def t_site(w):
    """Fase de espalhamento de um único sítio (|t|=1)."""
    return np.conj(Gamma(w)) / Gamma(w)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0 / (2 * np.pi * sigma ** 2)) ** 0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def correction(w, L):
    """Remove a amplitude completa de fóton único (dois sítios): conj(t^2),
    incluindo a fase de propagação e^{iwL} entre os sítios. Sem ela a
    referência linear e os termos não lineares ficam em convenções
    diferentes e S deixa de ser unitária para L != 0."""
    return np.conj(t_site(w)) ** 2 * np.exp(-1j * w * L)


def grade(sigma):
    """Grade de frequências e seu espaçamento.

    As quatro gaussianas confinam o integrando a |w-w_c| <~ k_sigma*sigma (o
    núcleo decai como 1/w⁴), então a janela é fixada só por sigma. Já o
    espaçamento precisa resolver a menor escala presente, min(sigma, gamma):
    é o que garante convergência tanto para sigma << gamma quanto sigma >> gamma.
    """
    half = k_sigma * sigma
    n = int(2 * half / (min(sigma, gamma) / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1   # ímpar: Simpson com nº par de intervalos
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


def compute_F(sigma, L):
    """
    F = 1 + G, com
    G = ∫∫∫ dwa dwb dna  xi(wa)xi(wb)xi(na)xi(nb) *
                          correction(wa) correction(wb) *
                          [termo 2 + termo 3](wa,wb,na,nb)
    onde nb = wa+wb-na (vem da delta).

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
    K = 1.0 / (1.0 + 2j * chi / (gamma + 1j * (2 * Delta - E)))

    def conv(a, b):
        return fftconvolve(a, b) * h

    # Termo 2 (não linearidade no sítio 1). Com sítios idênticos o termo 3
    # (não linearidade no sítio 2) dá exatamente o mesmo valor — as duas
    # convoluções apenas trocam de ordem —, daí o fator 2.
    termo = -(1j * chi * gamma ** 2 / np.pi) * simpson(
        K * conv(xi * c * eL * t / G, xi * c / G)
          * conv(xi / G, xi * eL * t / G), x=E)

    return 1.0 + 2 * termo


def fidelidade(F):
    """Fidelidade média da porta a partir do produto interno F.

    F1 = [|Tr(U† A)|^2 + Tr(A† A)] / [d(d+1)] com d = 4 e A = diag(1,1,1,F),
    ou seja F1(phi) = [6 + 3 Re(e^{i phi} F) + |F|^2]/10 (README, secao 1).
    Devolve (F1(phi), F1(phi_opt), |F|); phi_opt é a fase que maximiza F1,
    para a qual Re(e^{i phi} F) = |F|.
    """
    F_abs = np.abs(F)
    F1_phi = (6 + 3 * np.real(np.exp(1j * phi) * F) + F_abs ** 2) / 10.0
    F1_opt = (6 + 3 * F_abs + F_abs ** 2) / 10.0
    return F1_phi, F1_opt, F_abs


def field_label():
    """Rótulo da barra de cores para o `field` escolhido."""
    if field == 'F1_pi':
        r = phi / np.pi
        fase = '0' if r == 0 else (r'\pi' if r == 1 else rf'{r:g}\pi')
        return rf'$F_1(\phi={fase})$'
    return {'F1_opt': r'$F_1(\phi_{\mathrm{opt}})$',
            'F_abs': r'$|F|$'}.get(field, field)


def main():
    sigma_vals = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    L_vals = np.linspace(L_min, L_max, n_L)

    F_grid = np.zeros((len(L_vals), len(sigma_vals)), dtype=complex)
    for i, L in enumerate(L_vals):
        for j, sigma in enumerate(sigma_vals):
            F_grid[i, j] = compute_F(sigma, L)
        if not quiet:
            print(f"L={L:6.3f}  concluído ({i+1}/{len(L_vals)})")

    F1_pi, F1_opt, abs_F = fidelidade(F_grid)
    if np.any(abs_F > 1.0 + 1e-6) and not quiet:
        print("AVISO: |F| > 1 em alguns pontos -- considere aumentar "
              "'pts_per_width' ou 'k_sigma'.")

    z = {'F1_pi': F1_pi, 'F1_opt': F1_opt, 'F_abs': abs_F}[field]
    if not quiet:
        print(f"{field}: min={z.min():.4f}  max={z.max():.4f}")

    plt.figure(figsize=figsize)
    im = plt.pcolormesh(sigma_vals, L_vals, z, cmap=cmap,
                    vmin=vmin, vmax=vmax, shading='auto',
                    edgecolors='face', linewidth=0, antialiased=False,  rasterized=True)
    plt.xscale('log')
    plt.colorbar(im, label=field_label())
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'$L$')
    plt.title(rf'$\chi={chi:g},\ \gamma={gamma:g},\ \omega_0={omega0:g}$')
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return sigma_vals, L_vals, F_grid


if __name__ == '__main__':
    main()
