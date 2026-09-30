"""
Heatmap da amplitude do par espalhado no plano das frequências de SAÍDA,

    psi_out(wa, wb) = <wa wb| S |xi xi>,

para N sítios cross-Kerr idênticos, fótons contrapropagantes, com o pacote
de entrada fixo (centrado em omega0, largura sigma).

Generaliza heatmap_omegaa_omegab.py (N=2) usando a matriz S de N sítios, a
mesma de fidelidade_sigma/calcular_sigma_n_sitios.py:

    S = <ωa-|νa+><ωb-|νb+>
        - i(χγ²/π) K(νa,νb) δ(ωa+ωb-νa-νb) / [Γ(νb)Γ(νa)Γ(ωb)Γ(ωa)]
          × Σ_{j=1}^{N} A^{N-j} B^{j-1},

    A = a(ωa) a(νb),  B = a(ωb) a(νa),  a(ω) = t(ω) e^{iωL}
    <ω-|ν+> = S1(ω) δ(ω-ν),  S1(ω) = e^{i(N-1)ωL} t(ω)^N

A delta de energia deixa uma única integral em nu_a, que para cada termo j é
uma convolução 1D avaliada em E = wa + wb:

    psi_out = xi(wa) xi(wb) S1(wa) S1(wb)                                [linear]
            - (i chi gamma^2/pi) K(E) / (Gamma(wa) Gamma(wb))
              × Σ_j a(wa)^{N-j} a(wb)^{j-1} P_j(E)                      [não linear]

    P_j(E) = int dnu  xi(nu) a(nu)^{j-1} xi(E-nu) a(E-nu)^{N-j}
                      / (Gamma(nu) Gamma(E-nu))
"""
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------
# Parâmetros físicos fixos
# ----------------------------------------------------------------------
N = 10                 # número de sítios
gamma = 1.0           # taxa de decaimento (igual em todos os sítios)
chi = 1000.0          # acoplamento não-linear (igual em todos os sítios)
Delta = 0.0           # dessintonia (igual em todos os sítios)
omega0 = 0.0          # frequência central do pacote de entrada, relativa a Delta
L = 0                 # separação entre sítios vizinhos
sigma = 1.0           # largura do pacote de entrada

omega_c = Delta + omega0  # frequência central efetiva (derivada, não editar)

# ----------------------------------------------------------------------
# Plano (wa, wb) a ser desenhado
# ----------------------------------------------------------------------
omega_min = -3.0     # limite inferior dos dois eixos
omega_max = 3.0      # limite superior dos dois eixos
n_omega = 2000        # pontos por eixo do heatmap

# ----------------------------------------------------------------------
# Parâmetros numéricos das convoluções P_j(E)
# ----------------------------------------------------------------------
k_sigma = 9.0         # metade da janela de integração, em unidades de sigma
pts_per_width = 14    # pontos por menor escala relevante, min(sigma, gamma, 1/(N L))
n_min = 2001          # número mínimo de pontos na grade
n_max = 600001        # teto de pontos (proteção de tempo/memória)

# ----------------------------------------------------------------------
# O que colorir
# ----------------------------------------------------------------------
field = 'abs'         # 'abs', 'abs_nl' (só a parte não linear), 'real', 'imag' ou 'phase'
aplicar_correcao = True   # remove a fase de fóton único (não altera 'abs'/'abs_nl')
vmin, vmax = 0.0, None    # faixa da barra de cores; vmin=0 prende o preto no zero.
                          # Para 'real'/'imag'/'phase' (que têm valores negativos),
                          # use vmin=None ou uma faixa simétrica.

# ----------------------------------------------------------------------
# Cores do heatmap
# ----------------------------------------------------------------------
# Rampa sequencial do menor ao maior valor: começa no preto (0.0) e sobe até um
# dourado claro, passando por violeta. Os passos de luminosidade são uniformes
# (OKLab L de 0 a 0.99, passo 0.122), o que faz a cor codificar magnitude de
# forma monótona — e legível em tons de cinza e para daltônicos, já que é a
# luminosidade que carrega a informação. Edite os hex mantendo escuro -> claro.
cores = ['#000000', '#05050F', '#211546', '#562369', '#943A65',
         '#C75E50', '#E68F4D', '#F5C578', '#FDF6DC']
# alternativa fria, mesma família da curva teal (menos destaque nos picos):
# cores = ['#000000', '#030B11', '#0C2C44', '#1C5573', '#3B809B',
#          '#6DACBF', '#B1D6DE', '#FBFDFD']
cores_invertidas = False   # True inverte a rampa (fundo claro, lóbulos escuros)
n_niveis = 256             # níveis interpolados entre as cores acima
cmap = None                # nome de um colormap do matplotlib ('magma', 'cividis',
                           # 'twilight' para 'phase'); se definido, ignora `cores`

# ----------------------------------------------------------------------
# Parâmetros da figura
# ----------------------------------------------------------------------
outfile = fr'heatmap_omegaa_omegab_N{N}_l{L}_sigma_{sigma}.svg'   # arquivo de imagem de saída
dpi = 300
figsize = (7, 6)
xlabel = r'$\omega_a$'
ylabel = r'$\omega_b$'
title = None          # título; se None, é gerado a partir dos parâmetros
show = True           # abre a janela interativa além de salvar

FIELD_LABELS = {
    'abs': r'$|\psi_{\mathrm{out}}(\omega_a,\omega_b)|$',
    'abs_nl': r'$|\psi_{\mathrm{out}}^{\mathrm{n\tilde ao\ linear}}(\omega_a,\omega_b)|$',
    'real': r'$\mathrm{Re}\,\psi_{\mathrm{out}}(\omega_a,\omega_b)$',
    'imag': r'$\mathrm{Im}\,\psi_{\mathrm{out}}(\omega_a,\omega_b)$',
    'phase': r'$\arg\,\psi_{\mathrm{out}}(\omega_a,\omega_b)$',
}


def Gamma(w):
    return gamma / 2 + 1j * (Delta - w)


def t_site(w):
    """Fase de espalhamento de um único sítio (|t|=1)."""
    return np.conj(Gamma(w)) / Gamma(w)


def a_site(w):
    """Fase de um sítio mais a propagação até o próximo: t(w) e^{iwL}."""
    return t_site(w) * np.exp(1j * w * L)


def S1(w):
    """Amplitude completa de um fóton: N sítios mais a propagação entre eles."""
    return t_site(w) ** N * np.exp(1j * (N - 1) * w * L)


def xi_in(w):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0 / (2 * np.pi * sigma ** 2)) ** 0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def grade():
    """Grade de frequências para as convoluções P_j(E), e seu espaçamento.

    As gaussianas confinam o integrando a |w-w_c| <~ k_sigma*sigma; o
    espaçamento resolve a menor escala presente: sigma, gamma e, para L != 0,
    o período das fases e^{iwL} acumuladas ao longo dos N sítios.
    """
    half = k_sigma * sigma
    scales = [sigma, gamma]
    if L != 0:
        scales.append(1.0 / (N * abs(L)))
    n = int(2 * half / (min(scales) / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


def psi_out(WA, WB):
    """Amplitude do par espalhado nas frequências de saída (WA, WB).

    Devolve (total, parte não linear). Com `aplicar_correcao`, ambas são
    medidas em relação ao alvo S1 x S1 (o que só muda a fase: |S1| = 1).
    """
    w, h = grade()
    base = xi_in(w) / Gamma(w)
    a = a_site(w)
    E_grid = np.linspace(2 * w[0], 2 * w[-1], 2 * len(w) - 1)

    E = WA + WB
    aA, aB = a_site(WA), a_site(WB)

    soma = np.zeros_like(E, dtype=complex)
    for j in range(1, N + 1):
        # P_j(E) = (C * D)(E), com C = xi a^{j-1}/Gamma e D = xi a^{N-j}/Gamma
        P_grid = fftconvolve(base * a ** (j - 1), base * a ** (N - j)) * h
        P = (np.interp(E, E_grid, P_grid.real, left=0.0, right=0.0)
             + 1j * np.interp(E, E_grid, P_grid.imag, left=0.0, right=0.0))
        soma += aA ** (N - j) * aB ** (j - 1) * P

    K = 1.0 / (1.0 + 2j * chi / (gamma + 1j * (2 * Delta - E)))

    linear = xi_in(WA) * xi_in(WB) * S1(WA) * S1(WB)
    nao_linear = -(1j * chi * gamma ** 2 / np.pi) * K * soma / (Gamma(WA) * Gamma(WB))

    if aplicar_correcao:
        c = np.conj(S1(WA)) * np.conj(S1(WB))
        linear, nao_linear = linear * c, nao_linear * c
    return linear + nao_linear, nao_linear


def main():
    eixo = np.linspace(omega_min, omega_max, n_omega)
    WA, WB = np.meshgrid(eixo, eixo, indexing='xy')   # WA no eixo x, WB no eixo y

    total, nao_linear = psi_out(WA, WB)
    z = {'abs': np.abs(total),
         'abs_nl': np.abs(nao_linear),
         'real': np.real(total),
         'imag': np.imag(total),
         'phase': np.angle(total)}[field]

    plot_title = title if title is not None else (
        rf'$N={N},\ \chi={chi:g},\ \gamma={gamma:g},\ \omega_0={omega0:g},\ '
        rf'L={L:g},\ \sigma={sigma:g}$')

    if cmap is not None:
        mapa = plt.get_cmap(cmap)
    else:
        lista = cores[::-1] if cores_invertidas else cores
        mapa = LinearSegmentedColormap.from_list('heatmap', lista, N=n_niveis)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(eixo, eixo, z, cmap=mapa, vmin=vmin, vmax=vmax,
                       shading='auto', rasterized=True)
    fig.colorbar(im, ax=ax, label=FIELD_LABELS.get(field, field))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    if plot_title:
        ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return eixo, total


if __name__ == '__main__':
    main()
