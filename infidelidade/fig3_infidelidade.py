"""
Reprodução da Fig. 3 de Brod & Combes, "A passive CPHASE gate via cross-Kerr
nonlinearities" (arXiv:1604.04278), com a matriz S de N sítios que inclui o
deslocamento de fase e^{iωL} (a mesma de fidelidade_sigma/calcular_sigma_n_sitios.py).

    Fig. 3(a): F_avg(φ=π) vs σ para vários N.
    Fig. 3(b): 1 - F_max e σ_max vs N (log-log), com ajustes de lei de potência.

Matriz S (fótons contrapropagantes, ν = entrada, ω = saída, sítios idênticos):

    S = S1(ωa) S1(ωb) δ(ωa-νa) δ(ωb-νb)
        - i(χγ²/π) K(E) δ(ωa+ωb-νa-νb) / [Γ(νa)Γ(νb)Γ(ωa)Γ(ωb)]
          × Σ_{j=1}^{N} A^{N-j} B^{j-1},

    A = a(ωa) a(νb),   B = a(ωb) a(νa),   a(ω) = t(ω) e^{iωL},   t = Γ*/Γ,
    S1(ω) = e^{i(N-1)ωL} t(ω)^N,          K(E) = [1 + 2iχ/(γ + i(2Δ - E))]^{-1},
    E = νa + νb = ωa + ωb.

Com L = 0 isto é exatamente a Eq. do artigo (a(ω) = Γ*/Γ). O sinal de S1
(o artigo escreve -[Γ*/Γ]^N) é uma fase global de fóton único e some quando a
deformação de fóton único é removida.

Overlap (Eq. A3), já com S1† S1† aplicado (a parte produto vira identidade):

    Ov = 1 + G,
    G  = ∫ dωa dωb dνa dνb ξ(ωa)ξ(ωb) S1*(ωa)S1*(ωb) S_conn ξ(νa)ξ(νb)

Fidelidade média (Eq. A2):  F_avg(φ) = [6 + 3 Re(e^{iφ} Ov) + |Ov|²] / 10.

Uso:  python3 fig3_infidelidade.py            (validações + figuras)
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

# ======================================================================
# Parâmetros físicos
# ======================================================================
chi = 1000.0          # não linearidade (np.inf = limite saturado do artigo)
gamma = 1.0           # taxa de decaimento (igual em todos os sítios)
Delta = 0.0           # dessintonia dos sítios
omega0 = 0.0          # centro do pacote relativo a Delta: ωc = Delta + omega0
phi = np.pi           # fase da porta (CPHASE)
L = 0.2               # deslocamento de fase do phase shifter (L=0: artigo)
N_values = np.arange(1, 51)  # N = 1, ..., 20 (N=1 só aparece na Fig. 3(a))

# ======================================================================
# Varredura / otimização em σ
# ======================================================================
sigma_lo, sigma_hi = 1e-3, 10.0   # faixa da grade logarítmica
n_sigma = 121                     # pontos da grade grossa (~40 por década)
fit_range = (4, 20)               # N usados nos ajustes de lei de potência
N_fig3a = [1, 2, 3, 4, 5, 10, 20, 50]   # curvas da Fig. 3(a), como no artigo
N_fig3b_min = 2                     # a Fig. 3(b) começa em N = 2

# ======================================================================
# Parâmetros numéricos da quadratura
# ======================================================================
k_sigma = 9.0         # meia-janela de integração em unidades de σ
pts_per_width = 14    # pontos por menor escala do integrando
n_min = 2001          # mínimo de pontos na grade 1D
n_max = 800001        # teto (proteção de memória)

# ======================================================================
# Saída
# ======================================================================
here = os.path.dirname(os.path.abspath(__file__))
out_fig = os.path.join(here, f'fig3_L{L:g}_Nmax50.svg')
out_npz = os.path.join(here, f'fig3_L{L:g}_chi{chi:g}_gamma{gamma:g}_omega{omega0:g}.npz')
run_validations = True   # validações (~25 s)
recompute = False        # False: reaproveita o .npz se já existir (só refaz o plot)

omega_c = Delta + omega0


# ----------------------------------------------------------------------
# Blocos da matriz S
# ----------------------------------------------------------------------
def Gamma(w):
    return gamma / 2 + 1j * (Delta - w)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (2 * np.pi * sigma ** 2) ** -0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def t(w):
    """Fase de fóton único de um sítio, Γ*/Γ (|t| = 1)."""
    return np.conj(Gamma(w)) / Gamma(w)


def a_ph(w, L):
    """Fase 'sítio + deslocador': a(ω) = t(ω) e^{iωL}."""
    return t(w) * np.exp(1j * w * L)


def S1(w, N, L):
    """Amplitude de fóton único através dos N sítios."""
    return np.exp(1j * (N - 1) * w * L) * t(w) ** N


def chiK(E, chi_=None):
    """χ K(E). Com χ = ∞ usa o limite saturado (γ + i(2Δ-E)) / (2i)."""
    chi_ = chi if chi_ is None else chi_
    D = gamma + 1j * (2 * Delta - E)
    if np.isinf(chi_):
        return D / 2j
    return chi_ * D / (D + 2j * chi_)


def S2_conn(wa, wb, va, vb, N, L, chi_=None):
    """Parte conectada de <ωa ωb|S|νa νb> sem a delta (ωa+ωb = νa+νb).

    Soma em j pela forma fechada da série geométrica
        Σ_{j=1}^N A^{N-j} B^{j-1} = (A^N - B^N)/(A - B)   (→ N A^{N-1} se A→B).
    Usada só na checagem por força bruta; o cálculo de produção usa a
    fatoração em convoluções (compute_G), onde a soma explícita é necessária.
    """
    A = a_ph(wa, L) * a_ph(vb, L)
    B = a_ph(wb, L) * a_ph(va, L)
    diff = A - B
    close = np.abs(diff) < 1e-8
    geo = np.where(close, N * A ** (N - 1),
                   (A ** N - B ** N) / np.where(close, 1.0, diff))
    E = va + vb
    return (-1j * gamma ** 2 / np.pi) * chiK(E, chi_) * geo / (
        Gamma(va) * Gamma(vb) * Gamma(wa) * Gamma(wb))


# ----------------------------------------------------------------------
# Overlap
# ----------------------------------------------------------------------
def grade(sigma, N, L, pts=None, ks=None):
    """Grade uniforme de frequências para a quadratura.

    Janela: ±k_sigma·σ (as quatro gaussianas confinam o integrando).
    Espaçamento: resolve a menor escala entre σ, γ/(2N) (a fase t^N gira
    ~4N/γ rad por unidade de ω na ressonância) e 1/(N|L|) (fases e^{iωL}).
    """
    pts = pts_per_width if pts is None else pts
    ks = k_sigma if ks is None else ks
    half = ks * sigma
    scales = [sigma, gamma / (2 * N)]
    if L != 0:
        scales.append(1.0 / (N * abs(L)))
    n = int(2 * half / (min(scales) / pts)) + 1
    n = min(max(n, n_min), n_max) | 1       # ímpar → Simpson com nº par de intervalos
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


def _conv_terms(w, h, sigma, N, L):
    """Devolve (E, [C_out_j(E)], [C_in_j(E)]) — as convoluções de cada termo j."""
    xi, G = xi_in(w, sigma), Gamma(w)
    a = a_ph(w, L)
    base_out = xi * np.conj(S1(w, N, L)) / G    # fatores de ωa, ωb (com S1†)
    base_in = xi / G                            # fatores de νa, νb
    E = 2 * w[0] + h * np.arange(2 * len(w) - 1)
    conv_out, conv_in = [], []
    for j in range(1, N + 1):
        f, g = a ** (N - j), a ** (j - 1)
        conv_out.append(fftconvolve(base_out * f, base_out * g) * h)  # ∫dωa f(ωa)g(E-ωa)
        conv_in.append(fftconvolve(base_in * g, base_in * f) * h)     # ∫dνa g(νa)f(E-νa)
    return E, conv_out, conv_in


def compute_G(sigma, N, L, chi_=None, pts=None, ks=None):
    """Parte conectada do overlap, G = Ov - 1.

    A delta elimina ωb = E - ωa. Como K só depende de E e cada fator restante
    de A^{N-j}B^{j-1} depende de uma única frequência
        A^{N-j}B^{j-1} = [a^{N-j}(ωa) a^{j-1}(ωb)] [a^{j-1}(νa) a^{N-j}(νb)],
    a integral tripla (ωa, νa, νb) vira, para cada j, uma integral 1D em E do
    produto de duas convoluções (FFT). Custo O(N n log n) em vez de O(n³).
    """
    w, h = grade(sigma, N, L, pts, ks)
    E, co, ci = _conv_terms(w, h, sigma, N, L)
    soma = sum(o * i for o, i in zip(co, ci))
    return (-1j * gamma ** 2 / np.pi) * simpson(chiK(E, chi_) * soma, x=E)


def overlap(sigma, N, L, chi_=None, **kw):
    """Ov (Eq. A3), com a deformação de fóton único removida."""
    return 1.0 + compute_G(sigma, N, L, chi_, **kw)


def F_avg(Ov, phi_=phi):
    """Fidelidade média de Haar (Eq. A2)."""
    return (6 + 3 * np.real(np.exp(1j * phi_) * Ov) + np.abs(Ov) ** 2) / 10.0


def fidelity(sigma, N, L, **kw):
    return F_avg(overlap(sigma, N, L, **kw))


# ----------------------------------------------------------------------
# Otimização em σ
# ----------------------------------------------------------------------
def optimize_sigma(N, L, **kw):
    """σ_max = argmax F_avg: grade log grossa + refino com minimize_scalar
    (Brent limitado em log σ, entre os vizinhos do máximo da grade)."""
    sig = np.logspace(np.log10(sigma_lo), np.log10(sigma_hi), n_sigma)
    F = np.array([fidelity(s, N, L, **kw) for s in sig])
    k = int(np.argmax(F))
    if k in (0, len(sig) - 1):
        print(f'  AVISO: máximo na borda da grade (N={N}, L={L:g}, σ={sig[k]:.3g})')
    lo, hi = np.log10(sig[max(k - 1, 0)]), np.log10(sig[min(k + 1, len(sig) - 1)])
    res = minimize_scalar(lambda x: -fidelity(10 ** x, N, L, **kw),
                          bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-7})
    return 10 ** res.x, -res.fun, sig, F


def fit_power_law(N, y, rng=fit_range):
    """Ajuste linear em log-log: y = C N^{-p}. Devolve (C, p)."""
    m = (N >= rng[0]) & (N <= rng[1])
    slope, icpt = np.polyfit(np.log(N[m]), np.log(y[m]), 1)
    return np.exp(icpt), -slope


# ----------------------------------------------------------------------
# Validações
# ----------------------------------------------------------------------
def check(name, ok, detail):
    print(f'  [{"OK " if ok else "FALHOU"}] {name}: {detail}')
    return ok


def norm_defect(sigma, N, L, n_theta=8001, e_stride=None):
    """Unitaridade: ||ξξ + ψ̃||² - 1 = 2 Re G + ∫∫|ψ̃|², com
    ψ̃(ωa,ωb) = S1*(ωa)S1*(ωb) ∫dνa S_conn ξ(νa)ξ(E-νa) a parte conectada
    do estado de saída. Deve ser 0 se S for unitária.

    ∫∫dωa dωb = ∫dE ∫du, ωa = E/2 + u, ωb = E/2 - u. A integral em u vai até
    ±∞ (ψ̃ decai só como 1/u², fora da janela gaussiana) via u = (γ/2) tan θ.
    """
    w, h = grade(sigma, N, L)
    E, _, ci = _conv_terms(w, h, sigma, N, L)
    s = e_stride or max(1, len(E) // 600)   # sub-amostra E (integrando suave em escala σ)
    E, ci = E[::s], [c[::s] for c in ci]
    theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)[1:-1]
    u = gamma / 2 * np.tan(theta)
    jac = gamma / 2 / np.cos(theta) ** 2
    inner = np.empty(len(E))
    for k, Ek in enumerate(E):
        wa, wb = Ek / 2 + u, Ek / 2 - u
        aa, ab = a_ph(wa, L), a_ph(wb, L)
        amp = sum(aa ** (N - j) * ab ** (j - 1) * ci[j - 1][k] for j in range(1, N + 1))
        psi = (gamma ** 2 / np.pi) * chiK(Ek) * amp / (Gamma(wa) * Gamma(wb))
        inner[k] = simpson(np.abs(psi) ** 2 * jac, x=theta)
    return 2 * np.real(compute_G(sigma, N, L)) + simpson(inner, x=E)


def brute_force_G(sigma, N, L, n=161, ks=6.0):
    """G por integração 3D direta (trapézio em ωa, νa, νb) com S2_conn pontual."""
    x = np.linspace(omega_c - ks * sigma, omega_c + ks * sigma, n)
    h = x[1] - x[0]
    wa, va, vb = np.meshgrid(x, x, x, indexing='ij', sparse=True)
    wb = va + vb - wa
    integ = (xi_in(wa, sigma) * xi_in(wb, sigma) * xi_in(va, sigma) * xi_in(vb, sigma)
             * np.conj(S1(wa, N, L) * S1(wb, N, L)) * S2_conn(wa, wb, va, vb, N, L))
    return integ.sum() * h ** 3


def validate():
    print('=== Validações ===')
    ok = True
    for s in (1e-3, 0.1, 10.0):
        w = np.linspace(omega_c - 12 * s, omega_c + 12 * s, 4001)
        nrm = simpson(np.abs(xi_in(w, s)) ** 2, x=w)
        ok &= check(f'∫|ξ|² = 1 (σ={s:g})', abs(nrm - 1) < 1e-10, f'{nrm:.12f}')

    Ov0 = overlap(0.3, 5, 0.4, chi_=0.0)
    ok &= check('χ=0 → Ov=1, F_avg(φ=0)=1', abs(Ov0 - 1) < 1e-14 and abs(F_avg(Ov0, 0) - 1) < 1e-14,
                f'Ov={Ov0:.3g}, F_avg(0)={F_avg(Ov0, 0):.12f}, F_avg(π)={F_avg(Ov0):.3f} (=0.4 esperado)')
    ok &= check('Ov=-1 → F_avg(π)=1', abs(F_avg(-1.0) - 1) < 1e-15, f'{F_avg(-1.0)}')

    for N, L_, s in [(2, 0.0, 0.3), (3, 0.4, 0.3), (5, 0.4, 0.15)]:
        Gc, Gb = compute_G(s, N, L_), brute_force_G(s, N, L_)
        ok &= check(f'convolução vs força bruta 3D (N={N}, L={L_:g}, σ={s:g})',
                    abs(Gc - Gb) < 1e-5 * max(1, abs(Gb)), f'|ΔG|={abs(Gc - Gb):.1e}')

    for N, L_, s in [(2, 0.0, 0.2), (2, 0.4, 0.2), (6, 0.4, 0.1), (20, 0.2, 0.03)]:
        d = norm_defect(s, N, L_)
        ok &= check(f'unitaridade ||ψ_out||²=1 (N={N}, L={L_:g}, σ={s:g})', abs(d) < 1e-5,
                    f'defeito={d:.1e}')

    for N, L_ in [(4, 0.0), (20, 0.0), (20, L)]:
        s1, F1, *_ = optimize_sigma(N, L_)
        F2 = fidelity(s1, N, L_, pts=2 * pts_per_width, ks=k_sigma + 3)
        ok &= check(f'convergência (N={N}, L={L_:g}), resolução ×2 e janela +3σ',
                    abs(F2 - F1) < 1e-5, f'ΔF_max={abs(F2 - F1):.1e}')

    s_inf, F_inf, *_ = optimize_sigma(20, 0.0, chi_=np.inf)
    print(f'  [info] N=20, L=0, χ=∞: F_max={F_inf:.5f}, σ_max={s_inf:.4f} '
          f'(artigo: F≈0.996)')
    print('=== Todas as validações passaram ===\n' if ok else '=== HÁ VALIDAÇÕES FALHANDO ===\n')
    return ok


# ----------------------------------------------------------------------
# Cálculo (com cache em .npz)
# ----------------------------------------------------------------------
sigma_fig3a = np.logspace(-2, 0, 301)   # grade fina só para desenhar a Fig. 3(a)


def compute_results():
    t0 = time.time()
    smax, Fmax = [], []
    for N in N_values:
        s, F, *_ = optimize_sigma(N, L)
        smax.append(s), Fmax.append(F)
        print(f'L={L:g} N={N:2d}  σ_max={s:.5f}  F_max={F:.6f}  1-F={1 - F:.3e}')
    F_curves = np.array([[fidelity(s, N, L) for s in sigma_fig3a] for N in N_fig3a])
    print(f'[{time.time() - t0:.1f} s]')
    r = dict(N=N_values, sigma_max=np.array(smax), F_max=np.array(Fmax),
             N_fig3a=np.array(N_fig3a), sigma_fig3a=sigma_fig3a, F_curves=F_curves,
             L=L, chi=chi, gamma=gamma, omega0=omega0)
    np.savez(out_npz, **r)
    print(f'Dados: {out_npz}')
    return r


def load_or_compute():
    if not recompute and os.path.exists(out_npz):
        d = dict(np.load(out_npz))
        if list(d['N']) == list(N_values) and list(d['N_fig3a']) == list(N_fig3a):
            print(f'Usando dados salvos em {out_npz} (recompute=False)')
            return d
    return compute_results()


# ----------------------------------------------------------------------
# Figura no estilo da Fig. 3 do artigo (Mathematica)
# ----------------------------------------------------------------------
# Paleta padrão do Mathematica (ColorData[97]), na mesma ordem do artigo
MMA = ['#5E81B5', '#E19C24', '#8FB032', '#EB6235', '#8778B3',
       '#C56E1A', '#5D9EC7', '#FFBF00', '#A5609D', '#929600']
xlabel_a = r'Largura do pacote de onda ($\sigma$)'
ylabel_a = r'Fidelidade ($F$)'
xlabel_b = r'Número de sítios de interação ($N$)'
figsize = (5.0, 5.6)
# Altura dos rótulos "N = k" na Fig. 3(a); o x é calculado para o rótulo ficar
# logo à esquerda da subida da curva (como no artigo). Ajuste à mão se precisar.
label_y = {1: 0.56, 2: 0.66, 3: 0.715, 4: 0.77, 5: 0.825, 10: 0.885, 20: 0.925}
label_shift = 1.3      # rótulo centrado em σ_cruzamento / label_shift

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'font.sans-serif': ['Nimbus Sans', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'cm', 'font.size': 10, 'svg.fonttype': 'path',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#555555',
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.color': '#555555', 'ytick.color': '#555555',
    'xtick.labelcolor': 'black', 'ytick.labelcolor': 'black',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.frameon': False,
})
_tick_font = dict(family='sans-serif', size=8)   # rótulos numéricos sem serifa, como no Mathematica


def _fmt_ticks(ax, xt, xl, yt, yl):
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.set_xticklabels(xl, **_tick_font)
    ax.set_yticklabels(yl, **_tick_font)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_fig3(r, outfile=out_fig):
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=figsize,
                                     gridspec_kw=dict(height_ratios=[1.9, 1], hspace=0.28))

    # ---- (a) F vs σ -------------------------------------------------
    sig, Fc = r['sigma_fig3a'], r['F_curves']
    for k, (N, F) in enumerate(zip(r['N_fig3a'], Fc)):
        c = MMA[k % len(MMA)]
        ax_a.semilogx(sig, F, '-', color=c, lw=1.1)
        # rótulo à esquerda da subida da curva, como no artigo
        i_pk = int(np.argmax(F))
        y_lab = min(label_y.get(int(N), 0.7), F[i_pk] - 0.03)
        x_cross = np.exp(np.interp(y_lab, F[:i_pk + 1], np.log(sig[:i_pk + 1])))
        x_lab = max(x_cross / label_shift, sig[0] * 1.3)
        ax_a.text(x_lab, y_lab, f'N = {int(N)}', ha='center', va='center',
                  fontdict=dict(_tick_font, size=7),
                  bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.6',
                            fc='#E6E6E6', ec='none'))
    ax_a.set_xlim(sig[0], sig[-1])
    ax_a.set_ylim(0.35, 1.0)
    _fmt_ticks(ax_a, [0.01, 0.05, 0.1, 0.5, 1], ['0.01', '0.05', '0.1', '0.5', '1'],
               np.arange(0.4, 1.01, 0.1), [f'{v:.1f}' for v in np.arange(0.4, 1.01, 0.1)])
    ax_a.set_xlabel(xlabel_a, labelpad=2)
    ax_a.set_ylabel(ylabel_a)
    ax_a.text(0.975, 0.965, '(a)', transform=ax_a.transAxes, ha='right', va='top',
              fontdict=dict(_tick_font, size=9))

    # ---- (b) 1-F_max e σ_max vs N -------------------------------------
    m = r['N'] >= N_fig3b_min
    N, smax, inf = r['N'][m], r['sigma_max'][m], 1 - r['F_max'][m]
    B, q = fit_power_law(r['N'], r['sigma_max'])
    A, p = fit_power_law(r['N'], 1 - r['F_max'])
    Nf = np.linspace(N.min(), N.max() + 0.6, 100)
    ax_b.loglog(Nf, B * Nf ** -q, '--', color=MMA[0], lw=1.0, dashes=(4, 2.5))
    ax_b.loglog(Nf, A * Nf ** -p, '--', color=MMA[1], lw=1.0, dashes=(4, 2.5))
    ax_b.loglog(N, smax, 'o', color=MMA[0], ms=4.2, mec='none', label=r'$\sigma_{\max}$')
    ax_b.loglog(N, inf, 'o', mfc='white', mec=MMA[1], mew=0.9, ms=4.2, label=r'$1-F$')
    ax_b.set_xlim(N.min() * 0.93, N.max() * 1.06)
    lo = min(smax.min(), inf.min())
    hi = max(smax.max(), inf.max())
    yt = [v for v in (0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.200, 0.500)
          if lo / 1.3 <= v <= hi * 1.3]
    ax_b.set_ylim(lo / 1.3, hi * 1.3)
    _fmt_ticks(ax_b, [2, 5, 10, 20,50], ['2', '5', '10', '20', '50'], yt, [f'{v:.3f}' for v in yt])
    ax_b.set_xlabel(xlabel_b, labelpad=2)
    ax_b.legend(loc='lower left', handletextpad=0.1, borderaxespad=0.3,
                labelspacing=0.3, fontsize=10)
    ax_b.text(0.975, 0.93, '(b)', transform=ax_b.transAxes, ha='right', va='top',
              fontdict=dict(_tick_font, size=9))

    print(f'Ajustes em N ∈ {list(fit_range)} (L={L:g}):  1-F = {A:.3f} N^(-{p:.3f}),'
          f'  σ_max = {B:.3f} N^(-{q:.3f})')
    print('Artigo (L=0, χ→∞):            1-F = 0.537 N^(-1.61),  σ_max = 0.350 N^(-0.81)')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'Figura: {outfile}')
    return fig


# ----------------------------------------------------------------------
def main():
    if run_validations:
        validate()
    plot_fig3(load_or_compute())
    plt.show()


if __name__ == '__main__':
    main()
