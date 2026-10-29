# -*- coding: utf-8 -*-
"""
Fig. 3(a) de Brod-Combes com um oscilador em cada atomo no regime de MOLA
LENTA, omega^(a,b) << gamma (e << sigma).  Gera F_1(sigma) para varios N e
salva em data/*.npz para serem plotados por plot_fock_N.py.

Por que um codigo separado.  Em calcular_fock_N.py a soma sobre os estados
finais m dos osciladores e truncada em poucos osciladores excitados.  Com
acoplamento g fixo e omega -> 0, o fator de Huang-Rhys S = (2g/omega)^2
cresce sem limite: cada passagem do foton muda o estado do oscilador de
muitos quanta e a truncagem deixa de valer.  Por outro lado, nesse limite o
oscilador nao se move durante a passagem dos fotons:

  * a posicao X = b + b^dagger comuta com o hamiltoniano quando omega = 0;
  * para cada valor de X o atomo so tem a ressonancia deslocada,
        Delta -> Delta + 2 g X                                         (L1)
    (a media de 2gX em |n> e zero e a variancia e 4 g^2 (2n+1): sao a media
    e a variancia de (k-n) omega - S omega com pesos F_kn^2, ou seja, a
    distribuicao de Franck-Condon vista com omega -> 0);
  * a matriz S fica diagonal em X, e o deslocamento de cor (m-n) omega -> 0.

Fidelidade de estado misto.  Com x_k(m) = <m| X_k |n> (operadores no espaco
dos 2N osciladores) e X_k diagonal em X,
    F = sum_m x(m)^dagger W1 x(m) = <n| x^dagger W1 x |n>
      = int d^{2N}X  prod_i |psi_{n_i}(X_i)|^2  x(X)^dagger W1 x(X),      (L2)
em que x(X) sao as amplitudes de Brod-Combes para N sitios com dessintonias
diferentes (uma por atomo).  A soma sobre TODOS os m (todas as ordens) esta
incluida.  A integral em 2N dimensoes e feita por quase-Monte Carlo (Sobol).

Validade: omega << min(sigma, gamma/N), ou seja, o oscilador quase nao gira
durante a passagem do pacote.  Correcoes sao O(omega/sigma, N omega/gamma).

Uso:  python3 calcular_fock_N_lenta.py
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.fft import fft, ifft, next_fast_len
from scipy.special import eval_hermite, factorial
from scipy.stats import qmc

# ======================================================================
# Parametros fisicos dos sitios (iguais aos da Fig. 3 de Brod & Combes)
# ======================================================================
chi = 1000.0         # acoplamento cross-Kerr em cada sitio (np.inf: limite saturado)
gam = 1.0            # gamma: taxa de emissao de cada atomo no guia (unidade de frequencia)
Delta = 0.0          # dessintonia dos atomos
omega0 = 0.0         # frequencia central do pacote, relativa a Delta
phi = np.pi          # fase da porta CPHASE ideal

omega_c = Delta + omega0   # frequencia central do pacote (derivada, nao editar)

# ======================================================================
# Osciladores lentos.  Cada caso e (g_a, n_a, g_b, n_b), igual em todos os
# sitios:  g = acoplamento atomo-oscilador, n = estado de Fock inicial.
# A frequencia omega nao entra no calculo (omega -> 0); ela so define a
# validade: omega << min(sigma, gamma/N).  Largura do deslocamento (L1):
#     2 g sqrt(2n + 1).
# g = 0 desliga as molas (Brod-Combes).
# ======================================================================
casos = [
    (0.05, 0, 0.05, 0),
    (0.05, 1, 0.05, 1),
    (0.05, 2, 0.05, 2),
]

# N_values = [1, 2, 3, 4, 5, 10, 20]   # numeros de sitios (curvas da Fig. 3(a))
N_values = [1, 10, 20]

# Fase u(w) do foton ideal xi~ = u xi:
#   'elastica' : fase da amplitude elastica media da cadeia, prod_k <n|t_k|n>(w)
#                (o mesmo que em calcular_fock_N.py, tomado com omega -> 0)
#   'sem_mola' : (-Gamma*/Gamma)^N
referencia = 'elastica'

# ======================================================================
# Amostragem das posicoes X dos 2N osciladores
# ======================================================================
log2_amostras = 9    # 2^9 = 512 pontos de Sobol em 2N dimensoes por embaralhamento
n_embaralhamentos = 16  # Sobols embaralhados independentes (total 8192 amostras); o erro
                        # estatistico e o desvio padrao das medias / sqrt(n_embaralhamentos)
semente = 1          # semente (mesmas amostras para todo sigma: curvas suaves)
n_hermite = 80       # pontos de Gauss-Hermite para as medias de um oscilador (referencia)

# ======================================================================
# Varredura em sigma e grade (como em calcular_fock_N.py)
# ======================================================================
sigma_min = 0.01
sigma_max = 1.0
n_sigma = 61
k_sigma = 9.0        # metade da janela de frequencias, em unidades de sigma
pts_per_width = 8    # pontos por menor escala, min(sigma, gamma/(2N))
n_min = 401
n_max = 40001
mem_bloco = 2e7      # numero de complexos por bloco de amostras (controla a memoria)

here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, 'data')
quiet = False
run_validations = True
n_proc = max(1, (os.cpu_count() or 1) - 1)


# ======================================================================
# Distribuicao da posicao X = b + b^dagger no estado de Fock |n>
# ======================================================================
# Com X = sqrt(2) x, a funcao de onda e a de Hermite,
#     |psi_n(x)|^2 = H_n(x)^2 e^{-x^2} / (2^n n! sqrt(pi)).
def densidade_X(X, n):
    """|psi_n(X)|^2 para X = b + b^dagger (normalizada em X)."""
    x = X / np.sqrt(2)
    return eval_hermite(n, x)**2 * np.exp(-x**2) / (2**n * factorial(n) * np.sqrt(np.pi)) / np.sqrt(2)


def quantis_X(u, n):
    """Inversa da distribuicao acumulada de X em |n> (para transformar o Sobol)."""
    L = 2 * np.sqrt(2 * n + 1) + 8
    X = np.linspace(-L, L, 20001)
    cdf = np.cumsum(densidade_X(X, n))
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return np.interp(u, cdf, X)


def nos_X(n):
    """Nos e pesos de quadratura para medias <n| f(X) |n> (Gauss-Hermite)."""
    x, w = hermgauss(n_hermite)
    pol = eval_hermite(n, x)**2 / (2**n * factorial(n) * np.sqrt(np.pi))
    return np.sqrt(2) * x, w * pol


def amostras_X(N, ns_a, ns_b, M=None, R=None):
    """Matrizes (R*M, N) com as posicoes X dos osciladores a e b de cada sitio:
    R conjuntos de Sobol embaralhados independentes, de M pontos cada."""
    M = 2**log2_amostras if M is None else M
    R = n_embaralhamentos if R is None else R
    u = np.vstack([qmc.Sobol(d=2 * N, scramble=True, seed=semente + r).random_base2(int(np.log2(M)))
                   for r in range(R)])
    u = np.clip(u, 1e-12, 1 - 1e-12)
    Xa = np.column_stack([quantis_X(u[:, k], ns_a[k]) for k in range(N)])
    Xb = np.column_stack([quantis_X(u[:, N + k], ns_b[k]) for k in range(N)])
    return Xa, Xb


# ======================================================================
# Blocos de Brod-Combes com a dessintonia de cada atomo   [Eqs. (L1), (N2)]
# ======================================================================
def Gam(D, w):
    """Gamma(w) = gamma/2 + i(D - w), com D = Delta + 2 g X a dessintonia do atomo."""
    return gam / 2 + 1j * (D - w)


def t1(D, w):
    """Amplitude de um foton de um atomo parado: -Gamma*/Gamma (|t| = 1)."""
    G = Gam(D, w)
    return -np.conj(G) / G


def chiK(Da, Db, E):
    """chi [1 + 2i chi / Sigma]^{-1}, Sigma = Gamma_a(xa) + Gamma_b(xb) = gamma + i(Da + Db - E)."""
    Sig = gam + 1j * (Da + Db - E)
    if np.isinf(chi):
        return Sig / 2j
    return chi * Sig / (Sig + 2j * chi)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4) de Brod & Combes."""
    return (2 * np.pi * sigma**2)**-0.25 * np.exp(-(w - omega_c)**2 / (4 * sigma**2))


def fase_ref(N, g, n, w):
    """u(w), |u| = 1, de um foton que atravessa N atomos com acoplamento g e Fock n."""
    if referencia == 'sem_mola':
        return t1(Delta, w)**N
    X, p = nos_X(n)
    tm = (p[:, None] * t1(Delta + 2 * g * X[:, None], w[None, :])).sum(0)   # <n| t |n>
    tm = tm**N
    a = np.abs(tm)
    return np.where(a > 1e-300, tm / np.where(a > 1e-300, a, 1.0), 1.0)


def grade(sigma, N):
    """Grade de frequencias: janela +-k_sigma*sigma, passo que resolve min(sigma, gamma/(2N))."""
    half = k_sigma * sigma
    escala = min(sigma, gam / (2 * N))
    n = int(2 * half / (escala / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


# ======================================================================
# Amplitudes para um bloco de amostras de X
# ======================================================================
# Para cada amostra (todas as dessintonias fixas) as amplitudes sao as de
# calcular_fock_N.py com S = 0, so que com um Delta por atomo:
#   tau_a = int dnu xi~_a^* xi prod_k t_k^a(nu),   tau_b igual com os atomos b
#   N_j   = -i gamma^2/pi int dE chiK_j(E) P_j(E) Q_j(E)
#   P_j   = conv[ xi prod_{k<j} t^a_k / Gamma^a_j ,  xi prod_{k>j} t^b_k / Gamma^b_j ]       (antes)
#   Q_j   = conv[ xi~_a^* prod_{k>j} t^a_k / Gamma^a_j ,  xi~_b^* prod_{k<j} t^b_k / Gamma^b_j ]   (depois)
# Os produtos prod_{k<j} e prod_{k>j} vem de produtos acumulados nos sitios.
W1 = (np.ones((4, 4)) + np.eye(4)) / 20


def amplitudes_bloco(Da, Db, w, h, sigma, ref_a, ref_b):
    """Devolve (tau_a, tau_b, Ncal) para M amostras; Da, Db tem forma (M, N)."""
    M, N = Da.shape
    n = len(w)
    xi = xi_in(w, sigma)
    Ga = Gam(Da[:, :, None], w)          # (M, N, n)
    Gb = Gam(Db[:, :, None], w)
    ta, tb = -np.conj(Ga) / Ga, -np.conj(Gb) / Gb
    um = np.ones((M, 1, n), complex)
    pre_a = np.concatenate([um, np.cumprod(ta, axis=1)[:, :-1]], axis=1)              # prod_{k<j}
    suf_a = np.concatenate([np.cumprod(ta[:, ::-1], axis=1)[:, ::-1][:, 1:], um], axis=1)  # prod_{k>j}
    pre_b = np.concatenate([um, np.cumprod(tb, axis=1)[:, :-1]], axis=1)
    suf_b = np.concatenate([np.cumprod(tb[:, ::-1], axis=1)[:, ::-1][:, 1:], um], axis=1)
    tot_a = pre_a[:, -1] * ta[:, -1]
    tot_b = pre_b[:, -1] * tb[:, -1]
    tau_a = (ref_a * xi * tot_a).sum(-1) * h
    tau_b = (ref_b * xi * tot_b).sum(-1) * h

    Nf = next_fast_len(2 * n - 1)
    E = 2 * w[0] + h * np.arange(2 * n - 1)
    Ncal = np.zeros(M, complex)
    for j in range(N):
        A = fft(xi * pre_a[:, j] / Ga[:, j], Nf, axis=-1)        # a antes
        B = fft(xi * suf_b[:, j] / Gb[:, j], Nf, axis=-1)        # b antes
        A2 = fft(ref_a * suf_a[:, j] / Ga[:, j], Nf, axis=-1)    # a depois
        B2 = fft(ref_b * pre_b[:, j] / Gb[:, j], Nf, axis=-1)    # b depois
        P = ifft(A * B, axis=-1)[:, :2 * n - 1]
        Q = ifft(A2 * B2, axis=-1)[:, :2 * n - 1]
        K = chiK(Da[:, j, None], Db[:, j, None], E[None, :])
        Ncal += -1j * gam**2 / np.pi * (K * P * Q).sum(-1) * h**3
    return tau_a, tau_b, Ncal


def fidelidade(N, g_a, n_a, g_b, n_b, sigma, X=None, erro=False):
    """F1(sigma) pela media (L2) sobre as posicoes X dos 2N osciladores.
    Com erro=True devolve tambem o erro estatistico (desvio entre os
    embaralhamentos independentes do Sobol)."""
    Xa, Xb = amostras_X(N, [n_a] * N, [n_b] * N) if X is None else X
    w, h = grade(sigma, N)
    ref_a = np.conj(fase_ref(N, g_a, n_a, w) * xi_in(w, sigma))
    ref_b = np.conj(fase_ref(N, g_b, n_b, w) * xi_in(w, sigma))
    M = len(Xa)
    bloco = max(1, int(mem_bloco / (N * len(w))))
    f = np.empty(M)
    for i in range(0, M, bloco):
        Da = Delta + 2 * g_a * Xa[i:i + bloco]
        Db = Delta + 2 * g_b * Xb[i:i + bloco]
        ta, tb, Nc = amplitudes_bloco(Da, Db, w, h, sigma, ref_a, ref_b)
        x = np.stack([np.ones_like(ta), tb, ta, np.exp(1j * phi) * (ta * tb + Nc)], axis=1)
        f[i:i + bloco] = np.real(np.einsum('si,ij,sj->s', np.conj(x), W1, x))
    F1 = f.mean()
    if not erro:
        return F1
    R = n_embaralhamentos if X is None else 1
    medias = f.reshape(R, -1).mean(1)
    return F1, (medias.std(ddof=1) / np.sqrt(R) if R > 1 else np.nan)


# ======================================================================
# Validacoes
# ======================================================================
def check(nome, ok, detalhe):
    print(f'  [{"OK " if ok else "FALHOU"}] {nome}: {detalhe}')
    return ok


def validar():
    print('=== Validacoes (mola lenta) ===')
    ok = True
    sys.path.insert(0, here)
    import calcular_fock_N as cf
    sys.path.pop(0)

    # 1) g = 0: Brod-Combes
    ref = os.path.join(here, '..', 'infidelidade', 'data', 'fig3_L0_chi1000_gamma1_omega0.npz')
    if os.path.exists(ref) and chi == 1000 and gam == 1 and omega0 == 0 and Delta == 0:
        d = np.load(ref)
        for N, s in [(1, 0.3), (5, 0.1), (20, 0.03)]:
            i = list(d['N_fig3a']).index(N)
            Fr = np.interp(np.log(s), np.log(d['sigma_fig3a']), d['F_curves'][i])
            F = fidelidade(N, 0.0, 0, 0.0, 0, s, X=amostras_X(N, [0] * N, [0] * N, 8))
            ok &= check(f'g=0 vs Brod-Combes (N={N}, sigma={s:g})', abs(F - Fr) < 2e-4,
                        f'F={F:.6f}, ref={Fr:.6f}')

    # 2) <n|t|n> com omega -> 0 vs t_nn de Franck-Condon do codigo exato
    #    (S = (2g/omega)^2 ate ~40; acima disso os F_kn do codigo exato perdem precisao)
    g, n0 = 0.05, 1
    X, p = nos_X(n0)
    x = np.linspace(-2, 2, 9)
    t_med = (p[:, None] * t1(Delta + 2 * g * X[:, None], x[None, :])).sum(0)
    for w_ in [0.05, 0.025]:
        S = (2 * g / w_)**2
        sit = cf.montar_cadeia(1, w_, S, n0, w_, S, n0)[0]
        dif = np.max(np.abs(cf.t_mn(sit, sit['a'], n0, x) - t_med))
        ok &= check(f'<n|t|n> estatico vs Franck-Condon (g={g}, n={n0}, omega={w_}, S={S:g})',
                    dif < 2 * w_, f'max|diferenca| = {dif:.1e} (deve ser O(omega/gamma))')

    # 3) N = 1: limite estatico vs soma exata sobre TODOS os m (codigo exato, omega pequeno)
    salvo = cf.dq_max
    cf.dq_max = 10
    try:
        w_ = 0.05
        S = (2 * g / w_)**2
        cad = cf.montar_cadeia(1, w_, S, n0, w_, S, n0)
        for s in [0.3, 1.0]:
            Fe = cf.fidelidade(cad, s, max_exc=2)
            Fs = fidelidade(1, g, n0, g, n0, s)
            ok &= check(f'N=1 estatico vs exato (g={g}, n={n0}, omega={w_}, S={S:g}, sigma={s:g})',
                        abs(Fe - Fs) < 5e-4, f'exato={Fe:.5f}, estatico={Fs:.5f}')
    finally:
        cf.dq_max = salvo

    # 4) erro estatistico do quase-Monte Carlo
    F, e = fidelidade(5, 0.05, 1, 0.05, 1, 0.1, erro=True)
    ok &= check('erro do quase-Monte Carlo (N=5, n=1, sigma=0.1)', e < 3e-3,
                f'F={F:.5f} +- {e:.1e}')
    print('=== Todas as validacoes passaram ===\n' if ok else '=== HA VALIDACOES FALHANDO ===\n')
    return ok


# ======================================================================
# Varredura e saida
# ======================================================================
def nome_arquivo(N, g_a, n_a, g_b, n_b):
    return (f"fidlenta_N{N}_chi{chi:g}_gam{gam:g}_w0{omega0:g}"
            f"_ga{g_a:g}_na{n_a}_gb{g_b:g}_nb{n_b}_{referencia}.npz")


def _um_sigma(args):
    return fidelidade(*args, erro=True)


def calcular_caso(N, g_a, n_a, g_b, n_b):
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    t0 = time.time()
    tarefas = [(N, g_a, n_a, g_b, n_b, s) for s in sigmas]
    if n_proc > 1:
        with ProcessPoolExecutor(n_proc) as ex:
            res = list(ex.map(_um_sigma, tarefas))
    else:
        res = [_um_sigma(t) for t in tarefas]
    F1s, erros = np.array(res).T
    k = int(np.argmax(F1s))
    if not quiet:
        print(f'  N={N:2d}: F1_max={F1s[k]:.5f} em sigma={sigmas[k]:.4f}  '
              f'(erro MC <= {erros.max():.1e})   [{time.time() - t0:.1f} s]', flush=True)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, nome_arquivo(N, g_a, n_a, g_b, n_b))
    np.savez(path, sigmas=sigmas, F1=F1s, erro=erros, N=N,
             chi=chi, gam=gam, Delta=Delta, omega0=omega0, phi=phi,
             g_a=g_a, n_a=n_a, g_b=g_b, n_b=n_b, referencia=referencia,
             amostras=2**log2_amostras, k_sigma=k_sigma, pts_per_width=pts_per_width)
    return path


def main():
    if run_validations:
        validar()
    for c in casos:
        if not quiet:
            print('caso (g_a, n_a, g_b, n_b) =', c, flush=True)
        for N in N_values:
            calcular_caso(N, *c)


if __name__ == '__main__':
    main()
