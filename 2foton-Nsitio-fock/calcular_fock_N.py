# -*- coding: utf-8 -*-
"""
Fidelidade da porta CPHASE de Brod-Combes com N sitios contrapropagantes em
que CADA atomo (A^(j) e B^(j), j = 1..N) esta acoplado a um oscilador
harmonico proprio ("mola"), inicialmente num estado de Fock.  Gera os dados da
Fig. 3(a) de Brod & Combes (arXiv:1604.04278) -- F_1(sigma) para varios N --
nesse novo cenario e salva em data/*.npz para serem plotados por plot_fock_N.py.

Matriz S de N sitios (matriz_S_N_sitios_osciladores.pdf, Eqs. N9-N11):

    <wa wb; m| S^(N) |nu_a nu_b; n> = F^(N) + sum_{j=1}^N X_j^(N)

  F^(N)  : os dois fotons atravessam os N sitios sem interagir (produto de
           amplitudes de um foton t^(kc)_{mn}, com as cores deslocadas pelos
           osciladores ja visitados);
  X_j^(N): o foton a atravessa sozinho 1..j-1, o foton b atravessa sozinho
           N..j+1, os dois interagem no sitio j (nucleo de Kerr C_j) e depois
           seguem sozinhos (a por j+1..N, b por j-1..1).

Os osciladores terminam num estado |m> que o foton nao "ve": o estado dos
fotons na saida e MISTO, e a fidelidade com a porta ideal (pura) e
    F = <psi_id| rho |psi_id> = sum_m x(m)^dagger W1 x(m),
media de Haar sobre os estados de entrada de 2 qubits (Eq. (A2) do artigo).

A soma em m e feita sobre um conjunto truncado de configuracoes (ver
`max_excitados`).  Como W1 e positiva, cada termo da soma e >= 0: truncar a
soma da um LIMITE INFERIOR da fidelidade, que converge por baixo.
Com max_excitados = 0 so entra o canal elastico (todos os osciladores voltam
ao estado inicial).

Uso:  python3 calcular_fock_N.py     (validacoes + um .npz por (caso, N))
"""
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.special import eval_genlaguerre, factorial

# ======================================================================
# Parametros fisicos dos sitios (iguais aos da Fig. 3 de Brod & Combes)
# ======================================================================
chi = 1000.0         # acoplamento nao linear (cross-Kerr) em cada sitio (np.inf: limite saturado)
gam = 1.0            # gamma: taxa de emissao de cada atomo no guia (unidade de frequencia)
Delta = 0.0          # dessintonia dos atomos
omega0 = 0.0         # frequencia central do pacote, relativa a Delta
phi = np.pi          # fase da porta CPHASE ideal

omega_c = Delta + omega0   # frequencia central do pacote (derivada, nao editar)

# ======================================================================
# Osciladores.  Cada caso e (w_a, S_a, n_a, w_b, S_b, n_b), igual em todos os
# sitios:  w = frequencia do oscilador, S = (2g/w)^2 = fator de Huang-Rhys,
#          n = estado de Fock inicial.  (a): osciladores dos atomos A^(j),
#          (b): dos atomos B^(j).  S = 0 desliga as molas (Brod-Combes).
# Para sitios diferentes entre si, edite `montar_cadeia`.
# ======================================================================
casos = [
    (1.0, 0.0, 0, 1.0, 0.0, 0),     # sem mola: Fig. 3(a) original
    (1.0, 0.01, 0, 1.0, 0.01, 0),
    (1.0, 0.01, 1, 1.0, 0.01, 1),
    (1.0, 0.01, 2, 1.0, 0.01, 2),
]

N_values = [1, 2, 3, 4, 5, 10, 20]   # numeros de sitios (curvas da Fig. 3(a))

# ======================================================================
# Fidelidade
# ======================================================================
# Fase u(w) que define o foton 'ideal' de saida xi~ = u xi (remove a
# deformacao de um foton, como Brod-Combes fazem com (-Gamma*/Gamma)^N):
#   'elastica' : fase da amplitude elastica da cadeia, prod_k t^(kc)_{nn}(w)
#                (recai em (-Gamma*/Gamma)^N quando S = 0; e a opcao 'media'
#                do codigo de um sitio)
#   'sem_mola' : (-Gamma*/Gamma)^N com Gamma = gam/2 + i(Delta - w)
referencia = 'elastica'

# Truncamento da soma sobre os estados finais m dos 2N osciladores:
max_excitados = 2    # no maximo este numero de osciladores termina com m != n (0: so o canal elastico)
dq_max = 1           # |m - n| maximo de cada oscilador
tol_sobreposicao = 1e-8    # descarta configuracoes cuja sobreposicao gaussiana com o foton ideal e menor

# ======================================================================
# Varredura em sigma
# ======================================================================
sigma_min = 0.01     # menor largura do pacote
sigma_max = 1.0      # maior largura do pacote
n_sigma = 61         # numero de pontos (escala log)

# ======================================================================
# Parametros numericos
# ======================================================================
k_sigma = 9.0        # metade da janela de frequencias, em unidades de sigma
pts_per_width = 8    # pontos por menor escala, min(sigma, gamma/(2N)) (convergencia espectral: 6 ja da ~1e-6)
n_min = 401          # numero minimo de pontos na grade
n_max = 40001        # teto de pontos (protecao de tempo/memoria)
eps_fc = 1e-6        # |F_kn| abaixo disso e tratado como zero (define o corte K)
eps_peso = 1e-8      # |F_km F_kn| abaixo disso: o nivel k nao entra nas somas

here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, 'data')   # pasta onde os .npz sao salvos
quiet = False        # se True, nao imprime o progresso
run_validations = True
n_proc = max(1, (os.cpu_count() or 1) - 1)   # processos em paralelo na varredura em sigma


# ======================================================================
# Oscilador: fatores de Franck-Condon
# ======================================================================
def FC(k, n, beta):
    """Fator de Franck-Condon F_kn = <k|D(beta)|n>, beta real."""
    if k >= n:
        return (np.sqrt(factorial(n) / factorial(k)) * beta**(k - n)
                * np.exp(-beta**2 / 2) * eval_genlaguerre(n, k - n, beta**2))
    return (-1)**(n - k) * FC(n, k, beta)


def oscilador(w, S, n):
    """Dicionario que descreve um oscilador de frequencia w, fator de
    Huang-Rhys S = (2g/w)^2 = beta^2, que comeca no estado de Fock |n>.

    K (dimensao do espaco do oscilador) cresce ate os niveis k acessiveis a
    partir de |n> e dos |m> com |m-n| <= dq_max nao encostarem no corte.
    `finais` sao os m != n que podem ser alcancados (|m-n| <= dq_max)."""
    beta = np.sqrt(S)
    cols = list(range(max(0, n - dq_max), n + dq_max + 1))
    K = n + dq_max + 6
    while True:
        F = np.array([[FC(k, m, beta) for m in range(K)] for k in range(K)])  # F[k, m]
        if np.all(np.abs(F[-3:, cols]) < eps_fc):
            break
        K += 5
    finais = [m for m in cols if m != n and S > 0
              and np.any(np.abs(F[:, m] * F[:, n]) > eps_peso)]
    # niveis intermediarios k (estados vestidos do atomo) que contam para cada m
    niv = {}
    for m in [n] + finais:
        p = F[:, m] * F[:, n]
        ks = np.nonzero(np.abs(p) > eps_peso)[0]
        niv[m] = (ks, p[ks])
    return dict(w=w, S=S, n=n, g=np.sqrt(S) * w / 2, desloc=S * w, K=K, F=F,
                finais=finais, niv=niv)


def niveis(osc, m):
    """Niveis intermediarios k (estados vestidos) com |F_km F_kn| > eps_peso."""
    return osc['niv'][m][0]


# ======================================================================
# A cadeia de sitios
# ======================================================================
def montar_cadeia(N, w_a, S_a, n_a, w_b, S_b, n_b):
    """Lista de N sitios.  Cada sitio j tem gamma_j, Delta_j, chi_j e dois
    osciladores independentes: 'a' (atomo A^(j), visto so pelo foton a) e
    'b' (atomo B^(j), visto so pelo foton b)."""
    return [dict(gam=gam, Delta=Delta, chi=chi,
                 a=oscilador(w_a, S_a, n_a), b=oscilador(w_b, S_b, n_b))
            for _ in range(N)]


def Gam(sit, osc, k, n, x):
    """Gamma^(jc)_{kn}(x) = gamma_j/2 + i(Delta_j - 4g^2/w + (k-n) w - x)   [Eq. (E2)]."""
    return sit['gam'] / 2 + 1j * (sit['Delta'] - osc['desloc'] + (k - n) * osc['w'] - x)


def t_mn(sit, osc, m, x):
    """Amplitude de um foton no sitio j, oscilador |n> -> |m>   [Eq. (N2)]:
        t_mn(x) = delta_mn - gamma_j sum_k F_km F_kn / Gamma_kn(x)."""
    ks, p = osc['niv'][m]
    x = np.asarray(x, float)
    G = Gam(sit, osc, ks[:, None], osc['n'], x.reshape(1, -1))
    s = (p @ (1.0 / G)).reshape(x.shape)
    return (1.0 if m == osc['n'] else 0.0) - sit['gam'] * s


def chiK(sit, oa, ob, ka, kb, E):
    """chi_j [1 + 2i chi_j / Sigma]^{-1}, com Sigma = Gamma^(ja)_{ka na}(xa) + Gamma^(jb)_{kb nb}(xb),
    que so depende de E = xa + xb.  chi = inf da o limite saturado Sigma/(2i)."""
    Sig = (sit['gam'] + 1j * (2 * sit['Delta'] - oa['desloc'] - ob['desloc']
                              + (ka - oa['n']) * oa['w'] + (kb - ob['n']) * ob['w'] - E))
    if np.isinf(sit['chi']):
        return Sig / 2j
    return sit['chi'] * Sig / (Sig + 2j * sit['chi'])


# ======================================================================
# Pacote e foton ideal
# ======================================================================
def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4) de Brod & Combes."""
    return (2 * np.pi * sigma**2)**-0.25 * np.exp(-(w - omega_c)**2 / (4 * sigma**2))


def fase_ref(cad, c, w):
    """u_c(w), |u| = 1: fase que define o foton ideal xi~_c = u_c xi do foton c = 'a' ou 'b'."""
    if referencia == 'sem_mola':
        u = np.ones_like(w, dtype=complex)
        for sit in cad:
            G0 = sit['gam'] / 2 + 1j * (sit['Delta'] - w)
            u = u * (-np.conj(G0) / G0)
        return u
    tm = np.ones_like(w, dtype=complex)
    for sit in cad:
        tm = tm * t_mn(sit, sit[c], sit[c]['n'], w)
    a = np.abs(tm)
    return np.where(a > 1e-300, tm / np.where(a > 1e-300, a, 1.0), 1.0)


def xi_ideal_conj(cad, c, w, sigma):
    """xi~_c^*(w) = conj(u_c(w) xi(w))."""
    return np.conj(fase_ref(cad, c, w) * xi_in(w, sigma))


def grade(sigma, N):
    """Passo h e numero de pontos n da grade de frequencias.

    Janela +-k_sigma*sigma (as gaussianas confinam o integrando); o passo
    resolve min(sigma, gamma/(2N)): o produto de N fatores t gira ~4N/gamma rad
    por unidade de frequencia perto da ressonancia."""
    half = k_sigma * sigma
    escala = min(sigma, gam / (2 * N))
    n = int(2 * half / (escala / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1
    return 2 * half / (n - 1), n


# Cache dos fatores que se repetem entre configuracoes (zerado a cada sigma).
_memo = {}


def t_rede(cad, k, c, m, h, n, i0, s):
    """t^(kc)_{m n}(x) nos pontos x = omega_c + h (i0 + arange(n)) - s, com cache."""
    chave = ('t', k, c, m, h, n, i0, round(s, 12))
    if chave not in _memo:
        x = omega_c + h * (i0 + np.arange(n)) - s
        _memo[chave] = t_mn(cad[k], cad[k][c], m, x)
    return _memo[chave]


def ref_rede(cad, c, sigma, h, n, i0, s):
    """xi~_c^*(x) nos pontos x = omega_c + h (i0 + arange(n)) - s, com cache."""
    chave = ('ref', c, h, n, i0, round(s, 12))
    if chave not in _memo:
        x = omega_c + h * (i0 + np.arange(n)) - s
        _memo[chave] = xi_ideal_conj(cad, c, x, sigma)
    return _memo[chave]


def pontos(sigma, h, n, s):
    """Grade de n pontos para uma funcao cujo suporte esta centrado em
    omega_c - s.  Os pontos ficam na rede omega_c + h*Z (para que as
    convolucoes de funcoes com centros diferentes se alinhem) e o indice do
    primeiro ponto nessa rede e devolvido junto."""
    i0 = int(np.rint(-s / h)) - (n - 1) // 2
    return omega_c + h * (i0 + np.arange(n)), i0


# ======================================================================
# Configuracoes dos osciladores
# ======================================================================
# Uma configuracao e (ma, mb): tuplas com o estado final m dos osciladores
# 'a' e 'b' de cada sitio.  Os deslocamentos de cor sao   [Eq. (N1)]
#     delta_a^(k) = (m_a^(k) - n_a^(k)) w^(ka),   delta_b^(k) = (m_b^(k) - n_b^(k)) w^(kb).
# O foton a percorre 1 -> N; o foton b percorre N -> 1.  Por isso usamos
#     pa[k] = sum_{l<k} delta_a^(l)   (o que o foton a ja deixou antes do sitio k)
#     sb[k] = sum_{l>k} delta_b^(l)   (o que o foton b ja deixou antes do sitio k)
def deslocamentos(cad, ma, mb):
    da = np.array([(m - s['a']['n']) * s['a']['w'] for s, m in zip(cad, ma)], float)
    db = np.array([(m - s['b']['n']) * s['b']['w'] for s, m in zip(cad, mb)], float)
    pa = np.concatenate([[0.0], np.cumsum(da)[:-1]])
    sb = np.concatenate([np.cumsum(db[::-1])[:-1][::-1], [0.0]])
    return da, db, pa, sb


def configuracoes(cad, max_exc):
    """Todas as configuracoes com no maximo max_exc osciladores em m != n
    (e |m - n| <= dq_max).  A primeira e a elastica (m = n)."""
    N = len(cad)
    m0a = tuple(s['a']['n'] for s in cad)
    m0b = tuple(s['b']['n'] for s in cad)
    oscs = [('a', k) for k in range(N)] + [('b', k) for k in range(N)]
    alt = [cad[k][c]['finais'] for c, k in oscs]
    yield m0a, m0b
    for r in range(1, max_exc + 1):
        for escolha in itertools.combinations(range(2 * N), r):
            if any(not alt[i] for i in escolha):
                continue
            for valores in itertools.product(*[alt[i] for i in escolha]):
                ma, mb = list(m0a), list(m0b)
                for i, v in zip(escolha, valores):
                    c, k = oscs[i]
                    (ma if c == 'a' else mb)[k] = v
                yield tuple(ma), tuple(mb)


# ======================================================================
# Amplitudes de um foton projetadas no foton ideal   [termo livre, Eq. (N10)]
#   tau_a(m_a) = int dnu xi~_a^*(nu - D_a) xi(nu) prod_k t^(ka)(nu - pa[k])
#   tau_b(m_b) = int dnu xi~_b^*(nu - D_b) xi(nu) prod_k t^(kb)(nu - sb[k])
# (D_c = deslocamento total do foton c; ele sai com cor nu - D_c.)
# ======================================================================
def tau_foton(cad, c, m, sigma, h, n):
    d = np.array([(mk - s[c]['n']) * s[c]['w'] for s, mk in zip(cad, m)], float)
    if c == 'a':
        antes = np.concatenate([[0.0], np.cumsum(d)[:-1]])
    else:
        antes = np.concatenate([np.cumsum(d[::-1])[:-1][::-1], [0.0]])
    D = d.sum()
    if np.exp(-D**2 / (8 * sigma**2)) < tol_sobreposicao:
        return 0.0
    nu, i0 = pontos(sigma, h, n, 0.0)
    f = ref_rede(cad, c, sigma, h, n, i0, D) * xi_in(nu, sigma)
    for k, (mk, sk) in enumerate(zip(m, antes)):
        f = f * t_rede(cad, k, c, mk, h, n, i0, sk)
    return np.sum(f) * h


def prob_um_foton(cad, c, sigma, max_exc=None):
    """P_c = sum_m |tau_c(m)|^2 truncada (no maximo max_exc osciladores do foton c com m != n):
    probabilidade de o foton c, sozinho, sair igual ao foton ideal."""
    max_exc = max_excitados if max_exc is None else max_exc
    h, n = grade(sigma, len(cad))
    _memo.clear()
    N = len(cad)
    m0 = tuple(s[c]['n'] for s in cad)
    alt = [s[c]['finais'] for s in cad]
    P = abs(tau_foton(cad, c, m0, sigma, h, n))**2
    for r in range(1, max_exc + 1):
        for escolha in itertools.combinations(range(N), r):
            for valores in itertools.product(*[alt[i] for i in escolha]):
                m = list(m0)
                for i, v in zip(escolha, valores):
                    m[i] = v
                P += abs(tau_foton(cad, c, tuple(m), sigma, h, n))**2
    _memo.clear()
    return P


def prob_um_foton_exata(cad, c, sigma, npts=201):
    """P_c somada sobre TODOS os m (todas as ordens, |m-n| <= dq_max), sem
    enumerar configuracoes.  Como cada oscilador so e tocado uma vez, tracar
    os osciladores sitio a sitio e aplicar um canal ao foton:
        R_q(nu, nu') -> sum_m t_m(nu - q) t_m^*(nu' - q) R_q(nu, nu'),  q -> q + delta_m,
    em que q e o deslocamento acumulado (o mesmo no ket e no bra, porque o m
    e o mesmo).  No fim, P_c = sum_q <xi~(. - q)| R_q |xi~(. - q)>.
    Serve para checar o truncamento `max_excitados` (so a parte de um foton)."""
    nu = np.linspace(omega_c - k_sigma * sigma, omega_c + k_sigma * sigma, npts)
    h = nu[1] - nu[0]
    xi = xi_in(nu, sigma)
    R = {0.0: np.outer(xi, xi).astype(complex)}
    ordem = cad if c == 'a' else cad[::-1]
    for sit in ordem:
        o = sit[c]
        novo = {}
        for q, M in R.items():
            for m in [o['n']] + o['finais']:
                t = t_mn(sit, o, m, nu - q)
                qq = round(q + (m - o['n']) * o['w'], 12)
                novo[qq] = novo.get(qq, 0) + np.outer(t, np.conj(t)) * M
        R = novo
    P = 0.0
    for q, M in R.items():
        r = xi_ideal_conj(cad, c, nu - q, sigma)
        P += np.real(r @ M @ np.conj(r)) * h * h
    return P


# ======================================================================
# Termo de interacao no sitio j   [Eq. (N11)]
#
#   N_j(m) = -i gamma_j^2/pi  sum_{ka,kb} F^(ja)_{ka ma}F^(ja)_{ka na}F^(jb)_{kb mb}F^(jb)_{kb nb}
#            int dE  chiK_{ka kb}(E) P(E) Q(E)
#
# Variaveis: u = x_a^(j), v = x_b^(j) (cores de entrada no sitio j),
# y = y_a^(j), z = y_b^(j) (cores de saida do sitio j).  A delta de energia
# total vira  y + z = u + v - D_j  (D_j = delta_a^(j) + delta_b^(j)), e
#   P(E) = int du alpha(u) beta(E - u),          E = u + v
#   Q(E) = int dy alpha'(y) beta'(E - D_j - y)
#   alpha(u)  = xi(u + pa[j]) prod_{k<j} t^(ka)(u + pa[j] - pa[k]) / Gamma^(ja)_{ka na}(u)     a antes
#   beta(v)   = xi(v + sb[j]) prod_{k>j} t^(kb)(v + sb[j] - sb[k]) / Gamma^(jb)_{kb nb}(v)     b antes
#   alpha'(y) = xi~_a^*(y - e_a) prod_{k>j} t^(ka)(y - (pa[k] - pa[j+1])) / Gamma^(ja)_{ka ma}(y)   a depois
#   beta'(z)  = xi~_b^*(z - e_b) prod_{k<j} t^(kb)(z - (sb[k] - sb[j] - delta_b^(j))) / Gamma^(jb)_{kb mb}(z)   b depois
#   e_a = sum_{l>j} delta_a^(l),  e_b = sum_{l<j} delta_b^(l)
# (omega_a = y - e_a e omega_b = z - e_b sao as cores finais).
# As convolucoes sao feitas por FFT; cada funcao e amostrada na rede
# omega_c + h Z, transladada para o centro do seu suporte.
# ======================================================================
def termo_interacao(cad, j, ma, mb, sigma, h, n, desl):
    da, db, pa, sb = desl
    N = len(cad)
    sit = cad[j]
    oa, ob = sit['a'], sit['b']
    Da, Db = da.sum(), db.sum()
    ea = Da - pa[j] - da[j]
    eb = Db - sb[j] - db[j]
    Dj = da[j] + db[j]

    # grades (indice inicial na rede omega_c + h Z)
    u, iu = pontos(sigma, h, n, pa[j])
    v, iv = pontos(sigma, h, n, sb[j])
    y, iy = pontos(sigma, h, n, -ea)
    z, iz = pontos(sigma, h, n, -(eb + Dj))   # beta'' (z) = beta'(z - D_j)
    zz = z - Dj

    fa = xi_in(u + pa[j], sigma)
    for k in range(j):
        fa = fa * t_rede(cad, k, 'a', ma[k], h, n, iu, pa[k] - pa[j])
    fb = xi_in(v + sb[j], sigma)
    for k in range(j + 1, N):
        fb = fb * t_rede(cad, k, 'b', mb[k], h, n, iv, sb[k] - sb[j])
    ga = ref_rede(cad, 'a', sigma, h, n, iy, ea)
    pa_j1 = pa[j] + da[j]
    for k in range(j + 1, N):
        ga = ga * t_rede(cad, k, 'a', ma[k], h, n, iy, pa[k] - pa_j1)
    gb = ref_rede(cad, 'b', sigma, h, n, iz, Dj + eb)
    for k in range(j):
        gb = gb * t_rede(cad, k, 'b', mb[k], h, n, iz, Dj + sb[k] - sb[j] - db[j])

    # pares de niveis vestidos (ka, kb) com peso F^a F^a F^b F^b relevante
    kas, kbs = niveis(oa, ma[j]), niveis(ob, mb[j])
    pesos = np.outer(oa['niv'][ma[j]][1], ob['niv'][mb[j]][1])
    ia, ib = np.nonzero(np.abs(pesos) > eps_peso)
    if len(ia) == 0:
        return 0.0
    # alinha Q com P na rede de E:  P[p] <-> indice iu+iv+p,  Q[q] <-> iy+iz+q
    off = (iy + iz) - (iu + iv)
    p0, p1 = max(0, off), min(2 * n - 1, 2 * n - 1 + off)
    if p1 <= p0:
        return 0.0

    Nf = next_fast_len(2 * n - 1)
    A = fft(np.array([fa / Gam(sit, oa, k, oa['n'], u) for k in kas]), Nf, axis=-1)
    B = fft(np.array([fb / Gam(sit, ob, k, ob['n'], v) for k in kbs]), Nf, axis=-1)
    A2 = fft(np.array([ga / Gam(sit, oa, k, ma[j], y) for k in kas]), Nf, axis=-1)
    B2 = fft(np.array([gb / Gam(sit, ob, k, mb[j], zz) for k in kbs]), Nf, axis=-1)
    P = ifft(A[ia] * B[ib], axis=-1)[:, p0:p1] * h                # P[par, p]
    Q = ifft(A2[ia] * B2[ib], axis=-1)[:, p0 - off:p1 - off] * h

    chave = ('K', j, ma[j], mb[j], iu + iv + p0, p1 - p0)
    if chave not in _memo:
        E = 2 * omega_c + h * (iu + iv + np.arange(p0, p1))
        _memo[chave] = chiK(sit, oa, ob, kas[ia][:, None], kbs[ib][:, None], E[None, :])
    tot = np.sum(pesos[ia, ib] * np.sum(_memo[chave] * P * Q, axis=-1))
    return -1j * sit['gam']**2 / np.pi * tot * h


def termo_nao_linear(cad, ma, mb, sigma, h, n):
    """N(m) = sum_j N_j(m): a parte de interacao projetada no par ideal."""
    desl = deslocamentos(cad, ma, mb)
    D = desl[0].sum() + desl[1].sum()
    if np.exp(-D**2 / (16 * sigma**2)) < tol_sobreposicao:
        return 0.0
    return sum(termo_interacao(cad, j, ma, mb, sigma, h, n, desl) for j in range(len(cad)))


# ======================================================================
# Fidelidade media de porta
# ======================================================================
# Estado de entrada |psi> = sum_k c_k |k>, k em (00, 01, 10, 11).  A amplitude
# de achar o estado ideal com os osciladores em |m> e A_m = sum_k |c_k|^2 x_k(m):
#   x00 = d(ma,na) d(mb,nb),    x01 = d(ma,na) tau_b(mb),    x10 = tau_a(ma) d(mb,nb),
#   x11 = e^{i phi} [tau_a(ma) tau_b(mb) + N(m)].
# Media de Haar em 2 qubits:  F1 = sum_m x(m)^dagger W1 x(m),  W1 = (J + I)/20.
# Sem mola (x = (1, 1, 1, e^{i phi} Ov)) isso e a Eq. (A2):
#   F1 = [6 + 3 Re(e^{i phi} Ov) + |Ov|^2] / 10.
W1 = (np.ones((4, 4)) + np.eye(4)) / 20


def espelho_simetrico(cad):
    """True se a cadeia e invariante pela troca a <-> b com inversao da ordem
    dos sitios (o foton a vai 1 -> N e o b vai N -> 1).  Nesse caso a
    configuracao (ma, mb) contribui igual a (mb invertida, ma invertida), e so
    metade das configuracoes precisa ser calculada."""
    chave = lambda o: (o['w'], o['S'], o['n'])
    return all(chave(cad[k]['a']) == chave(cad[-1 - k]['b'])
               and all(cad[k][q] == cad[-1 - k][q] for q in ('gam', 'Delta', 'chi'))
               for k in range(len(cad)))


def fidelidade(cad, sigma, max_exc=None, detalhes=False):
    """F1(sigma) da cadeia `cad`, somando sobre as configuracoes com no maximo
    max_exc osciladores excitados.  Com detalhes=True devolve tambem a
    contribuicao de cada configuracao."""
    max_exc = max_excitados if max_exc is None else max_exc
    h, n = grade(sigma, len(cad))
    simetrica = espelho_simetrico(cad)
    _memo.clear()
    m0a = tuple(s['a']['n'] for s in cad)
    m0b = tuple(s['b']['n'] for s in cad)
    cache = {}

    def tau(c, m):
        if (c, m) not in cache:
            cache[(c, m)] = tau_foton(cad, c, m, sigma, h, n)
        return cache[(c, m)]

    F1, partes = 0.0, []
    for ma, mb in configuracoes(cad, max_exc):
        peso = 1.0
        if simetrica:
            espelho = (mb[::-1], ma[::-1])
            if (ma, mb) > espelho:
                continue                 # ja contada pela configuracao espelhada
            if (ma, mb) < espelho:
                peso = 2.0
        ea, eb = ma == m0a, mb == m0b
        ta, tb = tau('a', ma), tau('b', mb)
        Nm = termo_nao_linear(cad, ma, mb, sigma, h, n)
        x = np.array([float(ea and eb), tb * ea, ta * eb,
                      np.exp(1j * phi) * (ta * tb + Nm)], complex)
        f = peso * float(np.real(np.conj(x) @ W1 @ x))
        F1 += f
        if detalhes:
            partes.append((ma, mb, f, ta, tb, Nm))
    _memo.clear()
    return (F1, partes) if detalhes else F1


# ======================================================================
# Validacoes
# ======================================================================
def check(nome, ok, detalhe):
    print(f'  [{"OK " if ok else "FALHOU"}] {nome}: {detalhe}')
    return ok


def forca_bruta_N(cad, ma, mb, sigma, npts=121, ks=6.0):
    """N(m) por integracao direta em (nu_a, nu_b, omega_a), com a Eq. (N11)
    escrita nas cores dos sitios, sem nenhuma reducao a convolucoes."""
    N = len(cad)
    da, db, pa, sb = deslocamentos(cad, ma, mb)
    Dtot = da.sum() + db.sum()
    x = np.linspace(omega_c - ks * sigma, omega_c + ks * sigma, npts)
    hh = x[1] - x[0]
    nua, nub, wa = np.meshgrid(x, x, x, indexing='ij', sparse=True)
    wb = nua + nub - Dtot - wa
    tot = 0.0
    for j in range(N):
        sit, oa, ob = cad[j], cad[j]['a'], cad[j]['b']
        f = (xi_in(nua, sigma) * xi_in(nub, sigma)
             * xi_ideal_conj(cad, 'a', wa, sigma) * xi_ideal_conj(cad, 'b', wb, sigma))
        xa = nua - pa[j]                                    # x_a^(j)
        xb = nub - sb[j]                                    # x_b^(j)
        ya = wa + sum(da[l] for l in range(j + 1, N))       # y_a^(j)
        yb = wb + sum(db[l] for l in range(j))              # y_b^(j)
        for k in range(j):
            f = f * t_mn(cad[k], cad[k]['a'], ma[k], nua - pa[k])                       # a antes
            f = f * t_mn(cad[k], cad[k]['b'], mb[k], wb + sum(db[l] for l in range(k + 1)))  # b depois: y_b^(k+1)
        for k in range(j + 1, N):
            f = f * t_mn(cad[k], cad[k]['b'], mb[k], nub - sb[k])                       # b antes
            f = f * t_mn(cad[k], cad[k]['a'], ma[k], wa + sum(da[l] for l in range(k, N)))  # a depois: y_a^(k-1)
        C = 0.0
        for ka in niveis(oa, ma[j]):
            for kb in niveis(ob, mb[j]):
                w_ = (oa['F'][ka, ma[j]] * oa['F'][ka, oa['n']]
                      * ob['F'][kb, mb[j]] * ob['F'][kb, ob['n']])
                C = C + w_ * chiK(sit, oa, ob, ka, kb, xa + xb) / (
                    Gam(sit, oa, ka, oa['n'], xa) * Gam(sit, ob, kb, ob['n'], xb)
                    * Gam(sit, oa, ka, ma[j], ya) * Gam(sit, ob, kb, mb[j], yb))
        tot += -1j * sit['gam']**2 / np.pi * np.sum(f * C) * hh**3
    return tot


def validar():
    print('=== Validacoes ===')
    ok = True

    # 1) S = 0: deve reproduzir a Fig. 3(a) de Brod-Combes (infidelidade/, L = 0)
    ref = os.path.join(here, '..', 'infidelidade', 'data', 'fig3_L0_chi1000_gamma1_omega0.npz')
    if os.path.exists(ref) and chi == 1000 and gam == 1 and omega0 == 0 and Delta == 0:
        d = np.load(ref)
        for N, s in [(1, 0.3), (3, 0.1), (10, 0.05)]:
            i = list(d['N_fig3a']).index(N)
            Fref = np.interp(np.log(s), np.log(d['sigma_fig3a']), d['F_curves'][i])
            F = fidelidade(montar_cadeia(N, 1, 0, 0, 1, 0, 0), s)
            ok &= check(f'S=0 vs Brod-Combes (N={N}, sigma={s:g})', abs(F - Fref) < 2e-4,
                        f'F={F:.6f}, ref={Fref:.6f}')

    # 2) N = 1, soma completa em m: deve reproduzir o codigo de um sitio
    pasta1 = os.path.join(here, '..', '2foton-1sitio-fock')
    if os.path.exists(os.path.join(pasta1, 'calcular_fock.py')):
        sys.path.insert(0, pasta1)
        import calcular_fock as c1
        sys.path.pop(0)
        g = globals()
        salvo = {k: g[k] for k in ('chi', 'gam', 'Delta', 'omega0', 'omega_c', 'dq_max')}
        g.update(chi=c1.chi, gam=c1.gam, Delta=c1.Delta, omega0=c1.omega0,
                 omega_c=c1.Delta + c1.omega0, dq_max=4)
        try:
            for S, n0, s in [(0.1, 0, 1.0), (0.1, 1, 2.0), (0.1, 2, 0.5)]:
                cad = montar_cadeia(1, 1.0, S, n0, 1.0, S, n0)
                F = fidelidade(cad, s, max_exc=2)
                m1 = c1.mola(1.0, S, n0)
                Fr = c1.fidelidades(m1, m1, s)[0]
                ok &= check(f'N=1 vs codigo de um sitio (S={S}, n={n0}, sigma={s:g})',
                            abs(F - Fr) < 1e-4, f'F={F:.6f}, ref={Fr:.6f}')
        finally:
            g.update(salvo)

    # 3) termo de interacao com deslocamentos (m != n) por FFT vs forca bruta 3D
    cad = montar_cadeia(3, 0.3, 0.2, 1, 0.4, 0.15, 1)
    cad[1]['gam'], cad[2]['Delta'], cad[0]['chi'] = 1.3, 0.1, 2.0
    s = 0.4
    h, n = grade(s, 3)
    for ma, mb in [((1, 1, 1), (1, 1, 1)), ((2, 1, 0), (1, 1, 1)), ((1, 0, 1), (2, 1, 0))]:
        Nf = termo_nao_linear(cad, ma, mb, s, h, n)
        Nb = forca_bruta_N(cad, ma, mb, s)
        ok &= check(f'N(m) FFT vs forca bruta (N=3, ma={ma}, mb={mb})',
                    abs(Nf - Nb) < 2e-3 * max(abs(Nb), 1e-2), f'FFT={Nf:.5f}, direto={Nb:.5f}')

    # 4) conservacao da probabilidade de um foton: sum_m int |xi|^2 |prod t|^2 = 1
    salvo = dq_max
    globals()['dq_max'] = 5
    try:
        cad = montar_cadeia(2, 1.0, 0.1, 1, 1.0, 0.1, 1)
        nu = np.linspace(omega_c - 20, omega_c + 20, 40001)
        hh = nu[1] - nu[0]
        tot = 0.0
        for ma in itertools.product(*[[sit['a']['n']] + sit['a']['finais'] for sit in cad]):
            d = [(m - sit['a']['n']) * sit['a']['w'] for sit, m in zip(cad, ma)]
            f = xi_in(nu, 1.0)**2
            f = f * np.abs(t_mn(cad[0], cad[0]['a'], ma[0], nu))**2
            f = f * np.abs(t_mn(cad[1], cad[1]['a'], ma[1], nu - d[0]))**2
            tot += np.sum(f) * hh
        ok &= check('probabilidade de um foton (N=2, S=0.1, n=1, |m-n|<=5)', abs(tot - 1) < 1e-5,
                    f'sum_m int|psi_m|^2 = {tot:.7f}')
    finally:
        globals()['dq_max'] = salvo

    print('=== Todas as validacoes passaram ===\n' if ok else '=== HA VALIDACOES FALHANDO ===\n')
    return ok


# ======================================================================
# Varredura e saida
# ======================================================================
def nome_arquivo(N, w_a, S_a, n_a, w_b, S_b, n_b):
    return (f"fid_N{N}_chi{chi:g}_gam{gam:g}_w0{omega0:g}"
            f"_wa{w_a:g}_Sa{S_a:g}_na{n_a}_wb{w_b:g}_Sb{S_b:g}_nb{n_b}"
            f"_exc{max_excitados}_{referencia}.npz")


def calcular_caso(N, w_a, S_a, n_a, w_b, S_b, n_b):
    cad = montar_cadeia(N, w_a, S_a, n_a, w_b, S_b, n_b)
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    t0 = time.time()
    if n_proc > 1:
        with ProcessPoolExecutor(n_proc) as ex:
            F1s = np.array(list(ex.map(fidelidade, [cad] * len(sigmas), sigmas)))
    else:
        F1s = np.array([fidelidade(cad, s) for s in sigmas])
    for s, F1 in zip(sigmas, F1s):
        if F1 > 1 + 1e-6:
            print(f'AVISO: F1={F1:.5f} > 1 em sigma={s:.3f} (aumente pts_per_width/k_sigma)')
    k = int(np.argmax(F1s))
    if not quiet:
        print(f'  N={N:2d}: F1_max={F1s[k]:.5f} em sigma={sigmas[k]:.4f}   [{time.time() - t0:.1f} s]',
              flush=True)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, nome_arquivo(N, w_a, S_a, n_a, w_b, S_b, n_b))
    np.savez(path, sigmas=sigmas, F1=F1s, N=N,
             chi=chi, gam=gam, Delta=Delta, omega0=omega0, phi=phi,
             w_a=w_a, S_a=S_a, n_a=n_a, w_b=w_b, S_b=S_b, n_b=n_b,
             max_excitados=max_excitados, dq_max=dq_max, referencia=referencia,
             k_sigma=k_sigma, pts_per_width=pts_per_width)
    return path


def main():
    if run_validations:
        validar()
    for c in casos:
        if not quiet:
            print('caso (w_a, S_a, n_a, w_b, S_b, n_b) =', c)
        for N in N_values:
            calcular_caso(N, *c)


if __name__ == '__main__':
    main()
