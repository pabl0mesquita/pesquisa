# -*- coding: utf-8 -*-
"""
Fidelidade da porta CPHASE de UM sitio de Brod-Combes em que cada atomo
(a e b) esta acoplado a uma "mola" inicialmente em estado TERMICO, na
aproximacao de mola lenta (quase-estatica), omega^(a,b) << gamma.
Calcula F1(sigma) e F2(sigma) e salva em data/*.npz para plot_termico.py.

Por que uma aproximacao: a fidelidade e linear no estado inicial da mola e o
estado termico e diagonal na base de Fock, entao o resultado exato e
    F_termico = sum_{na,nb} p_na p_nb F_Fock(na, nb),
com p_n = nbar^n/(1+nbar)^(n+1).  Cada F_Fock ja e o calculo completo de
calcular_fock.py, e a soma dupla precisa de ~10^2 pares por sigma.  Isso e
lento demais para uma varredura; fica so em `fidelidades_exato`, usada para
validar a aproximacao em poucos pontos.

Aproximacao de mola lenta: com H_mola = omega b^dag b e acoplamento
2g (b + b^dag) no estado excitado, se a mola nao se mexe durante a interacao
(omega << gamma) o operador x = b + b^dag e conservado e
    U = int dx |x><x| (x) U_delta(x),     delta(x) = 2 g x  (desvio da linha).
Tracando a mola, rho_fotons = int dx P(x) U_delta|psi><psi|U_delta^dag: uma
MISTURA CLASSICA de sitios de Brod-Combes sem mola, cada um com a linha
deslocada de delta.  Para o estado termico P(x) e gaussiana e
    delta ~ N(0, s^2),    s^2 = 4 g^2 (2 nbar + 1) = S omega^2 (2 nbar + 1).
(o deslocamento polaronico -4g^2/omega cancela a media de delta).
"""
import os

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.fft import fft, ifft, next_fast_len

# ----------------------------------------------------------------------
# Parametros do sitio (iguais aos da Fig. 5 de Brod & Combes)
# ----------------------------------------------------------------------
chi = 5.0            # acoplamento nao linear (cross-Kerr) entre os atomos a e b
gam = 4.5            # gamma: taxa de emissao de cada atomo no guia
Delta = 0.0          # dessintonia dos atomos (sem mola)
omega0 = 1.1         # frequencia central do pacote, relativa a Delta
phi = np.pi          # fase da porta CPHASE ideal

omega_c = Delta + omega0   # frequencia central do pacote (derivada, nao editar)

# ----------------------------------------------------------------------
# Correcao da deformacao de foton unico (o que conta como "foton ideal" na saida)
#   'media'    : fase da amplitude elastica media termica  <t_delta(nu)>  (recai
#                em -Gam*/Gam de Brod-Combes quando nao ha mola)
#   'sem_mola' : fase de Brod-Combes sem mola, -Gam*/Gam com Gam = gam/2 + i(Delta - nu)
# ----------------------------------------------------------------------
referencia = 'media'

# ----------------------------------------------------------------------
# Varredura em sigma
# ----------------------------------------------------------------------
sigma_min = 0.03     # menor largura do pacote
sigma_max = 100.0    # maior largura do pacote
n_sigma = 100        # numero de pontos (escala log)

# ----------------------------------------------------------------------
# Parametros numericos
# ----------------------------------------------------------------------
k_sigma = 9.0        # metade da janela de frequencias, em unidades de sigma
pts_per_width = 10   # pontos por menor escala, min(sigma, gamma)
n_min = 501          # numero minimo de pontos na grade
n_max = 20001        # teto de pontos (protecao de tempo/memoria)
n_gh = 40            # nos de Gauss-Hermite na media sobre delta (por mola)
tol_pop = 1e-3       # cauda termica desprezada (so em fidelidades_exato)

outdir = 'data'      # pasta onde os .npz sao salvos
quiet = False        # se True, nao imprime o progresso


# ----------------------------------------------------------------------
# Mola termica lenta: distribuicao do desvio da linha
# ----------------------------------------------------------------------
def largura_desvio(w, S, nbar):
    """s = desvio padrao de delta = 2 g x:  s^2 = 4 g^2 (2 nbar+1) = S w^2 (2 nbar+1)."""
    return w * np.sqrt(S * (2 * nbar + 1))


def nos_desvio(w, S, nbar):
    """Nos e pesos de Gauss-Hermite para  int d delta N(0, s^2)(delta) f(delta)."""
    s = largura_desvio(w, S, nbar)
    if s == 0:
        return np.zeros(1), np.ones(1)
    x, p = hermgauss(n_gh)
    return np.sqrt(2) * s * x, p / np.sqrt(np.pi)


def Gam(delta, nu):
    """Gamma(nu) de um sitio sem mola com a linha deslocada: gam/2 + i(Delta + delta - nu)."""
    return gam / 2 + 1j * (Delta + delta - nu)


def t_1foton(delta, nu):
    """Amplitude de foton unico sem mola, t = 1 - gam/Gamma = -Gamma*/Gamma (|t| = 1)."""
    return 1.0 - gam / Gam(delta, nu)


def fase_ref(deltas, pesos, nu):
    """Fase u(nu), |u| = 1, que define o foton 'ideal' de saida xi~ = u xi."""
    if referencia == 'sem_mola':
        return t_1foton(0.0, nu)
    tm = sum(p * t_1foton(d, nu) for d, p in zip(deltas, pesos))   # <t_delta(nu)>
    a = np.abs(tm)
    return np.where(a > 1e-300, tm / np.where(a > 1e-300, a, 1.0), 1.0)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4) de Brod & Combes."""
    return (1.0 / (2 * np.pi * sigma**2))**0.25 * np.exp(-(w - omega_c)**2 / (4 * sigma**2))


def grade(sigma):
    """Grade de frequencias: janela k_sigma*sigma, espacamento que resolve min(sigma, gamma)."""
    half = k_sigma * sigma
    n = int(2 * half / (min(sigma, gam) / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


# ----------------------------------------------------------------------
# Fidelidade media de porta
# ----------------------------------------------------------------------
# Para desvios fixos (delta_a, delta_b) o sitio e o de Brod-Combes sem mola,
# com as linhas dos atomos deslocadas.  Com o foton ideal xi~ = u xi:
#   tau_a = int dnu xi~*(nu) xi(nu) t_{delta_a}(nu)                (idem b)
#   N = -i chi gam^2/pi int dE Kf(E) P(E) Q(E)
#   P(E) = int dnu [xi/Gam_a](nu) [xi/Gam_b](E - nu)
#   Q(E) = int dw  [xi~*/Gam_a](w) [xi~*/Gam_b](E - w)
#   Kf(E) = 1/(1 + 2 i chi/Sigma),  Sigma = Gam_a(nu_a) + Gam_b(nu_b) = gam + i(2 Delta + delta_a + delta_b - E)
# x = (1, tau_b, tau_a, e^{i phi}(tau_a tau_b + N)) e, como a mola nao muda,
#   F = <psi_id|rho|psi_id>,  media Haar  ->  F_{1,2} = int P(da)P(db) x^dag W_{1,2} x.
# Sem mola (delta = 0, tau = 1) isso recai nas Eqs. (A2) e (A5) do artigo.
W1 = (np.ones((4, 4)) + np.eye(4)) / 20
W2 = np.kron([[2, 1], [1, 2]], [[2, 1], [1, 2]]) / 36


def fidelidades(mola_a, mola_b, sigma):
    """F1, F2 para molas termicas lentas mola_a = (w, S, nbar), mola_b idem."""
    w, h = grade(sigma)
    n = len(w)
    Nf = next_fast_len(2 * n - 1)
    xi = xi_in(w, sigma)
    E = 2 * w[0] + np.arange(2 * n - 1) * h

    def lado(ml):
        d, p = nos_desvio(*ml)
        ref = np.conj(fase_ref(d, p, w) * xi)                     # xi~*(w)
        G = Gam(d[:, None], w[None, :])                           # G[i, w]
        tau = np.sum(ref * xi * t_1foton(d[:, None], w[None, :]), axis=1) * h
        A = fft(xi / G, Nf, axis=1)                                # entrada
        C = fft(ref / G, Nf, axis=1)                               # saida
        return d, p, tau, A, C

    da, pa, ta, Aa, Ca = lado(mola_a)
    db, pb, tb, Ab, Cb = lado(mola_b)

    N = np.empty((len(da), len(db)), complex)
    for i in range(len(da)):
        P = ifft(Aa[i] * Ab, axis=1)[:, :2 * n - 1] * h            # P[j, E]
        Q = ifft(Ca[i] * Cb, axis=1)[:, :2 * n - 1] * h
        Sig = gam + 1j * (2 * Delta + da[i] + db[:, None] - E[None, :])
        N[i] = np.sum(P * Q / (1 + 2j * chi / Sig), axis=1) * h
    N *= -1j * chi * gam**2 / np.pi

    TA, TB = np.meshgrid(ta, tb, indexing='ij')
    x = np.stack([np.ones_like(TA), TB, TA, np.exp(1j * phi) * (TA * TB + N)]).reshape(4, -1)
    peso = np.outer(pa, pb).ravel()
    F1 = np.real(np.einsum('m,im,ij,jm->', peso, np.conj(x), W1, x))
    F2 = np.real(np.einsum('m,im,ij,jm->', peso, np.conj(x), W2, x))
    return F1, F2


# ----------------------------------------------------------------------
# Resultado termico EXATO (qualquer omega), so para validacao: soma dos
# casos de Fock de calcular_fock.py com pesos termicos.  O foton ideal tem que
# ser o mesmo para todos os (na, nb) -- ele nao conhece o estado da mola --,
# entao a fase de referencia e a da amplitude elastica media termica
#   sum_n p_n t_nn(nu) = 1 - gam sum_j cbar_j / Gamma_j(nu),  cbar_j = sum_n p_n F_{n+j,n}^2.
# ----------------------------------------------------------------------
def populacoes(nbar):
    """p_n = nbar^n/(1+nbar)^(n+1), cortada quando a cauda < tol_pop e renormalizada."""
    p = [1.0 / (1.0 + nbar)]
    while 1.0 - sum(p) > tol_pop:
        p.append(p[-1] * nbar / (1.0 + nbar))
    p = np.array(p)
    return p / p.sum()


def fidelidades_exato(mola_a, mola_b, sigma):
    import calcular_fock as cf
    for nome in ('chi', 'gam', 'Delta', 'omega0', 'omega_c', 'phi',
                 'k_sigma', 'pts_per_width', 'n_min', 'n_max'):
        setattr(cf, nome, globals()[nome])
    cf.referencia = 'media'

    def molas(w, S, nbar):
        p = populacoes(nbar)
        ms = [cf.mola(w, S, n) for n in range(len(p))]
        big = ms[-1]                       # maior corte K: contem todos os F_{n+j,n}
        j = np.arange(-len(p) - 30, big['K'])
        c = np.array([sum(pn * big['F'][n + jj, n]**2 for n, pn in enumerate(p)
                          if 0 <= n + jj < big['K']) for jj in j])
        for m in ms:                       # mesma referencia termica para todo n
            m['ref_j'], m['ref_c'] = j, c
        return p, ms

    def fase_termica(ml, nu):
        if referencia == 'sem_mola':
            return t_1foton(0.0, nu)
        tm = 1.0 - gam * sum(cj / cf.Gam(ml, jj, nu)
                             for jj, cj in zip(ml['ref_j'], ml['ref_c']) if cj > 1e-14)
        return tm / np.abs(tm)

    original = cf.fase_ref
    cf.fase_ref = fase_termica
    try:
        pa, Ma = molas(*mola_a)
        pb, Mb = molas(*mola_b)
        F1 = F2 = 0.0
        for qa, ma_ in zip(pa, Ma):
            for qb, mb_ in zip(pb, Mb):
                f1, f2 = cf.fidelidades(ma_, mb_, sigma)
                F1 += qa * qb * f1
                F2 += qa * qb * f2
    finally:
        cf.fase_ref = original
    return F1, F2


# ----------------------------------------------------------------------
# Varredura e saida
# ----------------------------------------------------------------------
def nome_arquivo(mola_a, mola_b):
    (wa, Sa, na), (wb, Sb, nb) = mola_a, mola_b
    return (f"termico_chi{chi:g}_gam{gam:g}_w0{omega0:g}"
            f"_wa{wa:g}_Sa{Sa:g}_nbara{na:g}_wb{wb:g}_Sb{Sb:g}_nbarb{nb:g}_{referencia}.npz")


def calcular_caso(w_a, S_a, nbar_a, w_b, S_b, nbar_b):
    mola_a, mola_b = (w_a, S_a, nbar_a), (w_b, S_b, nbar_b)
    for w in (w_a, w_b):
        if w > 0.2 * gam:
            print(f"AVISO: omega={w:g} nao e << gamma={gam:g}; a aproximacao de mola lenta pode falhar")
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    F1s, F2s = [], []
    for s in sigmas:
        F1, F2 = fidelidades(mola_a, mola_b, s)
        F1s.append(F1)
        F2s.append(F2)
        if not quiet:
            print(f"  sigma={s:8.3f}   F1={F1:.4f}   F2={F2:.4f}", flush=True)
        if F1 > 1 + 1e-6:
            print(f"AVISO: F1={F1:.5f} > 1 em sigma={s:.3f} (aumente pts_per_width/k_sigma)")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, nome_arquivo(mola_a, mola_b))
    np.savez(path, sigmas=sigmas, F1=np.array(F1s), F2=np.array(F2s),
             chi=chi, gam=gam, Delta=Delta, omega0=omega0, phi=phi,
             w_a=w_a, S_a=S_a, nbar_a=nbar_a, w_b=w_b, S_b=S_b, nbar_b=nbar_b,
             s_a=largura_desvio(*mola_a), s_b=largura_desvio(*mola_b),
             referencia=referencia, n_gh=n_gh,
             k_sigma=k_sigma, pts_per_width=pts_per_width)
    if not quiet:
        print(f"Dados salvos em {path}")
    return path


# ----------------------------------------------------------------------
# Casos calculados: (w_a, S_a, nbar_a, w_b, S_b, nbar_b)
#   w: frequencia da mola (deve ser << gam),  S = (2g/w)^2: fator de Huang-Rhys,
#   nbar: ocupacao termica media.  Na aproximacao lenta so importa
#   s = w sqrt(S (2 nbar + 1)), a largura da distribuicao do desvio da linha.
# S = 0 desliga a mola e reproduz a curva "single site" da Fig. 5.
# ----------------------------------------------------------------------
casos = [
    #(w_a, S_a, nbar_a, w_b,  S_b,  nbar_b)
    (1.0,  0.0, 0.0,    1.0,  0.0,  0.0),
    (1.0,  1.0, 0.0,    1.0,  1.0,  0.0),
    (1.0,  1.0, 1.0,    1.0,  1.0,  1.0),
    (1.0,  1.0, 2.0,    1.0,  1.0,  2.0),
]


def main():
    for c in casos:
        if not quiet:
            print('caso (w_a, S_a, nbar_a, w_b, S_b, nbar_b) =', c)
        calcular_caso(*c)


if __name__ == '__main__':
    main()
