# -*- coding: utf-8 -*-
"""
Fidelidade da porta CPHASE de UM sitio de Brod-Combes em que cada atomo
(a e b) esta acoplado a uma "mola" (modo de banho) inicialmente num estado de
Fock |n>.  Calcula F1(sigma) e F2(sigma) (como na Fig. 5 de Brod & Combes) e
salva os dados em data/*.npz para serem plotados por plot.py.

Matriz S de dois fotons e um sitio (mola a: |n_a> -> |m_a>, mola b: |n_b> -> |m_b>):

  S = t^a_{ma na}(nu_a) t^b_{mb nb}(nu_b) delta(...) delta(...)
      - i chi gamma^2/pi delta(wa + wb + (ma-na)w^a + (mb-nb)w^b - nu_a - nu_b)
        * sum_{ka,kb} F^a_{ka ma} F^a_{ka na} F^b_{kb mb} F^b_{kb nb}
          (1 + 2 i chi / Sigma_{ka kb})^{-1}
          / [Gam^a_{ka na}(nu_a) Gam^b_{kb nb}(nu_b) Gam^a_{ka ma}(wa) Gam^b_{kb mb}(wb)]

Como a mola termina num estado |m> que o foton nao "ve", o estado dos fotons
na saida e MISTO.  O alvo (porta CPHASE ideal) e PURO, entao a fidelidade e
F = <psi_id| rho |psi_id>  (Wikipedia, "Fidelity of quantum states"),
media sobre os estados de entrada (Haar).
"""
import os

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.special import eval_genlaguerre, factorial

# ----------------------------------------------------------------------
# Parametros do sitio (iguais aos da Fig. 5 de Brod & Combes)
# ----------------------------------------------------------------------
chi = 5              # acoplamento nao linear (cross-Kerr) entre os atomos a e b
gam = 4.5            # gamma: taxa de emissao de cada atomo no guia
Delta = 0.0          # dessintonia dos atomos
omega0 = 1.1         # frequencia central do pacote, relativa a Delta
phi = np.pi          # fase da porta CPHASE ideal

omega_c = Delta + omega0   # frequencia central do pacote (derivada, nao editar)

# ----------------------------------------------------------------------
# Casos calculados: (w_a, S_a, n_a, w_b, S_b, n_b)
#   w: frequencia da mola,  S = (2g/w)^2: fator de Huang-Rhys,  n: estado de Fock inicial
# S = 0 desliga a mola e reproduz a curva "single site" da Fig. 5.
# ----------------------------------------------------------------------
casos = [
    (1, 0.0,    0,     1, 0.0,    0),
    (1, 0.25,   0,    1, 0.25,   0),
    (1, 0.25,   1,    1, 0.25,   1),
    (1, 0.25,   2,    1, 0.25,   2),
]



# ----------------------------------------------------------------------
# Correcao da deformacao de foton unico (o que conta como "foton ideal" na saida)
#   'media'    : fase da amplitude elastica t_nn(nu) (recai em -Gam*/Gam de
#                Brod-Combes quando g = 0)
#   'sem_mola' : fase de Brod-Combes sem mola, -Gam*/Gam com Gam = gam/2 + i(Delta - nu)
# ----------------------------------------------------------------------
referencia = 'media'

# ----------------------------------------------------------------------
# Varredura em sigma
# ----------------------------------------------------------------------
sigma_min = 0.03     # menor largura do pacote
sigma_max = 100.0    # maior largura do pacote
n_sigma = 100         # numero de pontos (escala log)

# ----------------------------------------------------------------------
# Parametros numericos
# ----------------------------------------------------------------------
k_sigma = 9.0        # metade da janela de frequencias, em unidades de sigma
pts_per_width = 10   # pontos por menor escala, min(sigma, gamma)
n_min = 501          # numero minimo de pontos na grade
n_max = 20001        # teto de pontos (protecao de tempo/memoria)
eps_fc = 1e-6        # |F_kn| abaixo disso e tratado como zero

outdir = 'data'      # pasta onde os .npz sao salvos
quiet = False        # se True, nao imprime o progresso


# ----------------------------------------------------------------------
# Mola: fator de Franck-Condon e cortes
# ----------------------------------------------------------------------
def FC(k, n, beta):
    """Fator de Franck-Condon <k|D(beta)|n>, beta real."""
    if k >= n:
        return (np.sqrt(factorial(n) / factorial(k)) * beta**(k - n)
                * np.exp(-beta**2 / 2) * eval_genlaguerre(n, k - n, beta**2))
    return (-1)**(n - k) * FC(n, k, beta)


def mola(w, S, n):
    """Dicionario com tudo que descreve uma mola de frequencia w, fator de
    Huang-Rhys S = (2g/w)^2, que comeca no estado de Fock |n>.

    So entram na conta os k com |F_kn| > eps_fc e os m alcancaveis a partir
    desses k. Guardamos as faixas de j = k - n e d = m - n, pois Gamma_kn e o
    deslocamento de frequencia do foton so dependem delas."""
    g = np.sqrt(S) * w / 2
    beta = 2 * g / w
    K = n + 6
    while True:
        F = np.array([[FC(k, m, beta) for m in range(K)] for k in range(K)])  # F[k, m]
        ativo = np.abs(F) > eps_fc
        ks = ativo[:, n]                        # k acessiveis a partir de |n>
        ms = ativo[ks].any(axis=0)              # m acessiveis desses k
        if not ms[-3:].any():                   # folga: a faixa nao encosta no corte K
            break
        K += 5
    kidx, midx = np.nonzero(ks)[0], np.nonzero(ms)[0]
    j = np.arange(kidx.min(), kidx.max() + 1) - n   # j = k - n possiveis
    d = np.arange(midx.min(), midx.max() + 1) - n   # d = m - n possiveis
    # c_j = F_{n+j,n}^2: pesos da amplitude elastica t_nn
    c = np.array([F[n + jj, n]**2 for jj in j])
    return dict(w=w, S=S, n=n, g=g, desloc=S * w, K=K, F=F, j=j, d=d, c=c)


def Gam(ml, j, nu):
    """Gamma_{kn}(nu) = gam/2 + i(Delta - 4g^2/w + (k-n) w - nu), com j = k - n."""
    return gam / 2 + 1j * (Delta - ml['desloc'] + j * ml['w'] - nu)


def t_mn(ml, m, n, nu):
    """Amplitude de foton unico t_mn(nu) = delta_mn - gam sum_k F_km F_kn / Gamma_kn(nu)."""
    F = ml['F']
    s = sum(F[k, m] * F[k, n] / Gam(ml, k - n, nu) for k in range(ml['K'])
            if F[k, m] * F[k, n] != 0)
    return (1.0 if m == n else 0.0) - gam * s


def fase_ref(ml, nu):
    """Fase u(nu), |u| = 1, que define o foton 'ideal' de saida xi~ = u xi."""
    if referencia == 'sem_mola':
        G0 = gam / 2 + 1j * (Delta - nu)
        return -np.conj(G0) / G0
    # t_nn(nu) = 1 - gam sum_j c_j / Gamma_j(nu)
    tm = 1.0 - gam * sum(cj / Gam(ml, jj, nu) for jj, cj in zip(ml['j'], ml['c']))
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
# Funcoes do foton de saida.  Um foton que sai com a mola em |m> (entrou com
# ela em |n>) tem frequencia w = w' - d*w_mola, d = m - n.  Usamos w' como
# variavel: Gamma_km(w) = Gamma_kn(w'), e o foton ideal vira xi~*(w' - d w_mola).
# Amostramos w' na grade de entrada transladada de i_d = round(d w_mola/h)
# pontos, para que tudo fique numa grade comum de espacamento h.
# ----------------------------------------------------------------------
def saida(ml, sigma, w, h):
    """ref[d] = xi~*(w' - d w_mola) nos pontos w' = w + i_d h, e os i_d."""
    i_d = np.rint(ml['d'] * ml['w'] / h).astype(int)
    ref = np.empty((len(ml['d']), len(w)), complex)
    for a, (dd, ii) in enumerate(zip(ml['d'], i_d)):
        x = w + ii * h - dd * ml['w']
        ref[a] = np.conj(fase_ref(ml, x) * xi_in(x, sigma))
    return ref, i_d


# ----------------------------------------------------------------------
# Amplitudes de um foton:
#   tau_mn = int dnu xi~*(nu - d w) xi(nu) t_mn(nu)
#          = delta_mn O - gam sum_k F_km F_kn L[k-n, m-n]
#   L[j, d] = int dnu xi~*(nu - d w) xi(nu) / Gamma_j(nu)
# ----------------------------------------------------------------------
def amplitudes_1foton(ml, sigma, w, h):
    """tau[m] = tau_{m n} para a mola que comeca em |n>."""
    xi = xi_in(w, sigma)
    n, K, F = ml['n'], ml['K'], ml['F']
    O = np.sum(np.conj(fase_ref(ml, w) * xi) * xi) * h
    L = np.empty((len(ml['j']), len(ml['d'])), complex)
    for b, dd in enumerate(ml['d']):
        x = w - dd * ml['w']
        ref = np.conj(fase_ref(ml, x) * xi_in(x, sigma)) * xi
        for a, jj in enumerate(ml['j']):
            L[a, b] = np.sum(ref / Gam(ml, jj, w)) * h
    tau = np.zeros(K, complex)
    for m in range(K):
        dd = m - n
        if not ml['d'][0] <= dd <= ml['d'][-1]:
            continue
        s = sum(F[k, m] * F[k, n] * L[k - n - ml['j'][0], dd - ml['d'][0]]
                for k in range(K) if ml['j'][0] <= k - n <= ml['j'][-1])
        tau[m] = (O if m == n else 0.0) - gam * s
    return tau


# ----------------------------------------------------------------------
# Termo nao linear.
#
# Com wa' = wa + da, wb' = wb + db (da = (ma-na) w^a), a delta vira
# wa' + wb' = nu_a + nu_b = E, e
#
#   N[ma,mb] = -i chi gam^2/pi sum_{ka,kb} F^a_{ka ma}F^a_{ka na}F^b_{kb mb}F^b_{kb nb}
#              T[ka-na, kb-nb, ma-na, mb-nb]
#   T[ja,jb,da,db] = int dE  Kf(E) P(E) Q(E)
#   P(E) = int dnu  [xi/Gam^a_ja](nu) [xi/Gam^b_jb](E - nu)
#   Q(E) = int dw'  [xi~*(.-da w^a)/Gam^a_ja](w') [xi~*(.-db w^b)/Gam^b_jb](E - w')
#   Kf(E) = 1/(1 + 2 i chi/Sigma),  Sigma = Gam^a_ja(nu_a) + Gam^b_jb(nu_b) (so depende de E)
#
# T so depende das diferencas j = k - n e d = m - n: e calculado UMA vez por
# sigma; os F de Franck-Condon entram depois.  As convolucoes e a integral
# em E sao feitas por FFT (comprimento Nf >= 4n, sem aliasing).
# ----------------------------------------------------------------------
def tabela_T(ma_, mb_, sigma, w, h):
    n = len(w)
    Nf = next_fast_len(4 * n)
    f = np.arange(Nf)
    xi = xi_in(w, sigma)
    E = 2 * w[0] + np.arange(2 * n - 1) * h

    def transformadas(ml):
        A = np.array([fft(xi / Gam(ml, jj, w), Nf) for jj in ml['j']])     # A[j, f]
        ref, i_d = saida(ml, sigma, w, h)
        C = np.empty((len(ml['j']), len(ml['d']), Nf), complex)           # C[j, d, f]
        for b, ii in enumerate(i_d):
            x = w + ii * h
            fase = np.exp(-2j * np.pi * f * ii / Nf)   # translacao de i_d pontos
            for a, jj in enumerate(ml['j']):
                C[a, b] = fft(ref[b] / Gam(ml, jj, x), Nf) * fase
        return A, C, i_d

    Aa, Ca, ia = transformadas(ma_)
    Ab, Cb, ib = transformadas(mb_)

    T = np.empty((len(ma_['j']), len(mb_['j']), len(ma_['d']), len(mb_['d'])), complex)
    for a, ja in enumerate(ma_['j']):
        R = np.empty((len(mb_['j']), Nf), complex)
        for b, jb in enumerate(mb_['j']):
            P = ifft(Aa[a] * Ab[b])[:2 * n - 1] * h
            Sig = gam + 1j * (2 * Delta - ma_['desloc'] - mb_['desloc']
                              + ja * ma_['w'] + jb * mb_['w'] - E)
            PK = P / (1 + 2j * chi / Sig)
            R[b] = np.conj(fft(np.conj(PK), Nf))      # sum_p PK_p e^{+2 pi i f p/Nf}
        # int dE Kf P Q = (h/Nf) sum_f R(f) Qhat(f),  Qhat = h Ca Cb
        X = R[:, None, :] * Ca[a][None, :, :]                     # X[jb, da, f]
        T[a] = np.matmul(X, np.transpose(Cb, (0, 2, 1))) * h * h / Nf   # [jb, da, db]
    # sem sobreposicao entre P e Q na grade de E: o resultado e zero
    T[:, :, np.abs(ia[:, None] + ib[None, :]) >= 2 * n - 1] = 0.0
    return T


def termo_nao_linear(ma_, mb_, T):
    """N[ma, mb] para molas iniciando em (na, nb), montado a partir de T."""
    def pesos(ml):
        """G[k, m] = F_km F_kn e os indices de j = k-n e d = m-n em T."""
        F, K, nn = ml['F'], ml['K'], ml['n']
        k = np.arange(K)
        jk = k - nn - ml['j'][0]
        dm = k - nn - ml['d'][0]
        ok_k = (jk >= 0) & (jk < len(ml['j']))
        ok_m = (dm >= 0) & (dm < len(ml['d']))
        G = F[np.ix_(ok_k, ok_m)] * F[ok_k, nn][:, None]
        return G, jk[ok_k], dm[ok_m], ok_m

    Ga, ja, da, oka = pesos(ma_)
    Gb, jb, db, okb = pesos(mb_)
    Tsub = T[np.ix_(ja, jb, da, db)]
    N = np.zeros((ma_['K'], mb_['K']), complex)
    N[np.ix_(oka, okb)] = np.einsum('am,bn,abmn->mn', Ga, Gb, Tsub, optimize=True)
    return -1j * chi * gam**2 / np.pi * N


# ----------------------------------------------------------------------
# Fidelidade media de porta
# ----------------------------------------------------------------------
# Para um estado de entrada |psi> = sum_k c_k |k>, k em (00, 01, 10, 11), e
# molas comecando em |na nb>, a amplitude de encontrar o estado ideal com a
# mola em |ma mb> e  A_m = sum_k |c_k|^2 x_k(m), com
#   x00 = d(ma,na) d(mb,nb),   x01 = d(ma,na) tau^b,   x10 = tau^a d(mb,nb),
#   x11 = e^{i phi} (tau^a tau^b + N).
# O estado dos fotons (mola tracada) e misto: rho = sum_m |out_m><out_m|, e
# F = <psi_id|rho|psi_id> = sum_m |A_m|^2.  A media sobre |psi> e a forma
# quadratica  x^dagger W x  com
#   W1 = (J + I)/20                  (Haar em 2 qubits, d = 4)   -> F1
#   W2 = kron(M, M)/36, M=[[2,1],[1,2]] (Haar em estados produto) -> F2
# Sem mola (x = (1,1,1,e^{i phi}F)) isso recai nas Eqs. (A2) e (A5) do artigo.
W1 = (np.ones((4, 4)) + np.eye(4)) / 20
W2 = np.kron([[2, 1], [1, 2]], [[2, 1], [1, 2]]) / 36


def fidelidades(ma_, mb_, sigma):
    w, h = grade(sigma)
    tau_a = amplitudes_1foton(ma_, sigma, w, h)
    tau_b = amplitudes_1foton(mb_, sigma, w, h)
    T = tabela_T(ma_, mb_, sigma, w, h)
    N = termo_nao_linear(ma_, mb_, T)
    Ka, Kb = ma_['K'], mb_['K']
    da = np.outer(np.arange(Ka) == ma_['n'], np.ones(Kb))
    db = np.outer(np.ones(Ka), np.arange(Kb) == mb_['n'])
    ta = np.outer(tau_a, np.ones(Kb))
    tb = np.outer(np.ones(Ka), tau_b)
    x = np.stack([da * db, da * tb, ta * db,
                  np.exp(1j * phi) * (ta * tb + N)]).reshape(4, -1)
    F1 = np.real(np.einsum('im,ij,jm->', np.conj(x), W1, x))
    F2 = np.real(np.einsum('im,ij,jm->', np.conj(x), W2, x))
    return F1, F2


def nome_arquivo(ma_, mb_):
    return (f"fid_chi{chi:g}_gam{gam:g}_w0{omega0:g}"
            f"_wa{ma_['w']:g}_Sa{ma_['S']:g}_na{ma_['n']}"
            f"_wb{mb_['w']:g}_Sb{mb_['S']:g}_nb{mb_['n']}_{referencia}.npz")


def calcular_caso(w_a, S_a, n_a, w_b, S_b, n_b):
    ma_, mb_ = mola(w_a, S_a, n_a), mola(w_b, S_b, n_b)
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    F1s, F2s = [], []
    for s in sigmas:
        F1, F2 = fidelidades(ma_, mb_, s)
        F1s.append(F1)
        F2s.append(F2)
        if not quiet:
            print(f"  sigma={s:8.3f}   F1={F1:.4f}   F2={F2:.4f}", flush=True)
        if F1 > 1 + 1e-6:
            print(f"AVISO: F1={F1:.5f} > 1 em sigma={s:.3f} (aumente pts_per_width/k_sigma)")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, nome_arquivo(ma_, mb_))
    np.savez(path, sigmas=sigmas, F1=np.array(F1s), F2=np.array(F2s),
             chi=chi, gam=gam, Delta=Delta, omega0=omega0, phi=phi,
             w_a=w_a, S_a=S_a, n_a=n_a, w_b=w_b, S_b=S_b, n_b=n_b,
             K_a=ma_['K'], K_b=mb_['K'], referencia=referencia,
             k_sigma=k_sigma, pts_per_width=pts_per_width)
    if not quiet:
        print(f"Dados salvos em {path}")
    return path




def main():
    for c in casos:
        if not quiet:
            print('caso (w_a, S_a, n_a, w_b, S_b, n_b) =', c)
        calcular_caso(*c)


if __name__ == '__main__':
    main()
