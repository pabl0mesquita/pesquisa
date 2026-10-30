# -*- coding: utf-8 -*-
"""
Fidelidade da porta CPHASE de UM sitio de Brod-Combes com cada atomo (a e b)
acoplado a uma "mola" inicialmente em estado TERMICO -- solucao COMPLETA,
sem a aproximacao de mola lenta de calcular_termico.py (vale para qualquer
omega).  Salva F1(sigma), F2(sigma) em data/*.npz para plotar_termico_completo.py.

A fidelidade e linear no estado inicial da mola e o estado termico e
diagonal na base de Fock, entao
    F_termico = sum_{na,nb} p_na p_nb F_Fock(na, nb),
    p_n = nbar^n/(1+nbar)^(n+1)  (cortada quando a cauda < tol_pop),
com F_Fock o calculo completo de ../2foton-1sitio-fock/calcular_fock.py.

O foton ideal de saida tem que ser o mesmo para todo (na, nb) -- ele nao
conhece o estado da mola --, entao a fase de referencia e a da amplitude
elastica media termica
    sum_n p_n t_nn(nu) = 1 - gam sum_j cbar_j / Gamma_j(nu),  cbar_j = sum_n p_n F_{n+j,n}^2.

Truque de custo: a tabela T[ja, jb, da, db] de calcular_fock.py so depende de
j = k - n, d = m - n, sigma e da fase de referencia, nao de n.  Usando para
todo n as faixas UNIAO de j e d, T e calculada uma vez por sigma e cada par
(na, nb) custa so a contracao com os fatores de Franck-Condon.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '2foton-1sitio-fock'))
import calcular_fock as cf  # noqa: E402

# ----------------------------------------------------------------------
# Parametros do sitio (iguais aos da Fig. 5 de Brod & Combes)
# ----------------------------------------------------------------------
chi = 5.0            # acoplamento nao linear (cross-Kerr) entre os atomos a e b
gam = 4.5            # gamma: taxa de emissao de cada atomo no guia
Delta = 0.0          # dessintonia dos atomos
omega0 = 1.1         # frequencia central do pacote, relativa a Delta
phi = np.pi          # fase da porta CPHASE ideal

omega_c = Delta + omega0   # frequencia central do pacote (derivada, nao editar)

# ----------------------------------------------------------------------
# Correcao da deformacao de foton unico (o que conta como "foton ideal" na saida)
#   'media'    : fase da amplitude elastica media termica sum_n p_n t_nn(nu)
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
eps_fc = 1e-6        # |F_kn| abaixo disso e tratado como zero
tol_pop = 1e-3       # cauda termica desprezada: sum_{n > n_max} p_n < tol_pop

outdir = 'data'      # pasta onde os .npz sao salvos
quiet = False        # se True, nao imprime o progresso
n_proc = max(1, min(6, (os.cpu_count() or 1) - 1))   # processos em paralelo na varredura em sigma

# repassa os parametros para calcular_fock (as funcoes de la leem os globais do modulo)
for _nome in ('chi', 'gam', 'Delta', 'omega0', 'omega_c', 'phi', 'referencia',
              'k_sigma', 'pts_per_width', 'n_min', 'n_max', 'eps_fc'):
    setattr(cf, _nome, globals()[_nome])


# ----------------------------------------------------------------------
# Mola termica: populacoes, molas de Fock com faixas uniao e referencia comum
# ----------------------------------------------------------------------
def populacoes(nbar):
    """p_n = nbar^n/(1+nbar)^(n+1), cortada quando a cauda < tol_pop e renormalizada."""
    p = [1.0 / (1.0 + nbar)]
    while 1.0 - sum(p) > tol_pop:
        p.append(p[-1] * nbar / (1.0 + nbar))
    p = np.array(p)
    return p / p.sum()


def molas_termicas(w, S, nbar):
    """Populacoes p_n e as molas de Fock |n> (dicionarios de calcular_fock.mola)
    com as faixas de j e d trocadas pela uniao sobre n e a referencia termica."""
    p = populacoes(nbar) if S > 0 else np.ones(1)   # sem mola o estado nao importa
    ms = [cf.mola(w, S, n) for n in range(len(p))]
    j = np.arange(min(m['j'][0] for m in ms), max(m['j'][-1] for m in ms) + 1)
    d = np.arange(min(m['d'][0] for m in ms), max(m['d'][-1] for m in ms) + 1)
    # cbar_j = sum_n p_n F_{n+j,n}^2 (cada F so vai ate o seu corte K)
    cbar = np.array([sum(pn * m['F'][n + jj, n]**2 for n, (pn, m) in enumerate(zip(p, ms))
                         if 0 <= n + jj < m['K']) for jj in j])
    for m in ms:
        m['j'], m['d'] = j, d
        m['ref_j'], m['ref_c'] = j, cbar
    return p, ms


def fase_termica(ml, nu):
    """Fase u(nu), |u| = 1, do foton 'ideal': a mesma para todo |n> da mola termica."""
    if referencia == 'sem_mola':
        G0 = gam / 2 + 1j * (Delta - nu)
        return -np.conj(G0) / G0
    tm = 1.0 - gam * sum(cj / cf.Gam(ml, jj, nu)
                         for jj, cj in zip(ml['ref_j'], ml['ref_c']) if cj > 1e-14)
    a = np.abs(tm)
    return np.where(a > 1e-300, tm / np.where(a > 1e-300, a, 1.0), 1.0)


cf.fase_ref = fase_termica


# ----------------------------------------------------------------------
# Fidelidade media de porta (mesma forma quadratica de calcular_fock.fidelidades)
# ----------------------------------------------------------------------
def fidelidades_par(ma_, mb_, tau_a, tau_b, T):
    """F1, F2 para molas comecando em |na nb>, com tau e T ja calculados."""
    N = cf.termo_nao_linear(ma_, mb_, T)
    Ka, Kb = ma_['K'], mb_['K']
    da = np.outer(np.arange(Ka) == ma_['n'], np.ones(Kb))
    db = np.outer(np.ones(Ka), np.arange(Kb) == mb_['n'])
    ta = np.outer(tau_a, np.ones(Kb))
    tb = np.outer(np.ones(Ka), tau_b)
    x = np.stack([da * db, da * tb, ta * db,
                  np.exp(1j * phi) * (ta * tb + N)]).reshape(4, -1)
    F1 = np.real(np.einsum('im,ij,jm->', np.conj(x), cf.W1, x))
    F2 = np.real(np.einsum('im,ij,jm->', np.conj(x), cf.W2, x))
    return F1, F2


def fidelidades(termica_a, termica_b, sigma):
    """F1, F2 termicos: soma de Fock com pesos p_na p_nb, T calculada uma vez."""
    (pa, Ma), (pb, Mb) = termica_a, termica_b
    w, h = cf.grade(sigma)
    taus_a = [cf.amplitudes_1foton(m, sigma, w, h) for m in Ma]
    taus_b = [cf.amplitudes_1foton(m, sigma, w, h) for m in Mb]
    T = cf.tabela_T(Ma[0], Mb[0], sigma, w, h)     # so usa j, d, w, desloc e a referencia
    F1 = F2 = 0.0
    for qa, ma_, ta in zip(pa, Ma, taus_a):
        for qb, mb_, tb in zip(pb, Mb, taus_b):
            f1, f2 = fidelidades_par(ma_, mb_, ta, tb, T)
            F1 += qa * qb * f1
            F2 += qa * qb * f2
    return F1, F2


# ----------------------------------------------------------------------
# Varredura e saida
# ----------------------------------------------------------------------
def nome_arquivo(mola_a, mola_b):
    (wa, Sa, na), (wb, Sb, nb) = mola_a, mola_b
    return (f"termico_completo_chi{chi:g}_gam{gam:g}_w0{omega0:g}"
            f"_wa{wa:g}_Sa{Sa:g}_nbara{na:g}_wb{wb:g}_Sb{Sb:g}_nbarb{nb:g}_{referencia}.npz")


def calcular_caso(w_a, S_a, nbar_a, w_b, S_b, nbar_b):
    mola_a, mola_b = (w_a, S_a, nbar_a), (w_b, S_b, nbar_b)
    termica_a, termica_b = molas_termicas(*mola_a), molas_termicas(*mola_b)
    if not quiet:
        print(f"  {len(termica_a[0])} x {len(termica_b[0])} pares de Fock, "
              f"j em [{termica_a[1][0]['j'][0]}, {termica_a[1][0]['j'][-1]}], "
              f"d em [{termica_a[1][0]['d'][0]}, {termica_a[1][0]['d'][-1]}]", flush=True)
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    t0 = time.time()
    if n_proc > 1:
        with ProcessPoolExecutor(n_proc) as ex:
            res = list(ex.map(fidelidades, [termica_a] * len(sigmas), [termica_b] * len(sigmas), sigmas))
    else:
        res = [fidelidades(termica_a, termica_b, s) for s in sigmas]
    F1s, F2s = np.array(res).T
    for s, F1, F2 in zip(sigmas, F1s, F2s):
        if not quiet:
            print(f"  sigma={s:8.3f}   F1={F1:.4f}   F2={F2:.4f}")
        if F1 > 1 + 1e-6:
            print(f"AVISO: F1={F1:.5f} > 1 em sigma={s:.3f} (aumente pts_per_width/k_sigma)")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, nome_arquivo(mola_a, mola_b))
    np.savez(path, sigmas=sigmas, F1=F1s, F2=F2s,
             chi=chi, gam=gam, Delta=Delta, omega0=omega0, phi=phi,
             w_a=w_a, S_a=S_a, nbar_a=nbar_a, w_b=w_b, S_b=S_b, nbar_b=nbar_b,
             n_fock_a=len(termica_a[0]), n_fock_b=len(termica_b[0]), tol_pop=tol_pop,
             referencia=referencia, k_sigma=k_sigma, pts_per_width=pts_per_width)
    if not quiet:
        print(f"Dados salvos em {path}   [{time.time() - t0:.1f} s]")
    return path


# ----------------------------------------------------------------------
# Casos calculados: (w_a, S_a, nbar_a, w_b, S_b, nbar_b)
#   w: frequencia da mola (qualquer, nao precisa ser << gam),
#   S = (2g/w)^2: fator de Huang-Rhys,  nbar: ocupacao termica media.
# S = 0 desliga a mola e reproduz a curva "single site" da Fig. 5.
# O custo cresce com o numero de estados de Fock (~ nbar ln(1/tol_pop)) e com S.
# ----------------------------------------------------------------------
casos = [
    #(w_a, S_a, nbar_a, w_b, S_b, nbar_b)
    (1.0,  0.0, 0.0,    1.0, 0.0, 0.0),
    (1.0,  0.1, 0.0,    1.0, 0.1, 0.0),
    (1.0,  0.1, 1.0,    1.0, 0.1, 1.0),
    (1.0,  0.1, 2.0,    1.0, 0.1, 2.0),
]


def main():
    for c in casos:
        if not quiet:
            print('caso (w_a, S_a, nbar_a, w_b, S_b, nbar_b) =', c)
        calcular_caso(*c)


if __name__ == '__main__':
    main()
