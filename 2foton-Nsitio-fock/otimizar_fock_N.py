# -*- coding: utf-8 -*-
"""
Busca, para cada N, os parametros que maximizam a fidelidade da porta CPHASE
com N sitios e osciladores em estado de Fock (calcular_fock_N.py).

Modelo (b): para CADA N sao otimizados independentemente
    delta = omega0 + S*omega   dessintonia do pacote em relacao a linha de zero
                               fonon (o nivel k = n esta deslocado de -S*omega,
                               ver Gam em calcular_fock_N.py)
    S                          fator de Huang-Rhys (opcional, ver `otimizar_S`; o
                               otimo e sempre S = 0, por isso o padrao e repetir a
                               otimizacao para cada S de `S_valores`, ver plot_fock_N_S.py)
    chi                        acoplamento cross-Kerr (opcional, ver `otimizar_chi`)
    sigma                      largura do pacote
com a frequencia omega e o estado inicial n dos osciladores fixos.
Tudo em unidades de gamma (gam = 1; Delta = 0, pois so omega_c = Delta + omega0 entra).

Etapas, para cada N:
  1) triagem: grade nos parametros livres (delta, S, log10 chi); em cada ponto,
     F* = max_sigma F (varredura grossa em log sigma + Brent), com max_excitados
     reduzido (`max_exc_triagem`).  Pontos da grade em paralelo.
  2) refinamento: Nelder-Mead nos parametros livres + log10 sigma, partindo dos
     `n_candidatos` maiores maximos locais da grade, com o truncamento completo
     (`max_excitados`).
  3) curva F(sigma) nos parametros otimos, salva em data/otim_*.npz junto com
     o mapa da triagem; plot_fock_N.py le esses arquivos (`casos_otim`).

Uso:  python3 otimizar_fock_N.py
"""
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize, minimize_scalar

import calcular_fock_N as cf

# ======================================================================
# Sistema (fixo)
# ======================================================================
w = 0.3              # frequencia dos osciladores (a e b iguais)
n0 = 0               # estado de Fock inicial dos osciladores
gam = 1.0            # unidade de frequencia
N_values = [1, 2, 3, 4, 5]

# ======================================================================
# Parametros otimizados e limites
# ======================================================================
delta_lim = (-1.0, 2.5)      # faixa de delta/gam
n_delta = 25                 # pontos da grade de triagem em delta

otimizar_S = False           # False: varre S nos valores de S_valores (uma otimizacao por S)
S_valores = [0.0, 0.05, 0.1, 0.25, 0.5]
S_lim = (0.0, 1.0)           # faixa de S se otimizar_S (o otimo cai em S = 0: Brod-Combes)
n_S = 6                      # pontos da grade de triagem em S

otimizar_chi = False         # False: chi fica em chi_fixo
chi_fixo = 1000              # (np.inf: limite saturado)
log10_chi_lim = (-0.5, 3.0)  # faixa de log10(chi/gam)
n_chi = 8                    # pontos da grade de triagem em log10 chi

fator_sigma = (10.0, 6.0)    # sigma buscado em [sigma_chute/f0, sigma_chute*f1]
n_sigma_grosso = 9           # varredura grossa em sigma antes do Brent
log10_sigma_lim = (-3.0, 1.5)   # limites de log10 sigma no Nelder-Mead
n_candidatos = 3             # maximos locais da grade que sao refinados (a paisagem tem mais de um pico)

# ======================================================================
# Numerica
# ======================================================================
max_exc_triagem = 1          # truncamento na triagem (limite inferior, mais barato)
max_excitados = cf.max_excitados   # truncamento no refinamento e na curva final
n_sigma_curva = 61           # pontos da curva final F(sigma)
sigma_curva_lim = (0.01, 1.0)   # faixa da curva final (estendida se sigma* estiver perto da borda)
n_proc = max(1, (os.cpu_count() or 1) - 1)
quiet = False

here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, 'data')


# ======================================================================
# Parametros livres
# ======================================================================
def livres():
    """Lista (nome, faixa, n_grade, passo do simplex) dos parametros otimizados,
    exceto sigma.  chi entra como log10 chi."""
    p = [('delta', delta_lim, n_delta, (delta_lim[1] - delta_lim[0]) / max(n_delta - 1, 1))]
    if otimizar_S:
        p.append(('S', S_lim, n_S, (S_lim[1] - S_lim[0]) / max(n_S - 1, 1) / 2))
    if otimizar_chi:
        p.append(('log10_chi', log10_chi_lim, n_chi, 0.3))
    return p


def parametros(x, fixos):
    """dict(delta, S, chi) a partir do vetor x dos parametros livres (na ordem de
    livres()) e dos valores fixos `fixos` = dict(S, chi)."""
    p = dict(fixos)
    for (nome, *_), v in zip(livres(), x):
        if nome == 'log10_chi':
            p['chi'] = 10**v
        else:
            p[nome] = v
    return p


# ======================================================================
# Avaliacao da fidelidade
# ======================================================================
def configurar(p):
    """Ajusta as variaveis globais de calcular_fock_N (cada processo chama isto)."""
    cf.gam, cf.Delta, cf.chi = gam, 0.0, p['chi']
    cf.omega0 = p['delta'] - p['S'] * w
    cf.omega_c = cf.Delta + cf.omega0


def F_ponto(N, p, sigma, exc):
    configurar(p)
    cad = cf.montar_cadeia(N, w, p['S'], n0, w, p['S'], n0)
    return cf.fidelidade(cad, sigma, max_exc=exc)


def sigma_chute(N):
    """sigma* ~ 0.37 gamma / N^0.85 (ajuste aos dados com S = 0)."""
    return 0.37 * gam / N**0.85


def F_max_sigma(N, p, exc):
    """max_sigma F: varredura grossa em log sigma e Brent em volta do melhor ponto."""
    c = np.log10(sigma_chute(N))
    ls = np.linspace(c - np.log10(fator_sigma[0]), c + np.log10(fator_sigma[1]), n_sigma_grosso)
    Fs = [F_ponto(N, p, 10**x, exc) for x in ls]
    i = int(np.argmax(Fs))
    a, b = ls[max(i - 1, 0)], ls[min(i + 1, len(ls) - 1)]
    r = minimize_scalar(lambda x: -F_ponto(N, p, 10**x, exc),
                        bounds=(a, b), method='bounded', options=dict(xatol=0.005))
    if -r.fun >= Fs[i]:
        return -r.fun, 10**r.x
    return Fs[i], 10**ls[i]


def maximos_locais(F):
    """Indices dos maximos locais de F (vizinhanca 3^d), do maior para o menor."""
    eh_max = F >= maximum_filter(F, size=3, mode='constant', cval=-np.inf)
    idx = np.argwhere(eh_max)
    return [tuple(i) for i in idx[np.argsort(-F[eh_max])]]


def _triagem(args):
    N, x, fixos = args
    return F_max_sigma(N, parametros(x, fixos), max_exc_triagem)


def _curva(args):
    N, p, sigma = args
    return F_ponto(N, p, sigma, max_excitados)


def fmt(p, sigma=None):
    s = f"delta={p['delta']:+.4f}"
    if otimizar_S:
        s += f", S={p['S']:.4f}"
    if otimizar_chi:
        s += f", chi={p['chi']:.4g}"
    return s + (f', sigma={sigma:.4f}' if sigma is not None else '')


# ======================================================================
# Otimizacao para um N
# ======================================================================
def otimizar_N(N, fixos, ex):
    """Otimiza para N sitios; `fixos` = dict(S, chi) com os valores dos parametros nao otimizados."""
    t0 = time.time()
    lv = livres()
    eixos = [np.linspace(*lim, ng) for _, lim, ng, _ in lv]
    forma = tuple(len(e) for e in eixos)

    # 1) triagem
    pts = list(itertools.product(*eixos))
    res = list(ex.map(_triagem, [(N, x, fixos) for x in pts]))
    Fgrid = np.array([r[0] for r in res]).reshape(forma)
    Sgrid = np.array([r[1] for r in res]).reshape(forma)
    cands = maximos_locais(Fgrid)[:n_candidatos]
    if not quiet:
        for c in cands:
            x = [e[i] for e, i in zip(eixos, c)]
            print(f'  N={N}: triagem F*={Fgrid[c]:.5f} em {fmt(parametros(x, fixos), Sgrid[c])}')
        print(f'        [{time.time() - t0:.1f} s]', flush=True)

    # 2) refinamento (Nelder-Mead com o truncamento completo) a partir de cada candidato
    lim = [l for _, l, _, _ in lv] + [log10_sigma_lim]
    passos = np.array([ps for *_, ps in lv] + [0.1])
    obj = lambda x: -F_ponto(N, parametros(x[:-1], fixos), 10**x[-1], max_excitados)
    melhor = None
    for c in cands:
        x0 = np.array([e[i] for e, i in zip(eixos, c)] + [np.log10(Sgrid[c])])
        # simplex para dentro da faixa (o candidato pode estar na borda)
        sinal = np.array([1.0 if x0[k] + passos[k] <= lim[k][1] else -1.0 for k in range(len(x0))])
        simplex = np.vstack([x0] + [x0 + sinal[k] * np.eye(len(x0))[k] * passos[k]
                                    for k in range(len(x0))])
        r = minimize(obj, x0, method='Nelder-Mead', bounds=lim,
                     options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-6, maxiter=600))
        if not quiet:
            print(f'        refinado F*={-r.fun:.5f} em {fmt(parametros(r.x[:-1], fixos), 10**r.x[-1])}'
                  f'  ({r.nfev} avaliacoes)', flush=True)
        if melhor is None or r.fun < melhor.fun:
            melhor = r
    p_opt = parametros(melhor.x[:-1], fixos)
    sigma_opt = 10**melhor.x[-1]
    F_opt = -melhor.fun
    if F_opt > 1 + 1e-6:
        print(f'AVISO: F={F_opt:.5f} > 1 (aumente cf.pts_per_width/cf.k_sigma)')
    nomes = [n for n, *_ in lv] + ['log10_sigma']
    for x, (a, b), nome in zip(melhor.x, lim, nomes):
        if min(x - a, b - x) < 1e-3 * (b - a):
            print(f'AVISO: {nome} otimo ({x:.4g}) esta na borda da faixa {a, b}')
    if not quiet:
        print(f'        [{time.time() - t0:.1f} s]', flush=True)

    # 3) curva F(sigma) nos parametros otimos
    s0 = min(sigma_curva_lim[0], sigma_opt / 5)
    s1 = max(sigma_curva_lim[1], sigma_opt * 5)
    sigmas = np.logspace(np.log10(s0), np.log10(s1), n_sigma_curva)
    F1 = np.array(list(ex.map(_curva, [(N, p_opt, s) for s in sigmas])))

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, nome_arquivo(N, fixos['S']))
    np.savez(path, sigmas=sigmas, F1=F1, N=N,
             F_opt=F_opt, sigma_opt=sigma_opt, delta_opt=p_opt['delta'],
             S_opt=p_opt['S'], chi_opt=p_opt['chi'], omega0_opt=p_opt['delta'] - p_opt['S'] * w,
             nomes_triagem=np.array([n for n, *_ in lv]),
             **{f'grade_{n}': e for (n, *_), e in zip(lv, eixos)},
             F_triagem=Fgrid, sigma_triagem=Sgrid,
             w=w, n0=n0, gam=gam, phi=cf.phi,
             max_excitados=max_excitados, max_exc_triagem=max_exc_triagem,
             dq_max=cf.dq_max, referencia=cf.referencia,
             otimizar_S=otimizar_S, otimizar_chi=otimizar_chi)
    return dict(N=N, F=F_opt, sigma=sigma_opt, path=path, **p_opt)


def nome_arquivo(N, S):
    chi_tag = 'chiotim' if otimizar_chi else f'chi{chi_fixo:g}'
    S_tag = 'Sotim' if otimizar_S else f'S{S:g}'
    return (f"otim_N{N}_{chi_tag}_gam{gam:g}_w{w:g}_{S_tag}_n{n0}"
            f"_exc{max_excitados}_{cf.referencia}.npz")


def main():
    print(f'oscilador: omega={w:g}, n={n0}   (gam={gam:g}, {n_proc} processos)')
    print('otimizando', ', '.join(n for n, *_ in livres()), 'e sigma',
          f'(chi={chi_fixo:g})' if not otimizar_chi else '')
    lista_S = [None] if otimizar_S else S_valores
    resultados = []
    with ProcessPoolExecutor(n_proc) as ex:
        for S in lista_S:
            if S is not None:
                print(f'--- S = {S:g} ---', flush=True)
            for N in N_values:
                resultados.append(otimizar_N(N, dict(S=S, chi=chi_fixo), ex))
    print('\n   N     F*      delta     omega0      S        chi      sigma*')
    for r in resultados:
        print(f'  {r["N"]:2d}  {r["F"]:.5f}  {r["delta"]:+.4f}  {r["delta"] - r["S"] * w:+.4f}  '
              f'{r["S"]:.4f}  {r["chi"]:9.4g}  {r["sigma"]:.4f}')
    print('\nArquivos em', outdir)


if __name__ == '__main__':
    main()
