# -*- coding: utf-8 -*-
"""
Plota a varredura em S de otimizar_fock_N.py (otimizar_S = False): para cada
S de `S_valores` e cada N, delta e sigma foram otimizados independentemente
(modelo (b)), com omega, n e chi fixos.

Gera duas figuras:
  fig_otim_S_w*_n*.svg      (a) F* x S, (b) delta* x S, (c) sigma* x S, uma curva por N
  fig_otim_sigma_w*_n*.svg  F(sigma) nos (delta, S) de cada curva, um painel por N,
                            uma curva por S (escala sequencial: S maior = mais escuro)

Uso:  python3 plot_fock_N_S.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

import plot_fock_N as pf   # estilo (rcParams), paleta e nomes dos arquivos

# ======================================================================
# Parametros do grafico
# ======================================================================
w = 0.3                                  # frequencia dos osciladores (fixa na otimizacao)
n = 0                                    # estado de Fock inicial
S_valores = [0.0, 0.05, 0.1, 0.25, 0.5]  # mesmos de otimizar_fock_N.py
N_values = pf.N_values
cmap_S = plt.get_cmap('Blues')           # sequencial: S -> claro..escuro
faixa_cmap = (0.35, 1.0)                 # trecho do mapa usado (evita o quase branco)
figsize_S = (10, 3.2)
figsize_sigma = (10, 6)


def carregar():
    """dados[N][S] = conteudo do .npz (so os que existem)."""
    dados = {}
    for N in N_values:
        dados[N] = {}
        for S in S_valores:
            f = pf.arquivo_otim(N, w, S, n)
            if os.path.exists(f):
                dados[N][S] = dict(np.load(f))
            else:
                print(f'  faltando {os.path.basename(f)} (rode otimizar_fock_N.py)')
    return dados


def cor_S(S):
    x = (S - min(S_valores)) / (max(S_valores) - min(S_valores))
    return cmap_S(faixa_cmap[0] + x * (faixa_cmap[1] - faixa_cmap[0]))


def fonte_ticks(ax):
    """Numeros sem serifa, como em plot_fock_N.py."""
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily('sans-serif')
        lab.set_fontsize(8)


def letra(ax, txt):
    ax.text(0.03, 0.965, txt, transform=ax.transAxes, ha='left', va='top',
            fontdict=dict(pf._tick_font, size=9))


def salvar(fig, outfile):
    fig.savefig(outfile, bbox_inches='tight')
    fig.savefig(outfile.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    print(f'Figura: {outfile}')
    plt.close(fig)


def titulo():
    return (rf"$\chi={pf.otim_chi[3:]},\ \gamma={pf.otim_gam:g},\ \omega^{{(a,b)}}={w:g},\ n={n}$;"
            r"  $\delta,\ \sigma$ otimizados para cada $(N, S)$")


# ======================================================================
# Figura 1: parametros otimos em funcao de S
# ======================================================================
def fig_parametros(dados, outfile):
    fig, axs = plt.subplots(1, 3, figsize=figsize_S)
    quantidades = [('F_opt', r'Fidelidade m$\acute{\rm a}$xima ($F^*$)'),
                   ('delta_opt', r'Dessintonia $\acute{\rm o}$tima ($\delta^*$)'),
                   ('sigma_opt', r'Largura $\acute{\rm o}$tima ($\sigma^*$)')]
    for k, N in enumerate(N_values):
        Ss = [S for S in S_valores if S in dados[N]]
        if not Ss:
            continue
        for ax, (chave, _) in zip(axs, quantidades):
            ax.plot(Ss, [float(dados[N][S][chave]) for S in Ss], '-o', color=pf.MMA[k % len(pf.MMA)],
                    lw=1.1, ms=3.5, label=f'N = {N}')
    for ax, (_, rot), l in zip(axs, quantidades, '(a) (b) (c)'.split()):
        ax.set_xlabel(r'Fator de Huang-Rhys ($S$)', labelpad=2)
        ax.set_ylabel(rot)
        ax.xaxis.set_major_locator(FixedLocator(S_valores))
        ax.set_xticklabels([f'{S:g}' for S in S_valores])
        letra(ax, l)
    axs[1].axhline(0, color='#AAAAAA', lw=0.5, zorder=0)   # delta = 0: linha de zero fonon
    axs[0].legend(loc='lower left', fontsize=7, handlelength=1.6)
    fig.suptitle(titulo(), fontsize=9)
    fig.tight_layout()
    for ax in axs:
        fonte_ticks(ax)
    salvar(fig, outfile)


# ======================================================================
# Figura 2: F(sigma) para cada N, uma curva por S
# ======================================================================
def fig_sigma(dados, outfile):
    ncol = 3
    nlin = int(np.ceil(len(N_values) / ncol))
    fig, axs = plt.subplots(nlin, ncol, figsize=figsize_sigma, sharex=True, sharey=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, N in zip(axs, N_values):
        for S in S_valores:
            if S not in dados[N]:
                continue
            d = dados[N][S]
            ax.semilogx(d['sigmas'], d['F1'], '-', color=cor_S(S), lw=1.1, label=f'S = {S:g}')
        ax.set_xlim(*pf.xlim)
        ax.set_ylim(*pf.ylim)
        ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1]))
        ax.set_xticklabels(['0.01', '0.1', '1'])
        ax.xaxis.set_minor_formatter(NullFormatter())
        yt = np.round(np.arange(np.ceil(pf.ylim[0] * 10) / 10, pf.ylim[1] + 1e-9, 0.2), 1)
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.set_yticklabels([f'{v:.1f}' for v in yt])
        ax.text(0.97, 0.965, f'N = {N}', transform=ax.transAxes, ha='right', va='top',
                fontdict=dict(pf._tick_font, size=8),
                bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.6', fc='#E6E6E6', ec='none'))
    for ax in axs[len(N_values):]:   # paineis que sobram: legenda
        ax.axis('off')
    h, l = axs[0].get_legend_handles_labels()
    if len(axs) > len(N_values):
        axs[-1].legend(h, l, loc='center', fontsize=8, title=r'$\delta^*,\ \sigma^*$ de cada $(N, S)$',
                       title_fontsize=8)
    else:
        axs[0].legend(h, l, loc='lower left', fontsize=7)
    for ax in axs[(nlin - 1) * ncol:]:
        ax.set_xlabel(pf.xlabel, labelpad=2)
    for ax in axs[::ncol]:
        ax.set_ylabel(pf.ylabel)
    # o ultimo painel de cada coluna que tem dados recebe o rotulo do eixo x
    for c in range(ncol):
        col = [i for i in range(c, len(N_values), ncol)]
        if col and col[-1] < (nlin - 1) * ncol:
            axs[col[-1]].set_xlabel(pf.xlabel, labelpad=2)
            axs[col[-1]].xaxis.set_tick_params(labelbottom=True)
    fig.suptitle(titulo(), fontsize=9)
    fig.tight_layout()
    for ax in axs[:len(N_values)]:
        fonte_ticks(ax)
    salvar(fig, outfile)


def main():
    dados = carregar()
    if not any(dados[N] for N in N_values):
        print('nenhum dado: figuras nao geradas')
        return
    fig_parametros(dados, os.path.join(pf.here, f'fig_otim_S_w{w:g}_n{n}.svg'))
    fig_sigma(dados, os.path.join(pf.here, f'fig_otim_sigma_w{w:g}_n{n}.svg'))


if __name__ == '__main__':
    main()
