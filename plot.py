"""
Plota F1(sigma) a partir dos dados gerados por calcular.py em
data/counterprop.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros de entrada/saída (todos configuráveis)
# ----------------------------------------------------------------------
infile = os.path.join('data', 'counterprop', 'fidelity_counterprop_l_1_chi_03.npz')  # dados de entrada
outfile = 'fidelity_counterprop_l_1_chi_03.svg'  # arquivo de imagem de saída
dpi = 300            # resolução da imagem salva
show = True          # se True, abre a janela interativa além de salvar

# ----------------------------------------------------------------------
# Parâmetros visuais do gráfico
# ----------------------------------------------------------------------
figsize = (7, 5)     # tamanho da figura (largura, altura) em polegadas
color = '#8A2BE2'     # cor das curvas
ylim = (0.3, 1.02)   # limites do eixo y
grid = False         # se True, mostra grade no gráfico

title = None         # título do gráfico; se None, é gerado a partir dos parâmetros salvos no .npz
xlabel = r'Largura do pacote de onda $\sigma$'
ylabel = 'Fidelidade $F_1$'
label_pi = r'$F_1(\phi=\pi)$ — 2 sítios, contrapropagante'
label_opt = r'$F_1(\phi_{opt})$ — 2 sítios, contrapropagante'


def default_title(data):
    chi1 = float(data['chi1']) if 'chi1' in data else None
    gamma1 = float(data['gamma1']) if 'gamma1' in data else None
    omega0 = float(data['omega0']) if 'omega0' in data else None
    L = float(data['L']) if 'L' in data else None
    if None in (chi1, gamma1, omega0, L):
        return ''
    return rf'$\chi={chi1:.3g},\ \gamma={gamma1:.3g},\ \omega_0={omega0:.3g},\ L={L:.3g}$'


def main():
    data = np.load(infile)
    sigmas = data['sigmas']
    F1_pi_list = data['F1_pi']
    F1_opt_list = data['F1_opt']

    plot_title = title if title is not None else default_title(data)

    plt.figure(figsize=figsize)
    # plt.semilogx(sigmas, F1_pi_list, '-', color=color, label=label_pi)
    # plt.semilogx(sigmas, F1_opt_list, '--', color=color, label=label_opt)
    plt.semilogx(
    sigmas, 
    F1_pi_list, 
    'D-',
    color=color,
    linewidth=1,          # espessura da linha
    markersize=1,           # tamanho da bolinha
    markeredgewidth=2.5,    # espessura da borda da bolinha
    markeredgecolor=(0.5, 0, 0.5, 0.4),# cor da borda
    markerfacecolor=(0.5, 0, 0.5, 0.4),# cor da borda,  # preenchimento (pode ser 'none' para vazado)
    label=label_pi
)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot_title:
        plt.title(plot_title)
    plt.ylim(*ylim)
    plt.axhline(1.0, color='gray', lw=0.8, ls=':')
    plt.legend(loc='upper right')
    if grid:
        plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()


if __name__ == '__main__':
    main()
