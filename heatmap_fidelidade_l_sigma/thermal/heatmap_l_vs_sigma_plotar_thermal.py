"""
Plota o heatmap da fidelidade no plano (sigma, L) a partir dos dados gerados por
heatmap_l_vs_sigma_calcular_thermal.py em data/heatmap.

Versão com banho térmico de defasagem: os dados agora incluem gamma_phi, a
sobreposição de fóton único f1 e fidelidades que usam só a parte COERENTE do
estado de saída. Por isso F1_pi e F1_opt são cotas inferiores da fidelidade
média da porta. Arquivos antigos (sem banho) continuam funcionando: nesse
caso assume-se gamma_phi = 0 e f1 = 1.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros de entrada/saída
# ----------------------------------------------------------------------
# O nome do arquivo segue a convenção de heatmap_l_vs_sigma_calcular.py.
# Ajuste estes valores para os usados no cálculo (ou defina `infile` à mão).
chi, gamma, omega0, gamma_phi = 1000.0, 1.0, 0.0, 1
infile = os.path.join(
    'data', 'heatmap',
    f'heatmap_chi{chi:g}_gamma{gamma:g}_omega{omega0:g}_gphi{gamma_phi:g}_thermal.npz')
outfile = None        # arquivo de imagem; se None, é gerado a partir de `field` e gamma_phi
dpi = 300             # resolução da imagem salva
show = True           # se True, abre a janela interativa além de salvar

# ----------------------------------------------------------------------
# O que colorir
# ----------------------------------------------------------------------
# 'F1_pi'  : fidelidade média com phi = pi, a porta CZ (cota inferior)
# 'F1_opt' : fidelidade média com phi otimizado (cota inferior)
# Diagnósticos (não são fidelidade):
# 'F_abs'  : |<alvo|parte coerente de dois fótons>|, o módulo do produto escalar
# 'F_arg'  : fase de F (em unidades de pi)
# 'f1_abs' : |f1|, sobreposição de fóton único com o alvo (não depende de L)
field = 'F1_pi'
vmin, vmax = None, None  # faixa da barra de cores; se None, usa o padrão de `field`
cbar_label = None     # rótulo da barra; se None, usa o padrão de `field`

# ----------------------------------------------------------------------
# Parâmetros visuais
# ----------------------------------------------------------------------
figsize = (8, 6)      # tamanho da figura (largura, altura) em polegadas
cmap = 'viridis'      # mapa de cores ('twilight' é uma boa escolha para 'F_arg')
xlabel = r'$\sigma/\gamma$'
ylabel = r'$L$'
title = None          # título; se None, é gerado a partir dos parâmetros salvos

FIELD_LABELS = {
    'F_abs': r'$|\langle \mathrm{alvo}|\psi_{\mathrm{coh}}\rangle|$',
    'F_arg': r'$\arg F/\pi$',
    'f1_abs': r'$|f_1|$ (fóton único)',
    'F1_pi': r'Fidelidade $F_1(\phi=\pi)$ (cota inferior)',
    'F1_opt': r'Fidelidade $F_1(\phi_{\mathrm{opt}})$ (cota inferior)',
}
# Para a fidelidade a faixa é 0-1 (e não 0,4-1 como no caso sem banho), pois a
# defasagem leva F1 bem abaixo de 0,4; a mesma escala permite comparar gamma_phi.
FIELD_RANGES = {
    'F_abs': (0.0, 1.0),
    'F_arg': (-1.0, 1.0),
    'f1_abs': (0.0, 1.0),
    'F1_pi': (0.0, 1.0),
    'F1_opt': (0.0, 1.0),
}


def get_field(data, field):
    """Devolve o campo pedido como matriz (n_L, n_sigma), incluindo os
    campos derivados que não estão salvos diretamente no .npz."""
    n_L, n_sigma = len(data['Ls']), len(data['sigmas'])
    if field in data.files:
        return data[field]
    if field == 'F_arg':
        return np.angle(data['F_re'] + 1j * data['F_im']) / np.pi
    if field == 'f1_abs':
        if 'f1_re' in data.files:
            f1 = np.hypot(data['f1_re'], data['f1_im'])
        else:                       # arquivo antigo, sem banho
            f1 = np.ones(n_sigma)
        return np.broadcast_to(f1, (n_L, n_sigma))
    raise KeyError(f"campo '{field}' não encontrado em {infile}")


def default_title(data):
    def val(key):
        return float(data[key]) if key in data.files else None
    chi_, gamma_, omega0_ = val('chi'), val('gamma'), val('omega0')
    if None in (chi_, gamma_, omega0_):
        return ''
    gphi = val('gamma_phi') or 0.0
    s = rf'$\chi={chi_:g},\ \gamma={gamma_:g},\ \omega_0={omega0_:g},\ \gamma_\phi={gphi:g}\gamma$'
    if 'use_T' in data.files and bool(data['use_T']):
        s += rf'  ($\alpha={val("alpha"):g},\ k_BT/\hbar\gamma={val("T_red"):g}$)'
    return s


def main():
    data = np.load(infile)
    sigmas = data['sigmas']
    Ls = data['Ls']
    z = get_field(data, field)

    lo, hi = FIELD_RANGES.get(field, (None, None))
    lo = vmin if vmin is not None else lo
    hi = vmax if vmax is not None else hi

    plot_title = title if title is not None else default_title(data)
    label = cbar_label if cbar_label is not None else FIELD_LABELS.get(field, field)

    gphi = float(data['gamma_phi']) if 'gamma_phi' in data.files else 0.0
    out = outfile if outfile is not None else f'heatmap_{field}_sigma_L_gphi{gphi:g}.svg'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(sigmas, Ls, z, cmap=cmap,
                       vmin=lo, vmax=hi, shading='auto',
                       edgecolors='face', linewidth=0, antialiased=True,
                       rasterized=True)   # recomendado dado o tamanho da malha (200x200)
    ax.set_xscale('log')
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_title:
        ax.set_title(plot_title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    print(f"Figura salva em {out}")
    if show:
        plt.show()
    return fig, ax


if __name__ == '__main__':
    main()