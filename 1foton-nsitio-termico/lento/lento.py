"""Regime lento (omega_b << gamma): limite quase-estatico do oscilador.

Com omega_b -> 0, x = b + b^dagger comuta com H durante o espalhamento. O acoplamento
g(1 - A_z)(b + b^dagger) = 2g |e><e| x vira uma DESSINTONIA ESTATICA 2 g x, com x
distribuido como no estado do oscilador. Para o termico, x e gaussiano com
<x^2> = 2 nbar + 1 (o "+1" e o ponto zero).

Por sitio e por valor de x:   t_x(nu) = -Gamma_x^*/Gamma_x,  Gamma_x = gamma/2 + i(Delta + 2 g x - nu).
|t_x| = 1: o foton so ganha uma fase aleatoria (defasagem pura, sem vazamento).

N sitios independentes (x_j independentes), referencia ideal = atomo sem banho na linha nua Delta:
    C   = int dw |xi|^2 [t0^* tbar]^N,            tbar(w) = E_x[t_x(w)]
    eta = int int |xi(w)|^2 |xi(w')|^2 [t0^*(w) t0(w') K(w,w')]^N,   K = E_x[t_x(w) t_x^*(w')]
    Fbar = (1 + eta + Re C)/3

Pacote estreito, foton na linha:  eta = 1,  C = C1(kappa)^N,
    kappa = (4 g/gamma)^2 (2 nbar + 1),
    C1 = 2 sqrt(pi/(2 kappa)) exp(1/(2 kappa)) erfc(1/sqrt(2 kappa)) - 1   (exato no limite estatico)
       ~ 1 - 2 kappa
    Fbar = (2 + C1^N)/3,   1 - Fbar ~ (2/3) N kappa.
"""
import numpy as np
from scipy.special import erfcx

gam = 1.0


def nbar_of(w, T):
    return 0.0 if T == 0 else 1.0 / np.expm1(w / T)


def kappa(g, nbar):
    return (4 * g / gam) ** 2 * (2 * nbar + 1)


def C1_fechado(k):
    """<(1-u^2)/(1+u^2)>, u gaussiano de variancia k. Usa erfcx para estabilidade."""
    k = np.asarray(k, float)
    z = 1.0 / np.sqrt(2 * k)
    return 2 * np.sqrt(np.pi / (2 * k)) * erfcx(z) - 1.0


def F_fechado(N, k):
    return (2 + C1_fechado(k) ** N) / 3


def xi(w, wc, sig):
    return (2 * np.pi * sig ** 2) ** -0.25 * np.exp(-(w - wc) ** 2 / (4 * sig ** 2))


def t_atom(nu, D):
    G = gam / 2 + 1j * (D - nu)
    return -np.conj(G) / G


def _grade_x(nbar, M=241):
    s = np.sqrt(2 * nbar + 1)
    x = np.linspace(-7 * s, 7 * s, M); dx = x[1] - x[0]
    P = np.exp(-x ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2) * dx
    return x, P / P.sum()


def estatico(wc, sig, N, g, nbar, Dref=0.0, Delta=0.0, L=161, M=241):
    """Modelo quase-estatico, com pacote de largura finita."""
    W = 6 * sig
    w = wc + np.linspace(-W, W, L); dw = w[1] - w[0]
    p = abs(xi(w, wc, sig)) ** 2 * dw; p = p / p.sum()
    x, P = _grade_x(nbar, M)
    Tx = t_atom(w[None, :], Delta + 2 * g * x[:, None])          # (M, L)
    tbar = P @ Tx
    t0 = t_atom(w, Dref)
    C = np.sum(p * (np.conj(t0) * tbar) ** N)
    K = (Tx * P[:, None]).T @ Tx.conj()                            # K[w,w'] = E[t(w) t*(w')]
    A = np.outer(np.conj(t0), t0) * K
    eta = np.real(p @ (A ** N) @ p)
    return dict(eta=eta, C=C, Fbar=(1 + eta + C.real) / 3, Fopt=(1 + eta + abs(C)) / 3,
                phase=np.angle(np.sum(p * t0 ** N)))
