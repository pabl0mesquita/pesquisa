"""Dois fotons contrapropagantes em N sitios de Brod-Combes, limite chi -> infinito,
com cada sitio acoplado a um oscilador no REGIME LENTO (dessintonia estatica).

Notacao de Brod-Combes (PRL 117, 080502, App. B):
    Gamma_j(w) = gamma/2 + i(Delta_j - w),   f_j(w) = -Gamma_j^*(w)/Gamma_j(w)   (1 foton, sitio j)
Matriz-S de 2 fotons contrapropagantes, sitios com parametros proprios (generaliza B.7 e B.9):
    S2 = S1a S1b - (1/pi) delta(wa+wb-na-nb) sum_j P_j K_j
    P_j = prod_{l>j} f_l(wa) f_l(nb) * prod_{l<j} f_l(na) f_l(wb)
    K_j = i chi gamma^2 [1 + 2 i chi/(Gamma_j(wa)+Gamma_j(wb))]^{-1} / [Gamma_j(nb)Gamma_j(na)Gamma_j(wb)Gamma_j(wa)]
        -> gamma^2 (Gamma_j(wa)+Gamma_j(wb)) / (2 Gamma_j(nb)Gamma_j(na)Gamma_j(wb)Gamma_j(wa))   (chi -> inf)

Regime lento: Delta_j = Delta + 2 g x_j, x_j ~ N(0, 2 nbar + 1) independentes; um oscilador por sitio,
que desloca as duas transicoes (a e b) do atomo em V do sitio.

Fidelidade media de porta (Brod-Combes Eq. 2, d=4) com a CZ (phi=pi), descontando a deformacao de
foton unico da cadeia SEM oscilador (t0^N, referencia centrada em Delta):
    mu = (1, mu_a, mu_b, lambda),  mu_k = <xi| t0^{*N} t_k |xi>,  lambda = e^{-i pi} <xi xi| (t0^*)^N (t0^*)^N S2 |xi xi>
    Fbar = E_x[ sum|mu_i|^2 + |sum mu_i|^2 ] / 20
"""
import numpy as np

gam = 1.0


def Gam(w, D):
    return gam / 2 + 1j * (D - w)


def f(w, D):
    G = Gam(w, D)
    return -np.conj(G) / G


def xi(w, wc, sig):
    return (2 * np.pi * sig ** 2) ** -0.25 * np.exp(-(w - wc) ** 2 / (4 * sig ** 2))


def mus(Ds_a, Ds_b, sig, Dref=0.0, wc=0.0, L=56, chi=np.inf):
    """Amplitudes logicas para UMA realizacao de dessintonias (listas por sitio)."""
    N = len(Ds_a)
    W = 6 * sig
    x = np.linspace(-W, W, L); dx = x[1] - x[0]
    w1 = wc + x
    p = xi(w1, wc, sig) ** 2 * dx
    t0N = f(w1, Dref) ** N
    ta = np.prod([f(w1, D) for D in Ds_a], axis=0)
    tb = np.prod([f(w1, D) for D in Ds_b], axis=0)
    mu_a = np.sum(p * np.conj(t0N) * ta)
    mu_b = np.sum(p * np.conj(t0N) * tb)
    # termo conectado: variaveis na, nb, wa ; wb = na + nb - wa
    na = w1[:, None, None]; nb = w1[None, :, None]; wa = w1[None, None, :]
    wb = na + nb - wa
    amp = xi(na, wc, sig) * xi(nb, wc, sig) * xi(wa, wc, sig) * xi(wb, wc, sig)
    ref = np.conj(f(wa, Dref) ** N * f(wb, Dref) ** N)
    soma = 0.0
    for j in range(N):
        Da, Db = Ds_a[j], Ds_b[j]
        Ga_na, Ga_wa = Gam(na, Da), Gam(wa, Da)
        Gb_nb, Gb_wb = Gam(nb, Db), Gam(wb, Db)
        if np.isinf(chi):
            K = gam ** 2 * (Ga_wa + Gb_wb) / (2 * Gb_nb * Ga_na * Gb_wb * Ga_wa)
        else:
            K = 1j * chi * gam ** 2 / (1 + 2j * chi / (Ga_wa + Gb_wb)) / (Gb_nb * Ga_na * Gb_wb * Ga_wa)
        P = 1.0
        for l in range(j + 1, N):          # sitios depois de j: a ja interagiu (wa), b ainda nao (nb)
            P = P * f(wa, Ds_a[l]) * f(nb, Ds_b[l])
        for l in range(j):                 # sitios antes de j: a ainda nao (na), b ja interagiu (wb)
            P = P * f(na, Ds_a[l]) * f(wb, Ds_b[l])
        soma = soma + P * K
    conectado = -(1 / np.pi) * np.sum(amp * ref * soma) * dx ** 3
    S11 = mu_a * mu_b + conectado
    lam = -S11                              # e^{-i pi}
    return np.array([1.0 + 0j, mu_a, mu_b, lam])


def F_de_mus(mu):
    return (np.sum(abs(mu) ** 2) + abs(np.sum(mu)) ** 2) / 20


def F_media(N, sig, kappa, M=200, seed=1, correlado=True, L=56, Delta=0.0):
    """Media sobre a desordem estatica: delta_j = (gamma/2) * u_j, u_j ~ N(0, kappa)."""
    if kappa == 0:
        D = [Delta] * N
        return F_de_mus(mus(D, D, sig, Dref=Delta, wc=Delta, L=L)), 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(M):
        ua = rng.normal(0, np.sqrt(kappa), N)
        ub = ua if correlado else rng.normal(0, np.sqrt(kappa), N)
        Da = Delta + (gam / 2) * ua; Db = Delta + (gam / 2) * ub
        vals.append(F_de_mus(mus(list(Da), list(Db), sig, Dref=Delta, wc=Delta, L=L)))
    vals = np.array(vals)
    return vals.mean(), vals.std() / np.sqrt(M)
