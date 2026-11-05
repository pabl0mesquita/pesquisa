"""Nucleo numerico: qubit dual-rail de 1 foton, 1 sitio de Brod-Combes com um oscilador (banho)
termico ou em estado de Fock. Usado por main.py.

Trilho 1 passa pelo atomo (acoplado ao oscilador b); trilho 2 e livre.
Ideal: trilho 1 com a matriz-S de um foton SEM banho (atomo na linha deslocada
pelo polaron, calibravel), trilho 2 identidade.
"""
import sys
import time
import numpy as np
from scipy.linalg import expm
from multiprocessing import Pool

gam = 1.0


def disp_matrix(beta, N):
    b = np.diag(np.sqrt(np.arange(1, N)), 1)
    return expm(beta * (b.T - b))  # D[k, n] = <k|D(beta)|n>


def thermal_p(nbar, nmax):
    n = np.arange(nmax + 1)
    if nbar <= 0:
        p = np.zeros(nmax + 1); p[0] = 1.0
        return p
    return nbar ** n / (nbar + 1) ** (n + 1)


def t_mn(m, n, nu, Delta, wb, g, D, K):
    """Elemento da Eq. (21) do PDF de Fock, vetorizado em nu."""
    ks = np.arange(K)
    num = D[ks, m] * D[ks, n]
    den = gam / 2 + 1j * (Delta - 4 * g**2 / wb + (ks[:, None] - n) * wb - nu[None, :])
    return (1.0 if m == n else 0.0) - gam * (num[:, None] / den).sum(0)


def t0(nu, Delta_eff):
    """Sem banho, atomo em Delta_eff: Eq. (24) de Brod-Combes."""
    G = gam / 2 + 1j * (Delta_eff - nu)
    return -np.conj(G) / G


def xi(w, wc, sig):
    return (2 * np.pi * sig**2) ** -0.25 * np.exp(-(w - wc) ** 2 / (4 * sig**2))


def quantities(wc, sig, Delta, wb, g, nbar, polaron_ref=True, nmax=None, side=None, Ngrid=4001, Dref=None, fock=None):
    """Retorna eta (populacao no modo ideal), C (coerencia), Fbar, Fbar_fase, fase ideal,
    e o peso total de bandas laterais.
    a_mn = <u1|phi_mn>,  u1 = xi*t0,  phi_mn(w) = xi(w+(m-n)wb) t_mn(w+(m-n)wb)."""
    if fock is not None:          # banho em estado de Fock |fock>: p_n = delta_{n,fock}
        nbar = float(fock)
    if nmax is None:
        nmax = int(max(6, nbar * 12 + 6)) if fock is None else int(fock)
    if side is None:
        side = int(min(40, 8 + 6 * (2 * g / wb) ** 2 * (2 * nbar + 1) + 4))
    p = _pop(nbar, nmax, fock)
    beta = 2 * g / wb
    N = nmax + side + 60
    D = disp_matrix(beta, N)
    K = N - 20
    if Dref is None:
        Dref = Delta - 4 * g**2 / wb if polaron_ref else Delta
    W = 8 * sig + 2 * gam
    w = np.linspace(wc - W, wc + W, Ngrid); dw = w[1] - w[0]
    u1 = xi(w, wc, sig) * t0(w, Dref)
    eta = 0.0; C = 0.0 + 0j; side_w = 0.0; unit = 0.0
    for n in range(nmax + 1):
        if p[n] < 1e-12:
            continue
        for m in range(max(0, n - side), n + side + 1):
            sh = (m - n) * wb
            nu = w + sh
            phi = xi(nu, wc, sig) * t_mn(m, n, nu, Delta, wb, g, D, K)
            a = np.sum(np.conj(u1) * phi) * dw
            eta += p[n] * abs(a) ** 2
            wt = np.sum(abs(xi(w, wc, sig) * t_mn(m, n, w, Delta, wb, g, D, K)) ** 2) * dw
            unit += p[n] * wt
            if m == n:
                C += p[n] * a
            else:
                side_w += p[n] * wt
    Fbar = (1 + eta + C.real) / 3
    Fopt = (1 + eta + abs(C)) / 3
    phase_id = np.angle(np.sum(abs(xi(w, wc, sig)) ** 2 * t0(w, Dref)) * dw)
    return dict(eta=eta, C=C, Fbar=Fbar, Fopt=Fopt, phase=phase_id, side=side_w, unit=unit)


def nbar_of(wb, T):
    return 0.0 if T == 0 else 1.0 / np.expm1(wb / T)


def _pop(nbar, nmax, fock=None):
    if fock is not None:
        p = np.zeros(nmax + 1); p[int(fock)] = 1.0
        return p
    p = thermal_p(nbar, nmax)
    return p / p.sum()


def line_center(wb, g, nbar, Delta=0.0, fock=None):
    """Frequencia em que a resposta coerente media sum_n p_n t_nn(nu) tem fase pi
    (o foton ganha a fase de ressonancia). Escolhe a raiz de interacao mais forte."""
    nmax = int(max(6, nbar * 12 + 6)) if fock is None else int(fock)
    p = _pop(nbar, nmax, fock)
    N = nmax + 100; D = disp_matrix(2 * g / wb, N); K = N - 20
    nu = np.linspace(Delta - 4 * g**2 / wb - 3, Delta + 3, 6001)
    tc = sum(p[n] * t_mn(n, n, nu, Delta, wb, g, D, K) for n in range(nmax + 1) if p[n] > 1e-12)
    im = tc.imag; idx = np.where((np.sign(im[:-1]) != np.sign(im[1:])) & (tc.real[:-1] < 0))[0]
    if len(idx) == 0:
        return nu[np.argmin(tc.real)]
    best = idx[np.argmin(tc.real[idx])]
    x0, x1 = nu[best], nu[best + 1]; y0, y1 = im[best], im[best + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def resonant(sig, wb, g, nbar, Ngrid=801, fock=None):
    c = line_center(wb, g, nbar, fock=fock)
    r = quantities(c, sig, 0.0, wb, g, nbar, Ngrid=Ngrid, Dref=c, fock=fock); r['wc'] = c
    return r


# =====================================================================
#  Barra de progresso com estimativa de tempo
#  (usa tqdm se estiver instalado; senao, uma barra simples sem dependencias)
# =====================================================================

def _fmt_t(s):
    s = int(round(s))
    h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


class _Barra:
    def __init__(self, total, desc, largura=30):
        self.total, self.desc, self.largura = total, desc, largura
        self.n, self.t0 = 0, time.time()
        self._mostrar()

    def update(self, k=1):
        self.n += k
        self._mostrar()

    def _mostrar(self):
        frac = self.n / self.total if self.total else 1.0
        cheio = int(self.largura * frac)
        dt = time.time() - self.t0
        if self.n > 0:
            resta = dt * (self.total - self.n) / self.n
            eta = f"restam ~{_fmt_t(resta)}"
        else:
            eta = "estimando..."
        barra = "#" * cheio + "-" * (self.largura - cheio)
        sys.stdout.write(f"\r{self.desc}: [{barra}] {self.n}/{self.total} "
                         f"({100*frac:5.1f}%)  decorrido {_fmt_t(dt)}  {eta}   ")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write(f"\n{self.desc}: concluido em {_fmt_t(time.time() - self.t0)}\n")
        sys.stdout.flush()


# Barra global (soma de todas as figuras). main.py a cria em core._BARRA_TOTAL;
# cada ponto calculado em progresso() tambem a avanca, dando o tempo total restante.
_BARRA_TOTAL = None

_FMT = ("{desc}: {percentage:5.1f}%|{bar:30}| {n}/{total} "
        "[decorrido {elapsed}, restam ~{remaining}]")


def progresso(iteravel, total, desc):
    """Itera mostrando barra de progresso, % concluido, tempo decorrido e tempo restante."""
    try:
        from tqdm import tqdm
    except ImportError:
        b = _Barra(total, desc)
        for x in iteravel:
            yield x
            b.update()
        b.close()
        return
    pos = 0 if _BARRA_TOTAL is None else 1
    with tqdm(total=total, desc=desc, position=pos, leave=True, dynamic_ncols=True,
              bar_format=_FMT) as barra:
        for x in iteravel:
            yield x
            barra.update()
            if _BARRA_TOTAL is not None:
                _BARRA_TOTAL.update()


def _mapa_paralelo(func, jobs, procs, desc):
    """Pool.imap (mantem a ordem) com barra de progresso."""
    with Pool(procs) as P:
        return list(progresso(P.imap(func, jobs, chunksize=1), len(jobs), desc))
