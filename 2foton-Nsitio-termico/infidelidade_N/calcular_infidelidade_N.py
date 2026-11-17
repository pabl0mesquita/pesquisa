"""Infidelidade da CZ de Brod-Combes (fotons contrapropagantes, chi -> inf, L = 0) em funcao
do numero de sitios N, com cada sitio acoplado a um oscilador no REGIME LENTO.
Salva ./data/infidelidade_N.npz (diretorio corrente) para plotar_infidelidade_N.py.

Pergunta: como kappa deve cair com N para que 1 - F_max continue caindo com N?

Modelo (o mesmo de ../lento/dois_fotons.py): dessintonia estatica por sitio
    Delta_j = (gamma/2) u_j,   u_j ~ N(0, kappa) independentes,   kappa = (4g/gamma)^2 (2 nbar + 1),
igual nas duas transicoes do atomo do sitio j. F e a fidelidade media de porta (d=4)
descontando a deformacao de foton unico da cadeia SEM oscilador, media sobre a desordem
(quase Monte Carlo, Sobol embaralhado, M amostras; as mesmas amostras para todo sigma e
kappa de um dado N, de modo que F_med(sigma) e suave e pode ser otimizada).

Calculo rapido do termo conectado (fatoracao em convolucoes, como em
infidelidade/calcular_infidelidade_LN.py, agora com sitios diferentes):
    P_j K_j = [out1_j(wa) out2_j(wb)] [in1_j(na) in2_j(nb)] (gamma + i(2 D_j - E)) gamma^2/2
    out1_j = xi conj(t0^N) prod_{l>j} f_l / Gam_j,   out2_j = xi conj(t0^N) prod_{l<j} f_l / Gam_j
    in1_j  = xi prod_{l<j} f_l / Gam_j,              in2_j  = xi prod_{l>j} f_l / Gam_j
    conectado = -(1/pi) sum_j int dE K_j(E) (out1_j * out2_j)(E) (in1_j * in2_j)(E)
Custo O(M N n log n) por sigma, em vez de O(M N^2 L^3).

Etapas:
    [1/4] validacoes (forca bruta de ../lento/dois_fotons.py e o caso ideal de BC)
    [2/4] grade (N, kappa): sigma_max e F_max                 -> plots I1, I4, I6
    [3/4] familia kappa = kappa0 N^-alpha: sigma_max e F_max  -> plots I2, I3
    [4/4] c(N) = -dF/dkappa|_0 no sigma otimo                 -> plots I2-I5

Uso:
    python calcular_infidelidade_N.py --procs 8
    python calcular_infidelidade_N.py --sem-validar
As etapas 2 e 3 sao salvas em data/parcial_*.npz ao terminar; se o calculo for
interrompido, rodar de novo reaproveita as etapas ja concluidas.
"""
import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.fft import fft, ifft, next_fast_len
from scipy.optimize import minimize_scalar
from scipy.stats import qmc, norm

AQUI = os.path.dirname(os.path.abspath(__file__))
PASTA = 'data'                       # relativa ao diretorio corrente
ARQUIVO = 'infidelidade_N.npz'

# =====================================================================
#  Parametros fisicos e das varreduras
# =====================================================================
gam = 1.0
NS = np.array(list(range(1, 21)) + [25, 30])          # numeros de sitios
KAPS = np.concatenate([[0.0], np.logspace(-6, -1, 21)])   # grade de kappa (inclui as decadas)
KAPPA0 = 1e-2                                         # familia kappa = KAPPA0 * N^-alpha
ALPHAS = np.array([0.0, 1.0, 1.5, 2.0, 2.6, 3.0])
M = 256                                               # amostras de Sobol por media (potencia de 2)
SEED = 7
H_DERIV = 0.05                                        # passo em u_j para d2F/du_j^2

# =====================================================================
#  Parametros numericos
# =====================================================================
K_SIG = 9.0          # meia-janela de frequencias em unidades de sigma
PTS = 12             # pontos por menor escala, min(sigma, gamma/(2N))
N_MIN = 401          # minimo de pontos na grade 1D
N_MAX = 20001        # teto
N_GROSSA = 13        # pontos da grade grossa em sigma antes do refino (Brent)
XATOL = 1e-4         # tolerancia do refino em log10(sigma)
LOTE = 16            # realizacoes por vez em F_lote (memoria por processo ~ LOTE*N*n*16 B * ~10 arrays)


# =====================================================================
#  Fisica
# =====================================================================
def Gam(w, D):
    return gam / 2 + 1j * (D - w)


def f(w, D):
    G = Gam(w, D)
    return -np.conj(G) / G


def xi(w, sig):
    return (2 * np.pi * sig ** 2) ** -0.25 * np.exp(-w ** 2 / (4 * sig ** 2))


def grade(sig, N):
    half = K_SIG * sig
    esc = min(sig, gam / (2 * N))
    n = int(2 * half / (esc / PTS)) + 1
    n = min(max(n, N_MIN), N_MAX) | 1
    w = np.linspace(-half, half, n)
    return w, w[1] - w[0]


def F_lote(D, sig):
    """F para cada realizacao. D: (R, N) dessintonias por sitio (iguais nas transicoes a e b).
    Processa LOTE realizacoes por vez para limitar a memoria (~LOTE*N*n complexos por array)."""
    D = np.atleast_2d(np.asarray(D, float))
    return np.concatenate([_F_lote(D[i:i + LOTE], sig) for i in range(0, len(D), LOTE)])


def _F_lote(D, sig):
    R, N = D.shape
    w, h = grade(sig, N)
    n = len(w)
    Nf = next_fast_len(2 * n - 1)
    x = xi(w, sig)
    f0c = np.conj(f(w, 0.0)) ** N                       # referencia: cadeia sem oscilador
    Dw = D[:, :, None]
    fl = f(w[None, None, :], Dw)                        # (R, N, n)
    G = Gam(w[None, None, :], Dw)
    cum = np.cumprod(fl, axis=1)
    um = np.ones((R, 1, n), complex)
    pre = np.concatenate([um, cum[:, :-1]], axis=1)     # prod_{l<j} f_l
    suf = np.concatenate([np.cumprod(fl[:, ::-1], axis=1)[:, ::-1][:, 1:], um], axis=1)   # prod_{l>j}
    mu = np.sum(x ** 2 * f0c * cum[:, -1], axis=1) * h  # mu_a = mu_b
    so = x * f0c / G
    si = x / G
    Co = ifft(fft(so * suf, Nf, axis=2) * fft(so * pre, Nf, axis=2), axis=2)[:, :, :2 * n - 1] * h
    Ci = ifft(fft(si * pre, Nf, axis=2) * fft(si * suf, Nf, axis=2), axis=2)[:, :, :2 * n - 1] * h
    E = 2 * w[0] + h * np.arange(2 * n - 1)
    K = gam ** 2 / 2 * (gam + 1j * (2 * Dw - E[None, None, :]))
    con = -(1 / np.pi) * np.sum(K * Co * Ci, axis=(1, 2)) * h
    lam = -(mu * mu + con)                              # e^{-i pi} S11
    soma2 = 1 + 2 * abs(mu) ** 2 + abs(lam) ** 2
    return (soma2 + abs(1 + 2 * mu + lam) ** 2) / 20


def amostras(N):
    """(M, N) gaussianas padrao (Sobol embaralhado), as mesmas para todo sigma e kappa de um N."""
    s = qmc.Sobol(d=N, scramble=True, seed=SEED).random(M)
    return norm.ppf(s)


def F_media(N, sig, kap, Z=None):
    if kap == 0:
        return float(F_lote(np.zeros((1, N)), sig)[0])
    Z = amostras(N) if Z is None else Z
    return float(np.mean(F_lote((gam / 2) * np.sqrt(kap) * Z, sig)))


def sigma_chute(N):
    """Otimo ideal de Brod-Combes, sigma_max ~ 0.35 N^-0.81."""
    return 0.35 * N ** -0.81


def otimizar(N, kap):
    """(sigma_max, F_max, borda): grade log grossa + refino de Brent em log10(sigma)."""
    Z = None if kap == 0 else amostras(N)
    s0 = sigma_chute(N)
    sig = np.logspace(np.log10(s0 / 5), np.log10(s0 * 4), N_GROSSA)
    F = np.array([F_media(N, s, kap, Z) for s in sig])
    k = int(np.argmax(F))
    borda = k in (0, len(sig) - 1)
    lo, hi = np.log10(sig[max(k - 1, 0)]), np.log10(sig[min(k + 1, len(sig) - 1)])
    r = minimize_scalar(lambda y: -F_media(N, 10 ** y, kap, Z), bounds=(lo, hi),
                        method='bounded', options={'xatol': XATOL})
    return 10 ** r.x, -r.fun, borda


def dFdk(N, sig, h=H_DERIV):
    """dF/dkappa em kappa=0 = (1/2) sum_j d2F/du_j^2 (diferencas finitas centradas)."""
    D = np.zeros((2 * N + 1, N))
    for j in range(N):
        D[1 + 2 * j, j] = +(gam / 2) * h
        D[2 + 2 * j, j] = -(gam / 2) * h
    F = F_lote(D, sig)
    return 0.5 * (np.sum(F[1:]) - 2 * N * F[0]) / h ** 2


# =====================================================================
#  Progresso e tempo
# =====================================================================
T_INICIO = time.time()


def _hms(t):
    t = int(round(max(t, 0)))
    return f"{t // 3600:d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def _custo(job):
    N, kap = job[1], job[2]
    return N * (1 if kap == 0 else M)


def _rodar(job):
    i, N, kap = job
    return i, otimizar(N, kap)


def _barra(pool, fn, jobs, pesos, desc, larg=30):
    """Roda os jobs (fora de ordem) com barra de progresso ponderada pelo custo,
    tempo decorrido, restante estimado e tempo total desde o inicio."""
    out = [None] * len(jobs); t0 = time.time()
    total = float(sum(pesos)); feito = 0.0
    for c, (i, r) in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
        out[i] = r
        feito += pesos[i]; fr = feito / total
        el = time.time() - t0; rest = el * (1 - fr) / fr
        cheio = int(larg * fr)
        print(f"\r{desc} [{'#' * cheio}{'-' * (larg - cheio)}] {100 * fr:5.1f}%  {c}/{len(jobs)}  "
              f"decorrido {_hms(el)}  faltam ~{_hms(rest)}  (total {_hms(time.time() - T_INICIO)})",
              end='', flush=True)
    print(f"\n{desc}: concluido em {_hms(time.time() - t0)}")
    return out


def _otimizar_lista(pool, pares, desc):
    jobs = [(i, int(N), float(k)) for i, (N, k) in enumerate(pares)]
    res = _barra(pool, _rodar, jobs, [_custo(j) for j in jobs], desc)
    s = np.array([r[0] for r in res]); F = np.array([r[1] for r in res])
    nb = sum(r[2] for r in res)
    if nb:
        print(f"  AVISO: {nb} otimizacoes terminaram na borda da grade em sigma")
    return s, F


# =====================================================================
#  Validacoes
# =====================================================================
def validar():
    print("[1/4] validacoes")
    t0 = time.time()
    sys.path.insert(0, os.path.join(AQUI, '..', 'lento'))
    from dois_fotons import mus, F_de_mus
    rng = np.random.default_rng(3)
    ok = True
    for N, sig in [(1, 0.3), (2, 0.2), (3, 0.15)]:
        D = list((gam / 2) * rng.normal(0, 0.3, N))
        Fb = F_de_mus(mus(D, D, sig, L=90))
        Fr = F_lote(np.array([D]), sig)[0]
        bom = abs(Fb - Fr) < 2e-3
        ok &= bom
        print(f"  [{'OK ' if bom else 'FALHOU'}] FFT x forca bruta (N={N}, sigma={sig}): "
              f"{Fr:.6f} x {Fb:.6f}  (|dF|={abs(Fb - Fr):.1e}, forca bruta com L=90)")
    for N in (5, 20):
        s, F, _ = otimizar(N, 0.0)
        print(f"  [info] ideal N={N}: F_max={F:.5f}, sigma_max={s:.4f}  "
              f"(BC: 1-F ~ 0.537 N^-1.61 = {0.537 * N ** -1.61:.4f}, obtido {1 - F:.4f})")
    N, s = 5, sigma_chute(5)
    F1 = F_media(N, s, 1e-3)
    pts0, nmin0 = globals()['PTS'], globals()['N_MIN']
    globals()['PTS'], globals()['N_MIN'] = 2 * pts0, 2 * nmin0 + 1
    F2 = F_media(N, s, 1e-3)
    globals()['PTS'], globals()['N_MIN'] = pts0, nmin0
    bom = abs(F1 - F2) < 1e-6
    ok &= bom
    print(f"  [{'OK ' if bom else 'FALHOU'}] convergencia da grade (N=5, kappa=1e-3, resolucao x2): "
          f"|dF|={abs(F1 - F2):.1e}")
    print(f"  {'todas as validacoes passaram' if ok else 'HA VALIDACOES FALHANDO'} "
          f"({_hms(time.time() - t0)})")
    return ok


# =====================================================================
#  Calculo
# =====================================================================
def _etapa(pool, nome, pares, desc):
    """Otimiza a lista de (N, kappa), salvando em data/ ao terminar. Se ja houver um arquivo
    da etapa com os mesmos (N, kappa) e M, reaproveita (retomada apos interrupcao)."""
    caminho = os.path.join(PASTA, f'parcial_{nome}.npz')
    chave = np.array(pares, float)
    if os.path.exists(caminho):
        r = np.load(caminho)
        if r['pares'].shape == chave.shape and np.allclose(r['pares'], chave) and int(r['M']) == M:
            print(f"{desc}: reaproveitado de {caminho}")
            return r['s'], r['F']
    s, F = _otimizar_lista(pool, pares, desc)
    os.makedirs(PASTA, exist_ok=True)
    np.savez(caminho, pares=chave, M=M, s=s, F=F)
    return s, F


def calcular(procs):
    with Pool(procs) as p:
        pares = [(N, k) for N in NS for k in KAPS]
        s, F = _etapa(p, 'grade', pares, '[2/4] grade (N, kappa)')
        sig_grade = s.reshape(len(NS), len(KAPS)); F_grade = F.reshape(len(NS), len(KAPS))

        pares = [(N, KAPPA0 * N ** -a) for a in ALPHAS for N in NS]
        s, F = _etapa(p, 'alpha', pares, '[3/4] familia kappa0 N^-alpha')
        sig_alpha = s.reshape(len(ALPHAS), len(NS)); F_alpha = F.reshape(len(ALPHAS), len(NS))

    t0 = time.time()
    i0 = int(np.where(KAPS == 0)[0][0])
    c = []
    for i, N in enumerate(NS):
        c.append(-dFdk(int(N), sig_grade[i, i0]))
        fr = (i + 1) / len(NS); el = time.time() - t0
        print(f"\r[4/4] c(N) [{'#' * int(30 * fr)}{'-' * (30 - int(30 * fr))}] {100 * fr:5.1f}%  "
              f"{i + 1}/{len(NS)}  decorrido {_hms(el)}  faltam ~{_hms(el * (1 - fr) / fr)}  "
              f"(total {_hms(time.time() - T_INICIO)})", end='', flush=True)
    print(f"\n[4/4] c(N): concluido em {_hms(time.time() - t0)}")

    return dict(NS=NS, KAPS=KAPS, sig_grade=sig_grade, F_grade=F_grade,
                KAPPA0=KAPPA0, ALPHAS=ALPHAS, sig_alpha=sig_alpha, F_alpha=F_alpha,
                c=np.array(c), sig0=sig_grade[:, i0], F0=F_grade[:, i0],
                M=M, SEED=SEED, H_DERIV=H_DERIV, gam=gam, K_SIG=K_SIG, PTS=PTS, N_MIN=N_MIN)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Infidelidade x N com osciladores lentos.")
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--sem-validar', action='store_true', help="pula as validacoes")
    A = ap.parse_args()
    print(f"N = {NS.tolist()}\n{len(KAPS)} valores de kappa, alphas = {ALPHAS.tolist()}, M = {M}")
    if not A.sem_validar:
        validar()
    else:
        print("[1/4] validacoes: puladas")
    dados = calcular(A.procs)
    os.makedirs(PASTA, exist_ok=True)
    caminho = os.path.join(os.getcwd(), PASTA, ARQUIVO)
    np.savez(caminho, **dados)
    print(f"salvo: {caminho}")
    print(f"ok -- tempo total {_hms(time.time() - T_INICIO)}")
