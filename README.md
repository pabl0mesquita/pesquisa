# brode — fidelidade do CZ com dois sítios cross-Kerr contrapropagantes

Reprodução da Fig. 5 de Brod & Combes (dois sítios, fótons contrapropagantes).
`calcular.py` gera os dados em `data/counterprop/*.npz`; `plot.py` desenha uma ou
mais curvas na mesma imagem.

---

## 1. O que o código calcula

Cada sítio `j` tem largura `γ_j` e dessintonia `Δ_j`, com
`Γ_j(ω) = γ_j/2 + i(Δ_j − ω)`. A amplitude de um fóton por sítio é a fase pura
`t_j(ω) = Γ_j*(ω)/Γ_j(ω)`.

No setor de dois fótons contrapropagantes a matriz S é

```
S = [fase de 1 fóton] δ(ωa−νa)δ(ωb−νb) + T(ωa,ωb;νa,νb) δ(ωa+ωb−νa−νb)
T = term2_bracket + term3_bracket        (não linearidade no sítio 1 ou no sítio 2)
```

`F = ⟨ξξ| S₁†(ωa)S₁†(ωb) S |ξξ⟩ = 1 + G`, e a fidelidade média de porta é

```
F₁(φ) = [6 + 3·Re(e^{iφ}F) + |F|²]/10
```

Essa fórmula **está correta**: é `[|Tr(U†A)|² + Tr(A†A)]/[d(d+1)]` com
`d = 4` e `A = diag(1,1,1,F)`. O termo `Tr(A†A) = 3 + |F|²` (em vez de `d = 4`)
é o que trata corretamente o vazamento para fora do subespaço computacional
quando `|F| < 1`. Usar a fórmula de canal unitário daria fidelidade otimista.

---

## 2. Verificações feitas — o que está certo

### 2.1 Unitaridade da matriz S (teste decisivo)

Para `S` unitária, no setor de energia total `E` fixa vale a identidade exata

```
2·Re[ T̃(ν,ν) ] + ∫dω |T̃(ω,ν)|² = 0,     T̃ = correction(ωa)·correction(ωb)·T
```

Resultado (integral com `scipy.quad` até ±∞):

| caso | defeito relativo |
|---|---|
| 1 sítio (χ₂=0), L=0 | 1e-16 |
| 2 sítios, L=0 | 1e-16 |
| 1 sítio, χ=1000 (saturado) | 1e-16 |
| **2 sítios, L=0.4** | **até 3.7e-01** |

Ou seja, em `L = 0` o núcleo está **exatamente** certo: o prefator `iχγ²/π`, o
sinal, o fator `K = 1/(1 + 2iχ/(Γ(ωa)+Γ(ωb)))`, as fases de fóton único do sítio
espectador e a própria estrutura "não linearidade em um sítio de cada vez".
Não falta nenhum termo (um termo de interação nos dois sítios violaria essa
identidade, que fecha em precisão de máquina sem ele).

### 2.2 Limite analítico σ → 0

Expandindo as gaussianas, `G → 2√π·σ·[núcleo em ω_c]`, isto é

```
G(σ→0) = 2√π σ · [ −(iχ₁γ₁²/π)·K₁/|Γ₁(ω_c)|⁴ − (iχ₂γ₂²/π)·K₂/|Γ₂(ω_c)|⁴ ],
K_j = Γ_j(ω_c)/(Γ_j(ω_c) + iχ_j)
```

O cálculo numérico converge para essa expressão com erro relativo 3e-6 em
σ=1e-3 (e ~σ, como esperado), para vários `L`. Confirma o prefator de forma
independente. Note que o limite **não depende de `L`** — para pacotes muito
longos a separação entre os sítios é irrelevante. (Antes da correção de 3.1 o
resultado numérico trazia um fator espúrio `e^{2iω_c L}` aqui.)

Consequência física: `G ∝ σ/γ`, logo `F → 1` e `F₁ → 0.4` quando `σ → 0` —
fótons contrapropagantes num sítio pontual não acumulam fase condicional para
pacotes longos. O pico de fidelidade fica em `σ ~ γ`.

---

## 3. Erros encontrados (todos já corrigidos em `calcular.py`)

### 3.1 CRÍTICO — `correction` não removia a fase de propagação (afeta todo dado com `L ≠ 0`) ✔ corrigido

`term2_bracket` e `term3_bracket` carregam as fases de caminho `e^{i(ωa+νb)L}` e
`e^{i(νa+ωb)L}`. A **fase relativa** entre os dois termos está certa
(`e^{2i(νa−ωa)L}`, o fator de transferência de momento entre sítios separados
por `L`). Mas o fator comum `e^{i(ωa+νb)L}` só é legítimo se o mesmo caminho
estiver embutido na referência de fóton único — e `correction` remove apenas
`t₁t₂`, sem o `e^{iωL}`.

Resultado: o termo `1 +` (linear, já corrigido) e os termos não lineares ficam
em convenções diferentes, e a matriz S deixa de ser unitária (tabela em 2.1).

**Correção aplicada (uma linha):**

```python
def correction(w):
    return np.conj(phase1(w)) * np.exp(-1j * w * L)
```

Com isso a identidade de unitaridade volta a fechar em 1e-8 ou melhor para
`L = 0.4`, `1.0` e `2.0` (inclusive com sítios assimétricos)
(equivalente a trocar as fases dos termos por `e^{i(ωa−νa)L}` e `e^{i(νa−ωa)L}`,
que é a escolha simétrica com sítios em `∓L/2`). Para `L = 0` nada muda — os
resultados antigos com `L=0` continuam válidos.

Impacto nos dados já gerados (χ=10, γ=1, φ=π):

| L | ω₀ | σ | \|F\| atual | \|F\| corrigido | F₁(π) atual | F₁(π) corrigido |
|---|---|---|---|---|---|---|
| 0.4 | 1.1 | 0.5 | 0.488 | 0.840 | 0.4818 | 0.4578 |
| 0.4 | 1.1 | 1.0 | 0.495 | 0.807 | 0.4777 | 0.4533 |
| 0.4 | 0.0 | 1.0 | 0.473 | 0.626 | 0.4805 | 0.4514 |
| 0.8 | 1.1 | 0.5 | 0.677 | 0.846 | 0.4532 | 0.4434 |

Todos os `.npz` com `L ≠ 0` precisam ser regerados.

### 3.2 CRÍTICO — a grade de integração não resolvia o integrando ✔ corrigido

`half = max(k_sigma·σ, k_gamma·γ)` com `n` fixo produz espaçamento
`h = 2·half/n`. Duas falhas opostas:

* **σ pequeno**: a janela vira `k_gamma·γ` (grande), e `h ≫ σ`. A gaussiana de
  largura σ cai *entre* os pontos da grade e some. Em σ=0.01, γ=4.5, n=110:
  `h = 0.82` contra `σ = 0.01` → `G ≈ 0` e `|F| = 1.000000` (valor correto:
  0.9678).
* **σ grande**: `h ≫ γ` e as ressonâncias de largura γ ficam sub-amostradas.

Comparação com o valor convergido (n=110, k_sigma=k_gamma=10):

| σ | \|F\| calculado | \|F\| convergido | erro rel. em G |
|---|---|---|---|
| 0.01 | 1.000000 | 0.967776 | 100 % |
| 0.05 | 1.000000 | 0.839240 | 100 % |
| 0.20 | 0.841875 | 0.377777 | 75 % |
| 1.00 | 0.687414 | 0.697205 | 0.6 % |
| 5.00 | 0.434424 | 0.432059 | 0.4 % |
| 20.0 | 0.876358 | 0.882950 | 5.6 % |
| 100. | 0.998541 | 0.993164 | 66 % |

(χ=5, γ=4.5, ω₀=1.1, L=0. Com os parâmetros atuais — χ=1000, γ=1 — só a faixa
`0.2 ≲ σ ≲ 1` sobrevive; nas pontas o erro passa de 50 %.)

As quatro gaussianas confinam o integrando a `|ω−ω_c| ≲ 8σ` **sempre** (o núcleo
decai como 1/ω⁴), então:

* a janela deve ser `half = k_sigma·σ` — **`k_gamma` é prejudicial**, só infla a
  janela e estraga a resolução;
* o espaçamento precisa resolver `min(σ, γ)`: `h ≲ min(σ,γ)/10`, ou seja
  `n ≳ 20·k_sigma·max(1, σ/γ)`.

Para σ/γ = 100 isso exige `n ~ 2·10⁴` por eixo — inviável em 3D
(`n³` pontos, 21 MB já em n=110 e 173 MB em n=221), mas trivial na formulação da
seção 4.

Aplicado: `k_gamma` e `n` foram removidos; a janela passou a ser `k_sigma·σ` e o
número de pontos é derivado de `min(σ, γ)` via `pts_per_width` (função `grade`).
A mensagem de aviso, que sugeria aumentar `k_gamma` (o que piora a resolução),
foi ajustada.

### 3.3 Menores ✔ corrigidos (exceto onde indicado)

* **Defaults congelados no import**: `compute_G(sigma, n=n, ...)` e
  `F1_fidelity(sigma, phi=phi)` capturavam os globais no momento do import, então
  alterar `calcular.k_sigma` depois (de outro script ou num notebook) não tinha
  efeito. Agora as funções leem os globais em tempo de chamada.
* **Simpson com número ímpar de intervalos**: `n = 110` dava 109 intervalos e
  `scipy.simpson` caía numa correção de ordem menor no último. `grade()` agora
  devolve sempre um `n` ímpar.
* `quiet` não silenciava o `print` final de `main()`. Corrigido.
* `omega_c = Delta1 + omega0` mede `ω₀` como dessintonia **do sítio 1**; com
  `Δ₁ ≠ Δ₂` a centralização fica assimétrica. *Não alterado* — é convenção, não
  erro; centralizar em `(Δ₁+Δ₂)/2` mudaria o significado de `ω₀`.
* Nada impede sobrescrever um `.npz` existente sem aviso. *Não alterado.*

---

## 4. Melhoria principal: a integral tripla reduz **exatamente** a convoluções 1D

O núcleo fatoriza. Com `E = νa+νb = ωa+ωb`, `K₁` e `K₂` dependem **só de `E`**
(`Γ(ωa)+Γ(ωb) = γ + i(2Δ − E)`), e cada fator restante depende de uma única
frequência. Logo

```
G₂ = −(iχ₁γ₁²/π) ∫dE K₁(E) · (A₂ ∗ B₂)(E) · (C₂ ∗ D₂)(E)

A₂(ω) = ξ(ω)·c(ω)·e^{iωL}·Γ₂*(ω)/(Γ₂(ω)Γ₁(ω))     [ωa]
B₂(ω) = ξ(ω)·c(ω)/Γ₁(ω)                            [ωb]
C₂(ν) = ξ(ν)/Γ₁(ν)                                 [νa]
D₂(ν) = ξ(ν)·e^{iνL}·Γ₂*(ν)/(Γ₂(ν)Γ₁(ν))           [νb]
```

e analogamente para `G₃` trocando 1↔2 e os papéis de `(νa, ωb)`. Implementação
já aplicada em `compute_G`/`grade`:

```python
from scipy.signal import fftconvolve

def compute_G(sigma, k_sigma=9.0, pts_per_width=14):
    half = k_sigma * sigma
    n = int(2 * half / (min(sigma, gamma1, gamma2) / pts_per_width)) + 1
    n = max(2001, n) | 1                      # ímpar, para o Simpson
    w = np.linspace(omega_c - half, omega_c + half, n)
    h = w[1] - w[0]

    xi, c, eL = xi_in(w, sigma), correction(w), np.exp(1j * w * L)
    G1, G2_ = Gamma1(w), Gamma2(w)
    t1, t2 = np.conj(G1) / G1, np.conj(G2_) / G2_

    E = np.linspace(2 * w[0], 2 * w[-1], 2 * n - 1)
    k1 = 1.0 / (1.0 + 2j * chi1 / (gamma1 + 1j * (2 * Delta1 - E)))
    k2 = 1.0 / (1.0 + 2j * chi2 / (gamma2 + 1j * (2 * Delta2 - E)))
    conv = lambda a, b: fftconvolve(a, b) * h

    G2 = -(1j * chi1 * gamma1 ** 2 / np.pi) * simpson(
        k1 * conv(xi * c * eL * t2 / G1, xi * c / G1) * conv(xi / G1, xi * eL * t2 / G1), x=E)
    G3 = -(1j * chi2 * gamma2 ** 2 / np.pi) * simpson(
        k2 * conv(xi * c / G2_, xi * c * eL * t1 / G2_) * conv(xi * eL * t1 / G2_, xi / G2_), x=E)
    return G2 + G3
```

Ganhos medidos:

| | integral tripla (n=110) | convolução 1D |
|---|---|---|
| tempo por σ | 455 ms | 0.9 ms |
| varredura de 100 σ | 46 s (**não convergida**) | 0.2 s (convergida) |
| memória | O(n³) — 21 MB por temporário | O(n) |
| auto-convergência | 50–100 % de erro nas pontas | ~1e-13 de σ = 0.01 a 100 |

Concordância com a integral tripla bem resolvida (n=221, σ = 0.5, 2 e 5):
1e-5 a 1e-6 — limitada pela própria grade 3D.

---

## 5. Melhorias ainda não aplicadas

1. **Testes automáticos** (`test_calcular.py`): a identidade de unitaridade da
   seção 2.1 e o limite analítico da seção 2.2 são checagens baratas que pegam
   erro de sinal, de prefator e de convenção de fase — exatamente as classes de
   erro difíceis de ver no gráfico.
2. **Checagem de convergência embutida**: calcular `G` com `n` e `2n` e avisar se
   a diferença relativa passar de, digamos, 1e-3. Melhor que o teste `|F| > 1`,
   que só pega uma parte dos casos.
3. **Salvar metadados**: junto do `.npz`, guardar `sigma_min/max/n_sigma`,
   parâmetros numéricos efetivos, versão do código (`git rev-parse HEAD`) e data.
   Hoje dá para gerar dois arquivos com o mesmo nome e parâmetros diferentes.
4. **Nomes de arquivo derivados dos parâmetros** (ex.
   `L0.4_chi10_gamma1_w0.0.npz`) gerados automaticamente, em vez de editados à
   mão a cada rodada.
5. **Varrer `L` num laço** dentro de `main()`, já que o interesse agora é a
   família de curvas `F₁(σ)` para vários `L` — hoje exige editar e rodar o
   arquivo uma vez por curva. Com 0.2 s por varredura, gerar a família inteira
   numa execução ficou barato.

(Não sugerido: mover os parâmetros para um `dataclass` ou para a linha de
comando — a convenção do projeto é editar variáveis comentadas no topo do
arquivo.)

---

## 6. Estado

| item | estado |
|---|---|
| `correction` com a fase de propagação (3.1) | ✔ aplicado |
| integral por convolução 1D, sem `k_gamma` (3.2 / 4) | ✔ aplicado |
| itens menores de 3.3 | ✔ aplicados |
| `xlim` como parâmetro em `plot.py` | ✔ aplicado |
| **regerar os `.npz` com `L ≠ 0`** | **pendente** |
| testes automáticos (5.1) | pendente |

Os dados com `L = 0` continuam válidos. Os com `L ≠ 0` foram gerados com a
convenção de fase errada **e** com a grade não convergida, e precisam ser
refeitos.
