# Script `heatmap_omegaa_omegab_n_sitios.py`

## Descrição

O script desenha um mapa de calor da **amplitude do par de fótons espalhado** no plano das frequências de saída $(\omega_a, \omega_b)$, para uma cadeia de $N$ sítios não lineares idênticos do tipo cross-Kerr, com os dois fótons se propagando em sentidos opostos (configuração contrapropagante). O pacote de entrada é fixo: os dois fótons chegam no mesmo pacote gaussiano, centrado em $\omega_c$ e com largura $\sigma$.

A grandeza calculada é

$$
\psi_{\rm out}(\omega_a,\omega_b) = \langle \omega_a\, \omega_b |\, S \,|\, \xi\, \xi \rangle ,
$$

isto é, a função de onda espectral de dois fótons depois da interação com os $N$ sítios. O eixo $x$ da figura é $\omega_a$ e o eixo $y$ é $\omega_b$; a cor codifica (por padrão) o módulo $|\psi_{\rm out}|$.

O script generaliza `heatmap_omegaa_omegab.py` (caso $N=2$) usando a mesma matriz $S$ de $N$ sítios de `fidelidade_sigma/calcular_sigma_n_sitios.py`. Enquanto aquele script reduz tudo a um número (a fidelidade $F_1$), este mostra **onde, no espaço de frequências, está a amplitude de saída**: quanto dela continua no pacote original, quanto foi redistribuída pela não linearidade e com que estrutura.

Arquivos gerados (com $\chi=1000$, $\gamma=1$, $\omega_0=0$, $L=0$, $\sigma=1$):

| Arquivo | $N$ | Faixa dos eixos |
|---|---|---|
| `heatmap_omegaa_omegab_N1_l0_sigma_1.0.svg` | $1$ | $[-3, 3]$ |
| `heatmap_omegaa_omegab_N2_l0_sigma_1.0.svg` | $2$ | $[-3, 3]$ |
| `heatmap_omegaa_omegab_N4_l0_sigma_1.0.svg` | $4$ | $[-10, 10]$ |
| `heatmap_omegaa_omegab_N10_l0_sigma_1.0.svg` | $10$ | $[-3, 3]$ |

---

## 1. Modelo físico

### 1.1 Um sítio

Cada sítio tem taxa de decaimento $\gamma$, dessintonia $\Delta$ e acoplamento não linear $\chi$ (iguais em todos os sítios). A resposta linear de um sítio é governada por

$$
\Gamma(\omega) = \frac{\gamma}{2} + i(\Delta - \omega),
$$

e a amplitude de transmissão de **um** fóton por **um** sítio é uma fase pura,

$$
t(\omega) = \frac{\Gamma^*(\omega)}{\Gamma(\omega)}, \qquad |t(\omega)| = 1 .
$$

Na ressonância ($\omega=\Delta$) $\Gamma = \gamma/2$ é real e $t=1$; longe dela ($|\omega-\Delta| \gg \gamma$) tem-se $t \to -1$. A fase de $t$ dá uma volta completa ($2\pi$) ao atravessar a ressonância, quase toda numa janela de largura $\sim\gamma$ em torno de $\Delta$.

### 1.2 Propagação entre sítios

Os sítios estão separados por uma distância $L$ (em unidades com velocidade de grupo $=1$), de modo que a propagação de um sítio ao próximo acrescenta a fase $e^{i\omega L}$. Define-se a fase "sítio + propagação"

$$
a(\omega) = t(\omega)\, e^{i\omega L},
$$

e a amplitude completa de um fóton atravessando a cadeia inteira:

$$
\langle \omega^- | \nu^+ \rangle = S_1(\omega)\, \delta(\omega - \nu), \qquad
S_1(\omega) = e^{i(N-1)\omega L}\, t(\omega)^N .
$$

O expoente é $N-1$ (e não $N$) porque há $N-1$ trechos de propagação entre $N$ sítios.

### 1.3 Matriz $S$ de dois fótons

No setor de dois fótons contrapropagantes a matriz $S$ é

$$
S = \langle \omega_a^- | \nu_a^+ \rangle \langle \omega_b^- | \nu_b^+ \rangle
\;-\; \frac{i\chi\gamma^2}{\pi}\,
\frac{K(\nu_a,\nu_b)\;\delta(\omega_a+\omega_b-\nu_a-\nu_b)}
{\Gamma(\nu_b)\,\Gamma(\nu_a)\,\Gamma(\omega_b)\,\Gamma(\omega_a)}
\sum_{j=1}^{N} A^{N-j} B^{j-1},
$$

com

$$
A = a(\omega_a)\, a(\nu_b), \qquad B = a(\omega_b)\, a(\nu_a),
$$

e o núcleo não linear

$$
K(\nu_a,\nu_b) = \left[1 + \frac{2i\chi}{\Gamma(\nu_a) + \Gamma(\nu_b)}\right]^{-1}.
$$

- O **primeiro termo** é o espalhamento linear: cada fóton atravessa a cadeia independentemente, e só ganha a fase $S_1$.
- O **segundo termo** é a contribuição não linear. O índice $j$ da soma indica em qual sítio a interação acontece; os fatores $A^{N-j}$ e $B^{j-1}$ contabilizam as fases lineares que cada fóton acumula antes e depois desse sítio (como os fótons são contrapropagantes, um fóton passa por $j-1$ sítios antes do sítio $j$ e o outro por $N-j$).
- A delta $\delta(\omega_a+\omega_b-\nu_a-\nu_b)$ impõe conservação da **energia total**, mas não das frequências individuais: a interação redistribui frequência entre os fótons.

Como $\Gamma(\nu_a)+\Gamma(\nu_b) = \gamma + i(2\Delta - E)$, com $E = \nu_a+\nu_b$, o núcleo só depende da energia total:

$$
K(E) = \left[1 + \frac{2i\chi}{\gamma + i(2\Delta - E)}\right]^{-1}.
$$

**Limite de $\chi$ grande.** O fator que aparece na matriz $S$ é $\chi K(E)$, e

$$
\chi K(E) = \frac{\chi\,[\gamma + i(2\Delta-E)]}{\gamma + i(2\Delta-E) + 2i\chi}
\;\xrightarrow{\;\chi\to\infty\;}\; \frac{\gamma + i(2\Delta-E)}{2i},
$$

ou seja, o termo não linear satura. Com $\chi = 1000$ e $|\gamma + i(2\Delta-E)| \sim 1$, os resultados já estão, na prática, nesse limite (correção relativa $\sim 10^{-3}$).

### 1.4 Pacote de entrada

Os dois fótons chegam no mesmo pacote gaussiano normalizado,

$$
\xi_{\rm in}(\omega) = \left(\frac{1}{2\pi\sigma^{2}}\right)^{1/4}
\exp\!\left[-\frac{(\omega-\omega_c)^{2}}{4\sigma^{2}}\right],
\qquad \omega_c = \Delta + \omega_0 ,
$$

com $\int d\omega\, |\xi_{\rm in}(\omega)|^2 = 1$. O estado de entrada é $|\xi\,\xi\rangle = \iint d\nu_a\, d\nu_b\; \xi_{\rm in}(\nu_a)\,\xi_{\rm in}(\nu_b)\, |\nu_a\, \nu_b\rangle$.

Valor de referência útil para ler a barra de cores: o pacote de entrada tem, no centro, $|\xi_{\rm in}(\omega_c)|^2 = 1/(\sqrt{2\pi}\,\sigma) \approx 0{,}399$ para $\sigma = 1$.

---

## 2. O que é calculado

Aplicando $S$ ao estado de entrada e projetando em $\langle \omega_a\, \omega_b|$, a parte linear fica diagonal e a parte não linear tem as duas integrais em $\nu_a, \nu_b$, das quais a delta de energia elimina uma ($\nu_b = E - \nu_a$, com $E = \omega_a + \omega_b$). O resultado é

$$
\psi_{\rm out}(\omega_a,\omega_b) =
\underbrace{\xi_{\rm in}(\omega_a)\,\xi_{\rm in}(\omega_b)\, S_1(\omega_a)\, S_1(\omega_b)}_{\text{linear}}
\;\underbrace{-\;\frac{i\chi\gamma^2}{\pi}\,
\frac{K(E)}{\Gamma(\omega_a)\,\Gamma(\omega_b)}
\sum_{j=1}^{N} a(\omega_a)^{N-j}\, a(\omega_b)^{j-1}\, P_j(E)}_{\text{não linear}},
$$

em que toda a dependência nas frequências de entrada ficou concentrada em

$$
P_j(E) = \int d\nu\;
\frac{\xi_{\rm in}(\nu)\, a(\nu)^{j-1}}{\Gamma(\nu)}\;
\frac{\xi_{\rm in}(E-\nu)\, a(E-\nu)^{N-j}}{\Gamma(E-\nu)} .
$$

$P_j$ é uma **convolução unidimensional** das funções

$$
C_j(\nu) = \frac{\xi_{\rm in}(\nu)\, a(\nu)^{j-1}}{\Gamma(\nu)}, \qquad
D_j(\nu) = \frac{\xi_{\rm in}(\nu)\, a(\nu)^{N-j}}{\Gamma(\nu)},
\qquad P_j(E) = (C_j * D_j)(E),
$$

avaliada em $E = \omega_a + \omega_b$. Essa é a observação que torna o cálculo barato: em vez de uma integral para cada um dos $n_\omega^2$ pixels, bastam $N$ convoluções 1D (feitas por FFT), que depois são interpoladas em $E$ para todos os pixels de uma vez.

### 2.1 Correção de fase (`aplicar_correcao`)

A fase linear $S_1(\omega_a) S_1(\omega_b)$ é a mesma que cada fóton ganharia sozinho, e não carrega informação sobre a interação. Com `aplicar_correcao = True`, as duas partes são multiplicadas por

$$
\mathcal{C}(\omega_a,\omega_b) = S_1^*(\omega_a)\, S_1^*(\omega_b),
$$

o que equivale a medir o estado de saída em relação ao alvo "linear" $S_1\times S_1$ (a mesma convenção de `correction()` em `calcular_sigma_n_sitios.py`). A parte linear passa a ser simplesmente $\xi_{\rm in}(\omega_a)\,\xi_{\rm in}(\omega_b)$, real e positiva. Como $|S_1| = 1$, a correção **não altera** `abs` nem `abs_nl`; ela só importa para `real`, `imag` e `phase`.

Nessa convenção, uma porta de fase condicional ideal com fase $\pi$ corresponderia a $\psi_{\rm out} = -\,\xi_{\rm in}(\omega_a)\,\xi_{\rm in}(\omega_b)$: o mesmo módulo do pacote de entrada, com sinal trocado.

---

## 3. Funções

### `Gamma(w)`

$$
\Gamma(\omega) = \frac{\gamma}{2} + i(\Delta - \omega).
$$

Denominador de ressonância de um sítio. Aparece nos denominadores do termo não linear e define $t(\omega)$ e $K(E)$. Aceita escalares ou arrays.

### `t_site(w)`

$$
t(\omega) = \frac{\Gamma^*(\omega)}{\Gamma(\omega)}.
$$

Amplitude de transmissão de um fóton por um único sítio. É uma fase pura ($|t|=1$).

### `a_site(w)`

$$
a(\omega) = t(\omega)\, e^{i\omega L}.
$$

Fase de um sítio seguida da propagação até o próximo. É o "tijolo" dos fatores $A$ e $B$ da matriz $S$ e das funções $C_j$, $D_j$.

### `S1(w)`

$$
S_1(\omega) = t(\omega)^N\, e^{i(N-1)\omega L}.
$$

Amplitude de um fóton atravessar a cadeia completa ($N$ sítios e $N-1$ trechos de propagação). Usada no termo linear e na correção de fase.

### `xi_in(w)`

$$
\xi_{\rm in}(\omega) = \left(\frac{1}{2\pi\sigma^{2}}\right)^{1/4}
\exp\!\left[-\frac{(\omega-\omega_c)^{2}}{4\sigma^{2}}\right].
$$

Pacote gaussiano normalizado de entrada (Eq. (4) da referência). Note que a largura de $|\xi_{\rm in}|^2$ (desvio padrão) é $\sigma$.

### `grade()`

Constrói a grade de frequências $\{\nu_k\}$ usada nas convoluções $P_j(E)$ e devolve `(w, h)`, a grade e seu espaçamento.

- **Janela:** $\nu \in [\omega_c - k_\sigma\sigma,\; \omega_c + k_\sigma\sigma]$. Fora dela a gaussiana é desprezível ($e^{-k_\sigma^2/4} \approx 1{,}6\times10^{-9}$ para $k_\sigma = 9$).
- **Espaçamento:** resolve a menor escala presente no integrando,
  $$
  h \approx \frac{\min(\sigma,\; \gamma,\; 1/(N|L|))}{\texttt{pts\_per\_width}},
  $$
  em que $1/(N|L|)$ só entra se $L \neq 0$: é o período (em $\omega$) da fase $e^{i\omega L}$ elevada até a potência $N-1$, que oscila mais rápido quanto maior a cadeia.
- **Número de pontos:** $n = \lfloor 2k_\sigma\sigma / h \rfloor + 1$, limitado a $[n_{\min}, n_{\max}]$ e forçado a ser ímpar (`| 1`), para que a grade tenha um ponto exatamente no centro $\omega_c$.

Exemplo: para $\sigma=\gamma=1$ e $L=0$, a regra dá $n = 253$, e o piso $n_{\min} = 2001$ prevalece ($h = 0{,}009$).

### `psi_out(WA, WB)`

Calcula $\psi_{\rm out}$ em todos os pontos das matrizes `WA`, `WB` (as coordenadas $\omega_a$, $\omega_b$ de cada pixel). Devolve a tupla `(total, nao_linear)`. Passo a passo:

1. Monta a grade `w` (espaçamento `h`) com `grade()` e pré-calcula `base` $= \xi_{\rm in}(\nu)/\Gamma(\nu)$ e `a` $= a(\nu)$.
2. Monta a grade de energias `E_grid`: a convolução discreta de dois vetores de $n$ pontos tem $2n-1$ pontos, cobrindo $E \in [2\nu_{\min},\, 2\nu_{\max}]$ com o mesmo espaçamento $h$.
3. Para cada $j = 1,\dots,N$:
   - calcula $P_j$ na grade de energias via `fftconvolve(C_j, D_j) * h` (o fator $h$ transforma a soma discreta na integral, regra do retângulo);
   - interpola $P_j$ (partes real e imaginária separadamente) em $E = \omega_a+\omega_b$ de cada pixel, com valor $0$ fora de `E_grid`;
   - acumula $a(\omega_a)^{N-j}\, a(\omega_b)^{j-1}\, P_j(E)$ em `soma`.
4. Calcula $K(E)$, o termo linear $\xi_{\rm in}(\omega_a)\xi_{\rm in}(\omega_b)S_1(\omega_a)S_1(\omega_b)$ e o termo não linear
   $$
   -\frac{i\chi\gamma^2}{\pi}\,\frac{K(E)}{\Gamma(\omega_a)\Gamma(\omega_b)}\;\texttt{soma}.
   $$
5. Se `aplicar_correcao`, multiplica ambos por $S_1^*(\omega_a)\,S_1^*(\omega_b)$.

O custo é dominado pelas $N$ interpolações sobre as $n_\omega^2$ posições (com $n_\omega = 2000$, são $4\times10^6$ pixels por termo); as convoluções em si são rápidas.

### `main()`

1. Cria o eixo $\omega \in [\omega_{\min}, \omega_{\max}]$ com $n_\omega$ pontos e a malha `(WA, WB)` com `indexing='xy'`, de modo que $\omega_a$ varia ao longo das colunas (eixo $x$) e $\omega_b$ ao longo das linhas (eixo $y$).
2. Chama `psi_out` e escolhe a grandeza a colorir conforme `field`:

   | `field` | Grandeza |
   |---|---|
   | `'abs'` | $\lvert\psi_{\rm out}\rvert$ |
   | `'abs_nl'` | $\lvert\psi_{\rm out}^{\text{não linear}}\rvert$ (só o segundo termo) |
   | `'real'` | $\mathrm{Re}\,\psi_{\rm out}$ |
   | `'imag'` | $\mathrm{Im}\,\psi_{\rm out}$ |
   | `'phase'` | $\arg\psi_{\rm out} \in (-\pi, \pi]$ |

3. Monta o título (se `title is None`) a partir de $N, \chi, \gamma, \omega_0, L, \sigma$.
4. Escolhe o colormap: `cmap` do matplotlib, se definido; caso contrário, interpola a lista `cores` (eventualmente invertida) em `n_niveis` níveis.
5. Desenha com `pcolormesh` (rasterizado, para o SVG não ter milhões de polígonos), com barra de cores rotulada por `FIELD_LABELS`, eixos com a mesma escala (`set_aspect('equal')`), salva em `outfile` e, se `show`, abre a janela.
6. Devolve `(eixo, total)`, útil para usar o script interativamente.

---

## 4. Parâmetros

Todas as frequências estão em unidades de $\gamma$ (com $\gamma = 1$), e $L$ em unidades de $1/\gamma$ (velocidade de grupo $=1$).

### 4.1 Parâmetros físicos

| Variável | Símbolo | Valor | Significado |
|---|---|---|---|
| `N` | $N$ | `10` | número de sítios da cadeia |
| `gamma` | $\gamma$ | `1.0` | taxa de decaimento de cada sítio (largura de linha) |
| `chi` | $\chi$ | `1000.0` | acoplamento não linear cross-Kerr de cada sítio |
| `Delta` | $\Delta$ | `0.0` | dessintonia de cada sítio |
| `omega0` | $\omega_0$ | `0.0` | centro do pacote, **relativo** a $\Delta$ |
| `L` | $L$ | `0` | separação entre sítios vizinhos |
| `sigma` | $\sigma$ | `1.0` | largura espectral do pacote de entrada |
| `omega_c` | $\omega_c = \Delta+\omega_0$ | derivado | centro absoluto do pacote (não editar) |

### 4.2 Plano $(\omega_a, \omega_b)$

| Variável | Valor | Significado |
|---|---|---|
| `omega_min`, `omega_max` | `-3.0`, `3.0` | limites dos dois eixos (a figura é quadrada) |
| `n_omega` | `2000` | pontos por eixo; o heatmap tem $n_\omega^2$ pixels |

A faixa deve cobrir o pacote ($\omega_c \pm 3\sigma$ contém praticamente todo o peso) e as estruturas criadas pela interação, que podem se estender além disso.

### 4.3 Parâmetros numéricos das convoluções $P_j(E)$

| Variável | Símbolo | Valor | Significado |
|---|---|---|---|
| `k_sigma` | $k_\sigma$ | `9.0` | meia-largura da janela de integração em $\nu$, em unidades de $\sigma$ |
| `pts_per_width` | — | `14` | pontos por menor escala, $\min(\sigma, \gamma, 1/(N\lvert L\rvert))$ |
| `n_min` | $n_{\min}$ | `2001` | número mínimo de pontos da grade |
| `n_max` | $n_{\max}$ | `600001` | teto de pontos (limita tempo e memória) |

Como `E_grid` cobre $[2\omega_c - 2k_\sigma\sigma,\; 2\omega_c + 2k_\sigma\sigma]$ e $P_j$ é posto em zero fora dela, a faixa do plano deve satisfazer $|\omega_a+\omega_b - 2\omega_c| < 2k_\sigma\sigma$ (folgadamente satisfeito com os valores acima).

### 4.4 O que colorir

| Variável | Valor | Significado |
|---|---|---|
| `field` | `'abs'` | grandeza plotada (ver tabela em `main()`) |
| `aplicar_correcao` | `True` | remove a fase de fóton único $S_1(\omega_a)S_1(\omega_b)$ |
| `vmin`, `vmax` | `0.0`, `None` | faixa da barra de cores. `vmin = 0` prende o preto em zero, adequado para `'abs'`/`'abs_nl'`; para `'real'`, `'imag'` e `'phase'`, que assumem valores negativos, use `vmin = None` ou uma faixa simétrica, senão os valores negativos são cortados |

### 4.5 Cores

| Variável | Valor | Significado |
|---|---|---|
| `cores` | 9 cores, `#000000` → `#FDF6DC` | rampa sequencial preto → violeta → dourado claro, com passos uniformes de luminosidade (OKLab $L$ de $0$ a $0{,}99$), o que a torna monótona, legível em tons de cinza e para daltônicos |
| `cores_invertidas` | `False` | `True` inverte a rampa (fundo claro, lóbulos escuros) |
| `n_niveis` | `256` | níveis interpolados entre as cores |
| `cmap` | `None` | nome de um colormap do matplotlib (`'magma'`, `'cividis'`, `'twilight'` para `'phase'`); se definido, `cores` é ignorado |

### 4.6 Figura

| Variável | Valor | Significado |
|---|---|---|
| `outfile` | `heatmap_omegaa_omegab_N{N}_l{L}_sigma_{sigma}.svg` | arquivo de saída (o formato é deduzido da extensão) |
| `dpi` | `300` | resolução da parte rasterizada |
| `figsize` | `(7, 6)` | tamanho da figura em polegadas |
| `xlabel`, `ylabel` | $\omega_a$, $\omega_b$ | rótulos dos eixos |
| `title` | `None` | se `None`, é gerado: $N,\ \chi,\ \gamma,\ \omega_0,\ L,\ \sigma$ |
| `show` | `True` | abre a janela interativa além de salvar |
| `FIELD_LABELS` | — | rótulo da barra de cores para cada `field` |

---

## 5. Resultados observados ($\chi=1000$, $\gamma=1$, $\omega_0=0$, $L=0$, $\sigma=1$)

Nas quatro figuras, $|\psi_{\rm out}|$ é simétrico sob $\omega_a \leftrightarrow \omega_b$ e sob $(\omega_a,\omega_b)\to(-\omega_a,-\omega_b)$, como esperado para $L=0$ e pacote centrado na ressonância ($\omega_c = \Delta = 0$).

- **$N=1$:** um único lóbulo centrado na origem, com máximo $\approx 0{,}71$, alongado ao longo da diagonal $\omega_a = \omega_b$ e com caudas ao longo dos eixos. A saída ainda é concentrada perto do pacote original, mas com forma bem diferente da gaussiana de entrada.
- **$N=2$:** surgem linhas nodais escuras (zeros de $|\psi_{\rm out}|$) próximas dos eixos $\omega_a \approx 0$ e $\omega_b \approx 0$, dividindo a amplitude em quatro lóbulos; o máximo cai para $\approx 0{,}46$.
- **$N=4$** (plotado em $[-10,10]$): toda a amplitude permanece confinada em $|\omega| \lesssim 3$, confirmando que a janela $[-3,3]$ usada nos demais casos é suficiente. A estrutura central tem a forma de estrela, com as linhas nodais mais finas e ramificadas.
- **$N=10$:** padrão de interferência com várias franjas hiperbólicas, que se afastam da origem nos quatro quadrantes. O número de franjas cresce com $N$, pois cada termo $j$ da soma carrega uma fase $a(\omega_a)^{N-j}a(\omega_b)^{j-1}$ diferente e os $N$ termos interferem. O máximo é $\approx 0{,}59$.

Em todos os casos a amplitude se espalha para fora do pacote de entrada, cuja amplitude máxima seria $\approx 0{,}40$ (Seção 1.4): o estado de saída **não** é uma cópia do de entrada com fase trocada. Essa redistribuição espectral é uma das causas de a fidelidade $F_1$ calculada em `fidelidade_sigma/` ficar abaixo de $1$.

---

## Referência

Brod, D. J. & Combes, J. — porta de fase condicional passiva por não linearidades do tipo cross-Kerr, com $N$ sítios e fótons contrapropagantes.
