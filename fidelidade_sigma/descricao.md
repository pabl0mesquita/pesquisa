# Figura `fidelity_sigma_all.svg`

## Descrição

A figura mostra a fidelidade $F_1$ da porta de fase condicional (CPHASE) implementada passivamente por dois fótons que interagem com dois sítios não lineares do tipo Kerr, em configuração contrapropagante, em função da largura espectral $\sigma$ do pacote de onda incidente. O eixo das abscissas está em escala logarítmica, $\sigma \in [10^{-2}, 10^{1}]$, e o eixo das ordenadas cobre $F_1 \in [0{,}3;\, 1{,}02]$. Cada curva corresponde a uma separação distinta $L$ entre os sítios, indicada na legenda como $L/\gamma$.

## O que é calculado

Os fótons de entrada são descritos por um pacote gaussiano normalizado no domínio das frequências,

$$
\xi_{\rm in}(\omega) = \left(\frac{1}{2\pi\sigma^{2}}\right)^{1/4} \exp\!\left[-\frac{(\omega-\omega_c)^{2}}{4\sigma^{2}}\right],
\qquad \omega_c = \Delta_1 + \omega_0 .
$$

Cada sítio $j=1,2$ é caracterizado pelo propagador $\Gamma_j(\omega) = \gamma_j/2 + i(\Delta_j - \omega)$ e por um acoplamento não linear $\chi_j$. A amplitude de espalhamento de dois fótons (matriz $S$) é dividida em um termo linear, dado pelo produto das fases de fóton único, e em contribuições não lineares em que o sítio 1 (termo 2) ou o sítio 2 (termo 3) sofre a interação, com o núcleo

$$
K_j(\omega_a,\omega_b) = \left[1 + \frac{2i\chi_j}{\Gamma_j(\omega_a)+\Gamma_j(\omega_b)}\right]^{-1}
$$

e a fase de propagação $e^{i\omega L}$ entre os sítios. Removida a fase linear de fóton único, a projeção do estado de saída sobre o estado de entrada define a amplitude complexa

$$
F = 1 + G, \qquad
G = \iiint d\nu_a\, d\nu_b\, d\omega_a\; \xi_{\rm in}(\nu_a)\xi_{\rm in}(\nu_b)\xi_{\rm in}(\omega_a)\xi_{\rm in}(\omega_b)\;
\mathcal{C}(\omega_a)\,\mathcal{C}(\omega_b)\,\big[T_2 + T_3\big],
$$

em que $\omega_b = \nu_a+\nu_b-\omega_a$ (conservação de energia), $\mathcal{C}$ é a correção que remove a amplitude linear e $T_2$, $T_3$ são os termos não lineares dos sítios 1 e 2. Como o núcleo depende da energia total $E=\omega_a+\omega_b=\nu_a+\nu_b$ e os demais fatores dependem de uma única frequência, a integral tripla foi reduzida a uma integral unidimensional em $E$ sobre o produto de duas convoluções (calculadas por FFT), integrada pela regra de Simpson.

A grandeza plotada é a fidelidade

$$
F_1(\phi) = \frac{6 + 3\,\mathrm{Re}\!\left(e^{i\phi}F\right) + |F|^{2}}{10},
$$

avaliada em $\phi=\pi$, isto é, para uma porta CPHASE com fase condicional $\pi$. Nessa convenção, $F=-1$ (fase $\pi$ adquirida sem perdas) fornece $F_1=1$, enquanto $F=1$ (ausência de efeito não linear) fornece $F_1=0{,}4$.

## Parâmetros utilizados

Todas as grandezas estão em unidades de $\gamma$ (com $\gamma_1=\gamma_2=\gamma=1$).

| Parâmetro | Valor |
|---|---|
| Acoplamentos não lineares $\chi_1=\chi_2$ | $1000$ |
| Taxas de decaimento $\gamma_1=\gamma_2$ | $1$ |
| Dessintonias $\Delta_1=\Delta_2$ | $0$ |
| Frequência central do pacote $\omega_0$ (relativa a $\Delta_1$) | $0$ |
| Fase da porta $\phi$ | $\pi$ |
| Separação entre os sítios $L/\gamma$ | $0;\ 0{,}4;\ 0{,}6;\ 0{,}8;\ 1{,}0;\ 2{,}0$ |
| Varredura em $\sigma$ (dados calculados) | 100 pontos, espaçados logaritmicamente entre $10^{-2}$ e $10^{2}$ (plotado em $10^{-2}$–$10^{1}$) |

Parâmetros numéricos da integração: janela de frequências $|\omega-\omega_c|\le 9\sigma$ ($k_\sigma=9$); espaçamento da grade escolhido para resolver a menor escala presente, $\min(\sigma,\gamma)$, com 14 pontos por escala (`pts_per_width = 14`), mínimo de 2001 e máximo de 600001 pontos.

Parâmetros de plotagem: `matplotlib` 3.10.1, eixo $x$ logarítmico, curvas com marcadores circulares semitransparentes ($\alpha=0{,}5$) e linhas finas, título $\chi=1000,\ \gamma=1,\ \omega_0=0$, eixos "Largura do pacote de onda $\sigma$" e "Fidelidade $F_1$".

## Resultados observados

- Para todos os valores de $L$, $F_1$ é não monotônica em $\sigma$, com um máximo em $\sigma \sim 0{,}1$–$0{,}2\,\gamma$, correspondente ao regime em que o pacote é espectralmente estreito em relação à largura de linha dos sítios, mas ainda suficientemente localizado para que a interação não linear ocorra dentro da janela temporal de interação.
- Para $\sigma \lesssim 0{,}03$ as curvas coincidem, com $F_1 \approx 0{,}42$ em $\sigma=10^{-2}$, o que mostra que a dependência em $L$ é desprezível para pacotes muito estreitos (longo comprimento de coerência).
- Para $\sigma \gg \gamma$ todas as curvas convergem para $F_1 \to 0{,}4$, valor correspondente a $F\to 1$: pacotes de banda larga em relação a $\gamma$ não adquirem a fase não linear.
- O valor de pico diminui com o aumento de $L$: aproximadamente $0{,}86$ para $L/\gamma=0$, $0{,}81$ para $0{,}4$, $0{,}76$ para $0{,}8$, $0{,}74$ para $1{,}0$ e $0{,}67$ para $2{,}0$. O pico se desloca ligeiramente para menores $\sigma$ quando $L$ aumenta.

## Referência

Brod, D. J. & Combes, J. — porta de fase condicional passiva por não linearidades do tipo cross-Kerr, caso de dois sítios contrapropagante (Fig. 5).
