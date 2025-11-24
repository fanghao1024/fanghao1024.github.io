---

layout: post
title: "《衍生证券教程》第三章 Black-Scholes 笔记"
description: ""
author:     "Hao"
category: true
tagline: "衍生品"
tags: [finance math , derivative pricing]

---
格式更好的[html版本](https://www.fanghao.work/2025/11/19/%E8%A1%8D%E7%94%9F%E8%AF%81%E5%88%B8%E6%95%99%E7%A8%8B-Chapter3-Black-Scholes/)
数字期权和股份数字期权是构成看涨期权和看跌期权的基本元素。

## 数字期权

+ $S(T)>K$则支付$1的情况

$$
x=\begin{cases} 1\quad \text{if }S(T)>K \\0 \quad \text{else}\end{cases}
$$

由风险中性定价公式（1.18）可知，数字期权在时刻0的价值为$e^{-rT}\mathbb{E}^R[x]$。由于
$$
\begin{align*}
\mathbb{E}^R[x]&=1 \times \text{prob}^R(x=1)+0 \times \text{prob}^R(x=0)\\
&=\text{prob}^R(x=1)\\
&=\text{prob}^R(S(T)>K)
\end{align*}
$$
由（2.27），以及Black-Scholes假设$\frac{dS}{S}=\mu dt + \sigma dB$下，有
$$
\frac{dS}{S}=(r-q)dt+\sigma dB^*
$$
等价于
$$
d\log S=(r-q-\frac{1}{2}\sigma^2)dt+\sigma dB^*
$$
采用（2.34）-（2.35），并取其中的$\alpha=r-q-\sigma^2/2$，得出
$$
\text{prob}^R(S(T)>K)=N(d_2)\\
\text{其中： }d_2=\frac{\log(\frac{S(0)}{K})+(r-q-\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}} \tag{3.2}
$$
从而：

> $S(T)>K$时支付$1的数字期权的价值为$$e^{-rT}N(d_2)$，其中$d_2$由（3.2）给出。

+ $S(T)<K$则支付$1的情况

$$
y=\begin{cases} 1\quad \text{if }S(T)<K \\0 \quad \text{else}\end{cases}
$$

再次应用风险中性定价公式，得出数字期权在0时的价值为
$$
e^{-rT}\mathbb{E}^R[y]=e^{-rT}\text{prob}^R(y=1)=e^{-rT}\text{prob}^R(S(T)<K)
$$
由（2.36）可得

> $S(T)<K$时支付$1的数字期权的价值为$$e^{-rT}N(-d_2)$，其中$d_2$由（3.2）给出。

## 股份数字期权

+ $S(T)>K$则支付一份标的资产的情况

$$
x=\begin{cases} 1\quad \text{if }S(T)>K \\0 \quad \text{else}\end{cases}
$$

则股份数字期权在时间T的回报为$xS(T)$。设$Y(t)$表示这个未定权益的价格，$0 \leq t \leq T$，则$Y(T)=xS(T)$，现计算$Y(0)$。

以$V(t)=e^{qt}S(t)$作为计价物，从基本定价公式（1.17）得出。
$$
Y(0)=S(0)\mathbb{E}^V[\frac{Y(T)}{e^{qT}S(T)}]=e^{-qT}S(0)\mathbb{E}^V[x]=e^{-qT}S(0)\text{prob}^T(x=1)
$$
由（2.28），得出
$$
\frac{dS}{S}=(r-q+\sigma^2)dt+\sigma dB^*
$$
等价于
$$
d\log S=(r-q+\frac{1}{2}\sigma^2)dt+\sigma dB^* \tag{3.3}
$$
采用（2.34）-（2.35），并取其中的$\alpha=r-q+\sigma^2/2$，得出
$$
\text{prob}^V(S(T)>K)=N(d_1)\\
\text{其中： }d_1=\frac{\log(\frac{S(0)}{K})+(r-q+\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}} \tag{3.4}
$$
从而：

> $S(T)>K$时支付一份标的资产的股份数字期权的价值为$e^{-qT}S(0)N(d_1)$，其中$d_1$由（3.4）给出。

+ $S(T)<K$则支付一份标的资产的情况

$$
y=\begin{cases} 1\quad \text{if }S(T)<K \\0 \quad \text{else}\end{cases}
$$

则股份数字期权在时间T的回报为$yS(T)$，得出数字股份期权在0时的价值为
$$
e^{-qT}S(0)\mathbb{E}^V[y]=e^{-qT}S(0)\text{prob}^V(y=1)=e^{-qT}S(0)\text{prob}^V(S(T)<K)
$$
由（2.36）可得

> $S(T)<K$时支付一份标的资产的股份数字期权的价值为$e^{-qT}S(0)N(-d_1)$，其中$d_1$由（3.4）给出。

## 看跌期权和看涨期权

+ 欧式看涨期权

在时间T，如果$S(T)>K$，一份欧式看涨期权的回报为$S(T)-K$，否则为0.令
$$
x=\begin{cases}
1\quad \text{if } S(T)>K\\ 0 \quad \text{else}
\end{cases}
$$
看涨期权的回报函数可以表示为$xS(T)-xK$，等价于一份股份数字期权减去K份数字期权，该数字期权在事件$S(T)>K$发生时进行支付。股份数字期权的0时价值为$e^{-qT}S(0)N(d_1)$，数字期权的价值为$e^{-rT}N(d_2)$。从（3.2）和（3.4）可以得出，$d_1$和$d_2$具有关系$d_2=d_1-\sigma\sqrt{T}$,综上可得出Black-Scholes公式：

欧式看涨期权在0时的价值为
$$
e^{-qT}S(0)N(d_1)-e^{-rT}KN(d_2) \tag{3.5}
$$

+ 欧式看跌期权

  欧式看跌期权在T时的回报为：如果$S(T)<K$，回报为$K-S(T)$，否则为0。设
  $$
  y=\begin{cases} 1 \quad \text{if } S(T)<K\\0 \quad \text{else}\end{cases}
  $$
  看跌期权的回报函数为$yK-yS(T)$，这等价于K份数字期权减去一份数字股份期权，两个期权在$S(T)<K$时进行支付。由此得出

  欧式看跌期权在0时的价值为
  $$
  e^{-rT}KN(-d_2)-e^{-qT}S(0)N(-d_1) \tag{3.6}
  $$

+ 欧式期权看跌看涨平均关系（CCP）
  $$
  e^{-rT}K+看涨期权价格=e^{-qT}S(0)+看跌期权价格
  $$

  ## 希腊字母

  |   变量   | Input Symbol | 希腊字母 | 希腊字母符号 |
  | :------: | :----------: | :------: | :----------: |
  | 股票价格 |      S       |  delta   |      δ       |
  |  德尔塔  |      δ       |  gamma   |      Γ       |
  | - 到期日 |      -T      |  theta   |      Θ       |
  |  波动率  |      σ       |   vega   |      V       |
  |   利率   |      r       |   rho    |      ρ       |

  其中希腊字母Θ是关于-T的导数而不是关于T的导数，原因在于到期时间T的减少（-T的增加）等价于时间的流逝，因此Θ衡量的是时间流逝对期权价格的影响。

  正态分布函数N的导数是正态密度函数n，即
  $$
  n(d)=\frac{1}{\sqrt{2\pi}}e^{-\frac{-d^2}{2}}
  $$
  容易验证
  $$
  e^{-qT}Sn(d_1)=e^{-rT}Kn(d_2) \tag{3.8}
  $$
  以欧式看涨期权为例计算希腊字母

  $$ \begin{align*} \delta &= \mathrm{e}^{-qT} \mathrm{N}(d_1) + \mathrm{e}^{-qT} S \mathrm{n}(d_1) \frac{\partial d_1}{\partial S} - \mathrm{e}^{-rT} K \mathrm{n}(d_2) \frac{\partial d_2}{\partial S} \\ &= \mathrm{e}^{-qT} \mathrm{N}(d_1) + \mathrm{e}^{-qT} S \mathrm{n}(d_1) \left( \frac{\partial d_1}{\partial S} - \frac{\partial d_2}{\partial S} \right) \\ &= \mathrm{e}^{-qT} \mathrm{N}(d_1), \end{align*} $$

  $$ \Gamma = \mathrm{e}^{-qT} \mathrm{n}(d_1) \frac{\partial d_1}{\partial S} = \mathrm{e}^{-qT} \mathrm{n}(d_1) \frac{1}{S\sigma\sqrt{T}}, $$

  $$ \begin{align*} \Theta &= -\mathrm{e}^{-qT} S \mathrm{n}(d_1) \frac{\partial d_1}{\partial T} + q\mathrm{e}^{-qT} S \mathrm{N}(d_1) \\ &\quad + \mathrm{e}^{-rT} K \mathrm{n}(d_2) \frac{\partial d_2}{\partial T} - r\mathrm{e}^{-rT} K \mathrm{N}(d_2) \\ &= \mathrm{e}^{-qT} S \mathrm{n}(d_1) \left( \frac{\partial d_2}{\partial T} - \frac{\partial d_1}{\partial T} \right) \\ &\quad + q\mathrm{e}^{-qT} S \mathrm{N}(d_1) - r\mathrm{e}^{-rT} K \mathrm{N}(d_2) \\ &= -\mathrm{e}^{-qT} S \mathrm{n}(d_1) \frac{\sigma}{2\sqrt{T}} + q\mathrm{e}^{-qT} S \mathrm{N}(d_1) - r\mathrm{e}^{-rT} K \mathrm{N}(d_2), \end{align*} $$

  $$ \begin{align*} \rho &= \mathrm{e}^{-qT} S \mathrm{n}(d_1) \frac{\partial d_1}{\partial r} - \mathrm{e}^{-rT} K \mathrm{n}(d_2) \frac{\partial d_2}{\partial r} + T\mathrm{e}^{-rT} K \mathrm{N}(d_2) \\ &= \mathrm{e}^{-qT} S \mathrm{n}(d_1) \left( \frac{\partial d_1}{\partial r} - \frac{\partial d_2}{\partial r} \right) + T\mathrm{e}^{-rT} K \mathrm{N}(d_2) \\ &= T\mathrm{e}^{-rT} K \mathrm{N}(d_2). \end{align*} $$



根据看涨-看跌平价公式可以计算出看跌期权的希腊字母。

## 德尔塔对冲

无套利定价得出BS公式的关键，在于用股票和期权构造一个完全对冲（无风险）投资组合。为实现完全对冲，必须随时对投资组合进行调整，因为$\delta$值会随标的资产变化和时间的推移而变化。即使所有模型假设都满足，实际中的对冲也不可能是完全对冲。

考虑到期日为T的一份欧式看涨期权，用$C(S,t)$表示时间t处股票价格为S的期权价格。考虑由一份看涨期权空头和$\delta$份标的资产多头形成的投资组合。同时考虑由$C-\delta S$的现金（空头）组成的投资组合，该投资组合在0时的价值为0。

投资组合价值在时间间隔dt的瞬间改变量为
$$
-dC+\delta dS + q\delta S dt+(C-\delta S)r dt \tag{3.9}
$$
第一项反映了期权自身价值的变化，第二项是持有δ份股票所产生的资本利得或损失，第三项是这δ份股票带来的股息收入，而第四项则是空头现金头寸所产生的利息费用。

另一方面，从伊藤公式得出
$$
\begin{align}
dC=&\frac{\partial{C}}{\partial{S}}dS+\frac{\partial{C}}{\partial{t}}dt+\frac{1}{2}\frac{\partial^2{C}}{\partial{S^2}}(dS)^2\\
&=\delta dS +\Theta dt+\frac{1}{2}\Gamma \sigma^2S^2 dt
\end{align}\tag{3.10}
$$
将（3.10）带入(3.9)，得出投资组合价值的变化为
$$
-\Theta \, \mathrm{d}t - \frac{1}{2}\Gamma \sigma^2 S^2 \, \mathrm{d}t + q\delta S \, \mathrm{d}t + (C - \delta S)r \, \mathrm{d}t. \tag{3.11}
$$
德尔塔对冲消除了标的资产价格变化带来的风险暴露，所以（3.11）没有dS项。其次，$\Theta$时负的，它衡量了期权时间价值的减少；期权空头将从到期时间的减少中获利，时间每减少一个单位所带来的获利为$-\Theta$。这样的投资组合时“伽玛空头”组合，也称为“凸性空头”投资组合。股票价格的波动使得凸性具有价值，凸性空头的投资组合将遭受损失。最后，投资组合得到红利，但支出利息。

带入各希腊字母的定义，可知，（3.11）中各项之和为0.时间减少导致的期权价值变化和来自标的资产的红利收入，正好可以对冲掉凸性损失和利息支出。因此，<u>德尔塔对冲是完全对冲的</u>。

## 伽玛对冲

在离散的德尔塔对冲中，为了改善对冲效果，可以用另外的期权构建德尔塔中性和伽玛中性的投资组合。

伽玛中性的含义是，投资组合的德尔塔不存在关于标的资产价格变化的风险暴露，这等价于投资组合价值关于标的资产价格的二阶导数等于0。如果德尔塔真的不发生变化，则没有必要对投资组合进行连续调整，但是，并不能保证离散调整的德尔塔/伽玛对冲比离散调整的德尔塔对冲好。

> **金融工程中没有“免费的午餐”**。
>
> 不能保证的几个原因：
>
> a) **高阶风险暴露**
>
> Delta是价值对价格的一阶导数，Gamma是二阶导数。但风险并没有在三阶停止。还有**三阶导数**，通常称为**Speed**或 **Gamma的导数**，它衡量Gamma本身的变化速度。当一个大幅价格变动发生时，不仅Delta会变，Gamma本身也会改变。一个在初始时刻Delta和Gamma中性的组合，可能会因为Gamma的改变而立刻产生新的Delta和Gamma风险。要管理这个风险，就需要引入第三种资产，使得组合对三阶导也中性，但这会极大地增加复杂性、成本和模型风险。
>
> b) **交易成本和复杂性**
>
> 构建Gamma中性组合需要引入**第二种期权**。这意味着：
>
> - **更高的交易成本**：需要多交易一种流动性可能较差的金融工具。
>
> - **更高的管理复杂度**：你需要同时监控和管理两个期权头寸的Delta、Gamma、Vega（波动率风险）等。
>
>   这些额外的成本和复杂性可能会“吃掉”Gamma中性可能带来的任何理论优势。有时，简单地更频繁地调整Delta对冲（即进行离散的Delta再平衡），可能比维持一个复杂的Delta/Gamma中性组合更经济、更有效。
>
> c) **模型风险**
>
> 无论是计算Delta还是Gamma，都严重依赖于所使用的期权定价模型（如布莱克-斯科尔斯模型）。这些模型基于一系列假设（如波动率恒定、市场连续等），而这些假设在现实中并不完全成立。如果模型本身有缺陷，那么基于模型计算出的“中性”头寸从一开始就是错的。一个基于错误模型构建的、看似更复杂的对冲策略，其表现可能还不如一个简单的策略。
>
> d) **其他风险暴露**
>
> Delta/Gamma对冲主要针对“方向性风险”。但投资组合还暴露在其他风险之下，最典型的是**Vega风险**，即对波动率的敏感性。在构建Delta/Gamma中性组合时，你很可能无意中改变了组合对波动率的暴露。如果市场波动率发生意外变化，这个对冲组合的表现可能会出乎意料地差。
>
> 总结：
>
> - **Delta对冲**：简单，但需要频繁再平衡，对离散调整敏感。
> - **Delta/Gamma对冲**：试图减少再平衡频率和对离散调整的敏感度，但引入了**更高阶的风险、更高的成本、更复杂的模型依赖以及新的风险暴露（如Vega）**。
>
> 因此，**“没有保证”**意味着，在现实世界中，第二种策略所引入的新问题和成本，完全有可能超过它所能解决的问题。最终哪种策略表现更好，取决于具体市场环境（价格是平滑变动还是跳跃式变动）、交易成本、以及模型准确性等多种因素，其结果是不确定的。

构造德尔塔/伽玛对冲组合：

假定卖出一份看涨期权，并打算用标的资产和零一份期权进行德尔塔对冲和伽玛对冲，比如采用零一分执行价格不同的看涨期权进行对冲。实践中，需要用一个流动性期权来达到这个目的。==所谓流动性期权是指期权执行价格接近标的资产当前价格的期权（即用于对冲的期权是近似平价期权）。==

设$\delta$和$\Gamma$表示卖出期权的德尔塔和伽玛，$\delta$'和$\Gamma$' 表示对冲期权的德尔塔和伽玛。考虑用a份股票和b份期权对卖出期权实施对冲。股票的德尔塔等于1（dS/dS=1),因此要得到0德尔塔投资组合，需要满足：
$$
0=-\delta+a+b\delta ' \tag{3.12}
$$
股票的伽玛为0（$d^2S/dS^2=d1/dS=0$)，因此要得到0伽玛的投资组合，需要满足：
$$
0=-\Gamma+b\Gamma ' \tag{3.13}
$$
（3.13）表明应该持有足够的用于对冲的第二种期权，使得卖出的期权空头具有伽玛中性，即：
$$
b=\frac{\Gamma}{\Gamma'}
$$
从而得到：
$$
a=\delta-\frac{\Gamma}{\Gamma'}\delta '
$$

## 隐含波动率

BS公式给出了期权价格和波动率之间的一一对应关系，因此可以在价格给定时从公式中推出$\sigma$，用这种方法计算出来的$\sigma$称为“隐含波动率”。从一个期权中得出的隐含波动率可用于另一个期权（也许是没有交易或交易很不活跃的期权）定价。

甚至在我们承认模型是不准确的情况下，隐含波动率的计算对于了解市场价格仍然是有用处的，因为**根据波动率的大小，可以迅速判断一份期权是“贵”还是“便宜”**。不能从期权的价格判断期权是贵还是便宜，因为期权的价格必须由执行价格和到期时间综合考虑。==一定程度上，隐含波动率用执行价格和到期时间将期权价格标准化==。

## 波动率期限结构

当实际当中的波动率不是常数，是一个非随机的时间函数时，期权定价公式仍然有效，此时$\log S(T)$的方差是：
$$
\int^T_0 \sigma^2(t)dt \tag{3.14}
$$
这个积分实际上是瞬间方差$\sigma^2(t)dt$的和。用（3.14）代替期权定价公式中$d_1$和$d_2$中的$\sigma^2 T$得出波动率不是常数时的期权定价公式。

下面给出更方便的表达式，设$\sigma_{avg}$为正数，使得
$$
\sigma^2_{avg}=\frac{1}{T}\int ^T_0\sigma^2(t)dt \tag{3.15}
$$
这样，只需要将$\sigma_{avg}$作为期权定价函数输入变量中的sigma就可以了。将$\sigma_{avg}$称为==“平均波动率”==。注意，实际上，$\sigma_{avg}$不是$\sigma(t)$的平均值，而是$\sigma^2(t)$平均值的平方根。其是期权所剩存活时间内的（在一定意义上的）平均波动率，而未必与期权整个存活期的平均波动率相同。定价和对冲中用到的是剩余存活期内的平均波动率。此外，在时间0（表示给期权定价的时间，并不一定是首次出售或者首次购买期权的时间，这很重要！）计算期权价格和实施对冲时，应该采用$\sigma_{avg}$。

用不同到期日期权计算隐含波动率时，通常会得出不同的结果。例如，考虑到期日分别为$T_1$和$T_2(T_2>T_1)$的两个平价期权，对应的隐含波动率分别用$\hat{\sigma_1}$和$\hat{\sigma_2}$表示。我们将$\hat{\sigma_1}$和$\hat{\sigma_2}$解释为时间区间$[0,T_1]$和$[0,T_2]$上的平均波动率，这说明存在函数$\sigma(t)$使得
$$
\hat{\sigma}_1^2=\frac{1}{T_1}\int ^{T_1}_0\sigma^2(t)dt\quad \hat{\sigma}_2^2=\frac{1}{T_2}\int ^{T_2}_0\sigma^2(t)dt
$$
由此得出
$$
\hat{\sigma}_2^2T_2-\hat{\sigma}_1^2T_1=\int^{T_2}_{T_1}\sigma^2(t)dt
$$
而这又要求
$$
\hat{\sigma}_2^2T_2-\hat{\sigma}_1^2T_1 \geq 0
$$
或等价地
$$
\hat{\sigma}_2 \geq \sqrt{\frac{T_1}{T_2}}\hat{\sigma}_1
$$
如果不等式成立，可以构造函数$\sigma(t)$
$$
\sigma(t)=\begin{cases}\hat{\sigma}_1 \quad \text{for }t \leq T_1\\\sqrt{\frac{\hat{\sigma}_2^2T_2-\hat{\sigma}_1^2T_1}{T_2-T_1}}\quad \text{for }T_1 < t \leq T_2 \end{cases}
$$
更一般地，给定到期日分别为$T_1<T_2<\cdots<T_N$的N个平价期权，对应的隐含波动率为$\hat{\sigma}_1,\hat{\sigma}_2,\cdots ,\hat{\sigma}_N$，对$T_i<t\leq T_{i+1}$时，定义
$$
\sigma(t)=\sqrt{\frac{\hat{\sigma}_{i+1}^2T_{i+1}-\hat{\sigma}_i^2T_i}{T_{i+1}-T_i}}
$$
这样的$\sigma(t)$常常称为“（隐含）波动率的期限结构”

通常情况下，如果当前市场剧烈波动，$\sigma(t)$应该是t的递减函数，而如果当前市场十分平稳，$\sigma(t)$应该是t的递增函数。

## 微笑现象

如果我们计算到期期限相同但行权价不同的期权的隐含波动率，会发现不同期权的隐含波动率依然存在差异。将隐含波动率与行权价绘制成图后，股票及股票指数期权通常呈现这样的规律：隐含波动率随行权价上升而下降，直至行权价接近标的资产当前价格（此时期权为平价期权）；之后，在更高行权价区间，隐含波动率一般会趋于平稳或小幅上升。该图形形似 “扭曲的微笑（偏斜微笑）”。

与隐含波动率的期限结构不同，这种 “实值程度相关的隐含波动率结构” 与模型存在直接矛盾。它表明，风险中性回报分布并非对数正态分布，而是比对数正态分布更可能出现极端回报（即具有 “厚尾” 特征），且极端负回报的发生概率高于极端正回报（即具有 “偏斜” 特征）

## 计算看涨看跌期权价格、德尔塔、伽玛和隐含波动率

```rust
pub mod European{
    //从 statrs::distribution 导入 Normal 分布和 Continuous 特质。Continuous 特质定义了 cdf 方法
    use statrs::distribution::{Normal, Continuous, ContinuousCDF};

    /// Black-Scholes看涨期权定价模型
    ///
    /// # 参数
    /// - S: 标的资产当前价格
    /// - K: 期权行权价
    /// - r: 无风险利率（年化）
    /// - sigma: 标的资产波动率（年化）
    /// - q: 股息收益率（年化）
    /// - T: 到期时间（年）
    ///
    /// # 返回值
    /// 看涨期权的理论价格
    ///
    /// # 公式
    /// C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
    /// d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    /// d2 = d1 - σ√T
    pub fn European_Call(S:f64,K:f64,r:f64,sigma:f64,q:f64,T:f64)->f64{
        //处理边界情况
        if T<=0.0{
            // 到期时期权价值为内在价值
            return (S-K).max(0.0);
        }

        if sigma == 0.0{
            // 波动率为0时的特殊情况
            let instrinsic_value=(S*(-q*T).exp()-K*(-r*T).exp()).max(0.0);
            return instrinsic_value;
        }

        let sqrt_t=T.sqrt();
        let d1=(S.ln()-K.ln()+(r-q+0.5*sigma.powi(2))*T)/(sigma*sqrt_t);
        let d2=d1-sigma*sqrt_t;

        //Normal::new(mean, std_dev) 创建一个正态分布实例。
        // 由于参数可能无效（如标准差为负），它返回一个 Result，我们用 unwrap() 来获取成功的结果
        // （在生产代码中，你应该更谨慎地处理 Result）
        let standard_norm=Normal::new(0.0,1.0).unwrap();
        //调用 standard_normal.cdf(x) 即可计算出 P (X ≤ x) 的值
        let n_d1=standard_norm.cdf(d1);
        let n_d2=standard_norm.cdf(d2);

        S*(-q*T).exp()*n_d1-K*(-r*T).exp()*n_d2
    }

    /// Black-Scholes看跌期权定价模型
    ///
    /// # 参数
    /// - s: 标的资产当前价格
    /// - k: 期权行权价
    /// - r: 无风险利率（年化）
    /// - sigma: 标的资产波动率（年化）
    /// - q: 股息收益率（年化）
    /// - t: 到期时间（年）
    ///
    /// # 返回值
    /// 看跌期权的理论价格
    ///
    /// # 公式
    /// P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
    /// d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    /// d2 = d1 - σ√T
    pub fn European_Put(S:f64,K:f64,r:f64,sigma:f64,q:f64,T:f64)->f64{
        if T<=0.0{
            return (K-S).max(0.0);
        }
        if sigma == 0.0{
            let instrinsic_value=K*(-r*T).exp()-S*(-q*T).exp();
            return instrinsic_value;
        }
        let sqrt_t=T.sqrt();
        let d1=(S.ln()-K.ln()+(r-q+0.5*sigma.powi(2))*T)/(sigma*sqrt_t);
        let d2=d1-sigma*sqrt_t;

        //Normal::new(mean, std_dev) 创建一个正态分布实例。
        // 由于参数可能无效（如标准差为负），它返回一个 Result，我们用 unwrap() 来获取成功的结果
        // （在生产代码中，你应该更谨慎地处理 Result）
        let standard_norm=Normal::new(0.0,1.0).unwrap();
        //调用 standard_normal.cdf(x) 即可计算出 P (X ≤ x) 的值
        let n_nd1=standard_norm.cdf(-d1);
        let n_nd2=standard_norm.cdf(-d2);

        K*(-r*T).exp()*n_nd2-S*(-q*T).exp()*n_nd1
    }

    pub fn Black_Scholes_Call_Delta(S:f64,K:f64,r:f64,sigma:f64,q:f64,T:f64)->f64{
        /// Black-Scholes看涨期权Delta计算
        ///
        /// # 参数
        /// - s: 标的资产当前价格
        /// - k: 期权行权价
        /// - r: 无风险利率（年化）
        /// - sigma: 标的资产波动率（年化）
        /// - q: 股息收益率（年化）
        /// - t: 到期时间（年）
        ///
        /// # 返回值
        /// 看涨期权的Delta值，范围在[0, 1]之间
        ///
        /// # 公式
        /// Δ = e^(-qT) * N(d1)
        /// d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)

        if T<=0.0{
            return if S>K {1.0}else{0.0}
        }
        if sigma == 0.0{
            return if S*(-q*T).exp()>K*(-r*T).exp(){
                (-q*T).exp()
            }else{
                0.0
            };
        }
        let sqrt_t=T.sqrt();
        let d1=(S.ln()-K.ln()+(r-q+0.5*sigma.powi(2))*T)/(sigma*sqrt_t);

        let standard_norm=Normal::new(0.0,1.0).unwrap();
        let n_d1=standard_norm.cdf(d1);
        (-q*T).exp()*n_d1
    }

    /// Black-Scholes看涨期权Gamma计算
    ///
    /// # 参数
    /// - S: 标的资产当前价格
    /// - K: 期权行权价
    /// - r: 无风险利率（年化）
    /// - sigma: 标的资产波动率（年化）
    /// - q: 股息收益率（年化）
    /// - T: 到期时间（年）
    ///
    /// # 返回值
    /// 看涨期权的Gamma值
    ///
    /// # 公式
    /// Γ = e^(-qT) * N'(d1) / (S * σ * √T)
    /// 其中 N'(d1) = e^(-d1²/2) / √(2π)
    /// d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    ///
    /// # 金融意义
    /// Gamma衡量Delta对标的资产价格变化的敏感性，是期权凸性的度量
    pub fn Black_Scholes_Call_Gamma(S:f64,K:f64,r:f64,sigma:f64,q:f64,T:f64)->f64{
        // 处理边界情况
        if T <= 0.0 || sigma <= 0.0 || S <= 0.0 {
            return 0.0;
        }

        // 计算d1
        let sqrt_t = T.sqrt();
        let d1 = (S.ln() - K.ln() + (r - q + 0.5 * sigma.powi(2)) * T) / (sigma * sqrt_t);

        // 计算标准正态分布的概率密度函数值 N'(d1)
        let standard_norm=Normal::new(0.0,1.0).unwrap();
        let nd1 = standard_norm.pdf(d1);

        // Black-Scholes看涨期权Gamma公式
        (-q * T).exp() * nd1 / (S * sigma * sqrt_t)
    }

    /// 计算Black-Scholes看涨期权的隐含波动率
    /// 参数:
    /// - S: 初始股票价格
    /// - K: 行权价格
    /// - r: 无风险利率
    /// - q: 股息收益率
    /// - T: 到期时间（年）
    /// - CallPrice: 看涨期权市场价格
    ///
    /// 返回: Result<f64, String> - 成功时返回隐含波动率，错误时返回错误信息
    pub fn Black_Scholes_Call_implied_Vol(S:f64,K:f64,r:f64,q:f64,T:f64,CallPrice:f64)->Result<f64,String> {
        if CallPrice<S*(-q*T).exp()-K*(-r*T).exp(){
            return Err("Option price violates the arbitrage bound.".to_string());
        }

        let tol=1e-6;
        let mut lower=0.0;
        let mut upper=1.0;

        let mut flower=European_Call(S,K,r,lower,q,T)-CallPrice;
        let mut fupper:f64=European_Call(S,K,r,upper,q,T)-CallPrice;

        while fupper<0.0{
            upper*=2.0;
            fupper=European_Call(S,K,r,upper,q,T)-CallPrice;

            // 防止无限循环
            if upper > 100.0 {
                return Err("Unable to find valid upper bound for volatility.".to_string());
            }
        }

        //二分法求解
        let mut guess=(upper+lower)/2.0;
        let mut fguess:f64=European_Call(S,K,r,guess,q,T)-CallPrice;

        let max_iter=1000;
        let mut iter=0;

        while (upper-lower)>tol && iter<max_iter{
            if fupper*fguess<0.0{
                lower=guess;
            }else{
                upper=guess;
            }

            guess=(upper+lower)/2.0;
            fguess=European_Call(S,K,r,guess,q,T)-CallPrice;
            iter+=1;
        }
        if iter>max_iter{
            return Err("Failed to converge within maxminum iterration.".to_string());
        }
        Ok(guess)
    }
}

mod calc;

fn main() {
    let S:f64=50.0;
    let K:f64=40.0;
    let r:f64=0.05;
    let sigma:f64=0.3;
    let q:f64=0.02;
    let T:f64=2.0;
    let call_price:f64=20.0;

    let call=calc::European::European_Call(S,K,r,sigma,q,T);
    println!("European call price: {}",call);

    let put:f64=calc::European::European_Put(S,K,r,sigma,q,T);
    println!("European put price: {}",put);

    let call_delta=calc::European::Black_Scholes_Call_Delta(S,K,r,sigma,q,T);
    println!("European call delta: {}",call_delta);

    let call_gamma:f64=calc::European::Black_Scholes_Call_Gamma(S,K,r,sigma,q,T);
    println!("European call gamma: {}",call_gamma);

    let call_impilied_vol:f64=calc::European::Black_Scholes_Call_implied_Vol(S,K,r,q,T,call_price).expect("some mistake");
    println!("European call impilied vol: {}",call_impilied_vol);

}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    #[test]
    fn Valid_European_Call(){
        let S:f64=50.0;
        let K:f64=40.0;
        let r:f64=0.05;
        let sigma:f64=0.3;
        let q:f64=0.02;
        let T:f64=2.0;

        let call=calc::European::European_Call(S,K,r,sigma,q,T);
        let call=format!("{:.5}",call);
        let call:f64=call.parse::<f64>().unwrap();
        assert_eq!(call,14.48306);

    }

    #[test]
    fn Valid_European_Put(){
        let S:f64=50.0;
        let K:f64=40.0;
        let r:f64=0.05;
        let sigma:f64=0.3;
        let q:f64=0.02;
        let T:f64=2.0;

        let put=calc::European::European_Put(S,K,r,sigma,q,T);
        let put=format!("{:.6}",put);
        let put:f64=put.parse::<f64>().unwrap();
        assert_eq!(put,2.637087);
    }

    #[test]
    fn Valid_BS_Call_Delta(){
        let S:f64=50.0;
        let K:f64=40.0;
        let r:f64=0.05;
        let sigma:f64=0.3;
        let q:f64=0.02;
        let T:f64=2.0;

        let delta:f64=calc::European::Black_Scholes_Call_Delta(S,K,r,sigma,q,T);
        let delta=format!("{:.6}",delta);
        let delta:f64=delta.parse::<f64>().unwrap();
        assert_eq!(delta,0.778659)
    }

    #[test]
    fn Valid_BS_Call_Gamma(){
        let S:f64=50.0;
        let K:f64=40.0;
        let r:f64=0.05;
        let sigma:f64=0.3;
        let q:f64=0.02;
        let T:f64=2.0;

        let gamma:f64=calc::European::Black_Scholes_Call_Gamma(S,K,r,sigma,q,T);
        let gamma=format!("{:.6}",gamma);
        let gamma:f64=gamma.parse::<f64>().unwrap();
        assert_eq!(gamma,0.012273);
    }

    #[test]
    fn Valid_BS_Call_ImpliedBlackVolatility(){
        let S:f64=50.0;
        let K:f64=40.0;
        let r:f64=0.05;
        let q:f64=0.02;
        let T:f64=2.0;
        let call_price=20.0;

        let impliedVol=calc::European::Black_Scholes_Call_implied_Vol(S,K,r,q,T,call_price).expect("some mistake");
        let impliedVol=format!("{:.6}",impliedVol);
        let impliedVol:f64=impliedVol.parse::<f64>().unwrap();
        assert_eq!(impliedVol,0.576602);
    }
}
```