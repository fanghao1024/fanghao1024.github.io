[toc]

# 基本思想

最大熵强化学习的思想除了要最大化累积奖励，还要使得策略更加随机。而熵正则化增加了强化学习算法的探索程度，$\alpha$越大，探索性就越强，能让策略尽可能随机，Agent可以更充分地探索状态空间S，避免策略早早落入局部最优点，并且可以探索到多个可行的方案完成指定任务，提高抗干扰能力。

SAC算法优势：基于能量的模型在面对多模态（multimodal）的值函数（ Q(s,a) ）时，具有更强的策略表达能力，而一般的高斯分布只能将决策集中在 Q 值更高的部分，忽略其他次优解。

# 最大熵学习（MERL）

熵
$$
H(p)=E_{x \sim p}[-\log P(x)] \tag{1}
$$
标准强化学习算法的目标
$$
\pi_{std}^{*}=arg \max_{\pi} \sum_tE_{(s_t,a_t)\sim \rho_{\pi}}[r(s_t,a_t)] \longrightarrow \text{找到能收集最多累计收益的策略} \tag{2}
$$
引入熵最大化的RL算法的目标
$$
\pi_{maxEntropy}^{*}=arg \max_{\pi} \sum_tE_{(s_t,a_t)\sim \rho_{\pi}}[r(s_t,a_t)+\alpha H(\pi(\cdot|s_t))]  \longrightarrow 从概率图模型推出，可参考SVI
$$
思想来源于最大熵方法，好处：模型在匹配观察到的信息时，对未知的假设最少

# soft value function & Energy based policy

## 原本的RL值函数

$$
Q^\pi (s,a)=E_{(s_t,a_t) \sim\rho_\pi} \left[ \sum_{t=0}^{\infty}\gamma^t\cdot r(s_t,a_t)|s_0=s,a_0=a \right]\\
V^\pi (s)=E_{(s_t,a_t) \sim\rho_\pi} \left[ \sum_{t=0}^{\infty}\gamma^t\cdot r(s_t,a_t)|s_0=s \right] \tag{3}
$$

## 根据目标函数（3），得到Soft value function(SVF)

$$
\text{soft Q function: } Q_{soft}^\pi (s,a)=E_{(s_t,a_t) \sim\rho_\pi} \left[ \sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)+\alpha\sum_{t=1}^{\infty}\gamma^t H(\pi(\cdot|s_t))|s_0=s,a_0=a \right]  \tag{4}
$$

$$
\text{soft V function: } V_{soft}^\pi (s)=E_{(s_t,a_t) \sim\rho_\pi} \left[ \sum_{t=0}^{\infty}(\gamma^t\cdot r(s_t,a_t)+\alpha H(\pi(\cdot|s_t)))|s_0=s \right] \tag{5}
$$

## soft Q function和soft V function的关系：

$$
\begin{aligned}
Q_{soft}^\pi (s,a)&=E_{(s_t,a_t) \sim\rho_\pi} \left[ \sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)+\alpha\sum_{t=1}^{\infty}\gamma^t H(\pi(\cdot|s_t))|s_0=s,a_0=a \right]\\
&=E_{(s_t,a_t) \sim\rho_\pi} \left[ r(s_0,a_0)+\sum_{t=1}^{\infty}\gamma^t\left(r(s_t,a_t)+\alpha H(\pi(\cdot|s_t))\right) |s_0=s,a_0=a\right]\\
&=E_{(s_t,a_t) \sim\rho_\pi} \left[r(s_o,a_0)+\gamma\cdot V_{soft}^\pi(s_{t+1})|s_0=s,a_0=a \right] 
\end{aligned}\tag{6}
$$

$$
\begin{aligned}
V_{soft}^\pi (s)&=E_{(s_t,a_t) \sim\rho_\pi} \left[ \sum_{t=0}^{\infty}(\gamma^t\cdot r(s_t,a_t)+\alpha H(\pi(\cdot|s_t)))|s_0=s \right]\\
&=E_{(s_t,a_t) \sim\rho_\pi} \left[ \sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)+\alpha\sum_{t=1}^{\infty}\gamma^t H(\pi(\cdot|s_t)+\alpha H(\pi(\cdot | s_0)))|s_0=s,a_0=a \right]\\
&=E_{a_t \sim\pi} \left[Q_{soft}^{\pi}(s_t,a_t)+\alpha\cdot H(\pi(\cdot|s_0))|s_0=s \right]\\
\end{aligned} \tag{7}
$$

## Energy Based Policy (EBP，基于能量的策略模型)

为了适用更复杂的任务，MERL不再是以往的高斯分布形式，使用EBP表示策略：
$$
\pi(a_t|s_t) \propto \exp{(-\xi(s_t,a_t))} \longrightarrow \xi是能量函数 \tag{8}
$$
为了让EBP与值函数联系起来，设定
$$
\xi(s_t,a_t)=-\frac{1}{\alpha}Q_{soft}(s_t,a_t) \tag{9}
$$
从而有
$$
\pi(a_t|s_t) \propto \exp{(\frac{1}{\alpha}Q_{soft}(s_t,a_t))} \tag{10}
$$

# soft Q-learning 中的策略评估和策略优化

## 策略评估

SAC中使用（6）（7）进行值迭代

soft Q-learning使用不同的值迭代公式
$$
V_{soft}(s)=\alpha \cdot\log{\int\exp \left(\frac{1}{\alpha}Q_{soft}(s,a)\right)da} \tag{11}
$$

> $\log{\int\exp}$相当于softmax，也是soft一词的由来，所以式（11）在离散情况下等价于$V(s)=\alpha\cdot softmax \frac{1}{\alpha}Q(s,a)$
>
> 解释：
>
> ​	记$x_{max}=\max\{x_1,x_2,\cdots ,x_n\}$，则$e^{x_{max}} <\sum_{i=1}^n e^{x_i} \le\sum_{i=1}^ne^{x_{max}}=n\cdot e^{x_{max}}$
>
> ​	对以上不等式两边取$\log$，得$x_{max}<\log{\sum_{i=1}^n}e^{x_i} \le x_max+\log{n}$	
>
> ​	误差$\log{n}$与x无关，故
> $$
> x_{max}/\tau < \log{\sum_{i=1}^ne^{x_i/\tau}}\le x_{max}/\tau \Rightarrow x_{max} < \tau\cdot \log{\sum_{i=1}^n}e^{x_i/\tau} \le x_{max}+\tau\cdot \log{n}
> $$
> ​	当$\tau \rightarrow 0$时，误差也趋近于0

式（3）可以用策略梯度算法暴力求解，但通过将soft value function和Energy based policy联系起来，可以推出值迭代算法，根据式（3）整理出目标函数，并加入折扣银子：
$$
J(\pi)=\sum_{t=0}^T \gamma^t E_{(s_t,a_t)\sim \rho_\pi}\left[r(s_t,a_t)+\alpha\cdot H(\pi(\cdot|s_t)) \right] \tag{12}
$$
现在的目标是调整输入$\pi$，使J最大化，由易到难，先解J的最后一项（t=T)：
$$
\begin{aligned}
J(\cdot|s_T)&=arg\max_{\pi(\cdot|s_T)} E_{a_T\sim\pi(\cdot|s_T)}[r(s_T,a_T)+\alpha\cdot H(\pi(\cdot|s_T))]\\
&=arg\max_{\pi(\cdot|s_T)}\int\left[r(s_T,a_T)-\alpha\cdot \log{\pi(a_T|s_T)} \right]\pi(a_T|s_T)d a_T
\end{aligned}
$$
对$\pi(\cdot|s_T)$求导，令导数为0，得

$r(s_T,a_T)-\alpha\cdot \log{\pi(a_T|s_T)-\alpha}=0 \Rightarrow \pi(a_T|s_T)=\frac{\exp{\frac{1}{\alpha}r(s_T,a_T)}}{e}$

又由于$\pi(a_T|s_T)$是概略，所以$\int\pi(a|s_T)da=1$

得到
$$
\begin{aligned}
\pi(a_T|s_T)&=\frac{\exp{\frac{1}{\alpha}r(s_T,a_T)}}{\int\exp(\frac{1}{\alpha}r(s_T,a_T))da} \longrightarrow 此时Q(s_t,a_t)=r(s_T,a_T)，V(s_T)=\alpha \log\int\exp\left[\frac{1}{\alpha}Q(s_T,a)\right]da\\
&=\frac{\exp\frac{1}{\alpha}Q(s_t,a_T)}{\exp{\frac{1}{\alpha}V(s_t)}}\\
&=\exp \left(\frac{1}{\alpha}(Q(s_T,a_T)-V(S_T) )\right)
\end{aligned}
$$

> 注：对于$F[y]=\int_a^bL(x,y(x),y'(x))dx$，该泛函取得极值的必要条件：$\frac{\partial{L}}{\partial{y}}-\frac{\partial}{\partial x}\frac{\partial L}{\partial y'}=0$

进一步推广到通常情况，依然可得：
$$
\pi(a_t|s_t)=\exp \left(\frac{1}{\alpha}(Q(s_t,a_t)-V(S_T) \right) \tag{13}
$$
得到soft Q-learning得值迭代算法：

> $V(s_{T+1})=0$
>
> for t=T to 0:
>
> ​		$Q(s_t,a_t)=r(s_t,a_t)+\gamma E_{p(s_{t+1}|s_t,a_t)}[V(s_{t+1})] \longrightarrow 不是人为控制得，所以用E，而不是max$
>
> ​		$V(s_t)=\alpha \log\int\exp(\frac{1}{\alpha}Q(s_t,a_t))da_t $



## 策略优化

$$
\pi_{new}^{a_t|s_t}=\frac{\exp{\frac{1}{\alpha}Q_{soft}^{\pi_{old}}(s_t,a_t)}}{\exp{\frac{1}{\alpha}V_{soft}^{\pi_{old}}(s_t)}} \tag{14}
$$

## soft Q-learning(SQL)存在的问题及解决方法

+ 策略评估时，根据式（11），对动作求积分，这个操作在连续空间不可能实现

  解决方法：采样+importance sampling 以近似V的期望，即：
  $$
  V_{soft}^\theta(s)=\alpha\log{E_{q_{a'}}}\left[\frac{\exp{\frac{1}{\alpha}Q_{soft}^\theta}(s_t,a')}{q_{a'}(a')} \right]
  $$
  其中，q是用于采样的分布，初期随即均匀采样，后期根据policy采样。

+ 基于能量的模型式难以处理的

  策略函数可以用值函数表示成EBP，却无法用它来采样（即可以算出$\pi(a_t|s_t)$，但无法根据$\pi(\cdot|s)$这个概率分布来采样一个action，计算$\pi(a_t|s_t)$时带入$a_t$即可，而采样的时候因为这个概率分布很难表示而无法采样）

  算法的作者使用近似推理，如马尔科夫链蒙特卡洛，同时为了加速推理，使用了SVGD训练的推理网络生成近似样本，然后利用KL散度来缩小代理策略$\pi^{\phi}$与EBP的差距:
  $$
  J_{\pi^\phi}(s_t)=D_{KL}(\pi^{\phi}(\cdot|s_t)||\exp{\frac{1}{\alpha}(Q_{soft}^\theta(s_t,\cdot)-V_{soft}^\theta(s_t))})
  $$

  # SAC中的策略评估和策略优化

  ## 策略评估

  在SAC中，算法作者放弃了使用softmax来直接求V函数的值（即式11）

  

  方法一：只打算维持一个值函数Q，即式（6）：
  $$
  Q_{soft}^\pi(s,a)=E_{s'\sim p(s'|s,a),a'\sim \pi}[r(s,a)+\gamma(Q_{soft}^\pi(s',a')+\alpha H(\pi(\cdot|s')))]
  $$
  方法二：同时维持V、Q两个函数，即式（6）（7）
  $$
  Q_{soft}^\pi(s,a)=E_{s'\sim p(s'|s,a)}[r(s,a)+\gamma\cdot V_{soft}^\pi(s')]\\
  V_{soft}^\pi(s)=E_{a \sim\pi} \left[Q_{soft}^{\pi}(s,a)-\alpha\cdot \log\pi(a|s) \right]
  $$

  ## 策略优化

  SAC中的理想策略依然是式（10）的EBP形式，不过因为EBP的采样问题依然存在，只能利用一个高斯分布$\pi$来代替EBP与环境互动。策略优化时让高斯分布尽量靠近EBP。
  $$
  \pi_{new}=arg\min_{x\in\pi}D_{KL}(\pi(\cdot|s_t)||\frac{\exp(\frac{1}{\alpha}Q_{soft}^{\pi_{old}}(s_t,\cdot))}{z_{soft}^{\pi_{old}}(s_t)}) \tag{15}
  $$
  其中$\pi$表示可选的策略集合，实际上是带参数的高斯分布。

  Z函数取代了式14中的$\exp{\frac{1}{\alpha}V_{soft}^{\pi_{old}}(s_t)}$作为配分函数，用于归一化分布，不过对于$\pi(s_t)$来说，两者都是常数，在实际计算中都可以忽略。同时也因为这个原因，在SAC中不再维护V函数。

  作者证明了式15可以像式14一样保证策略的优化。

  ## soft policy iteration

  交替执行策略评估和策略优化将收敛到最优的值函数和最优策略

  ## 实现

  $$
  Q_\theta(s,a): S\times A \rightarrow R\\
  \pi_{\phi}(\cdot|s):S\rightarrow \mu,\sigma \text{高斯分布的均值和方差}
  $$

  根据式（6），Q的损失函数：
  $$
  J_Q(\theta)=E_{(s_t,a_t,s_{t+1})\sim D,a_{t+1}\sim \pi_\phi}[\frac{1}{2}\left(Q_\theta(s_t,a_t)-(r(s_t,a_t)+\gamma\cdot (Q_\theta(s_{t+1},a_{t+1})-\alpha \log{\pi_{\phi}(a_{t+1}|s_{t+1})})) \right)^2] \tag{16}
  $$
  其中$(s_t,a_t,s_{t+1})$是从agent过往与环境的交互中产生的数据（replay buffer)抽取的，但$a_t$是在训练时临时从$\pi_\phi$中采集出来的。

  策略$\pi_\phi$训练时的损失函数：
  $$
  \begin{aligned}
  J_\pi(\phi)&=D_{KL}(\pi_\phi(\cdot|s_t)||\exp{(\frac{1}{\alpha}Q_\theta(s_t,\cdot)-\log{Z(s_t)})})\\
  &=E_{s_t\sim D,a_t\sim\pi_\phi}\left[\log{\left(\frac{\pi_\phi(a_t|s_t)}{\exp{\frac{1}{\alpha}Q_\theta(s_t,a_t)-\log{Z(s_t)}}}\right)}\right]\\
  &=E_{s_t\sim D,a_t\sim\pi_\phi}\left[ \log{\pi_\phi(a_t|s_t)}-\frac{1}{\alpha}Q_\theta(s_t,a_t)+\log{Z(s_t)}\right]
  \end{aligned}
  $$
  同样，这里的$s_t$从缓存中获得，$a_t$从当前的策略$\pi_\phi$中采样得到。

  因为高斯分布采样的过程不可导，由$a_t\sim\pi_\phi$，引入reparameterization技术，有：
  $$
  a_t=f_\phi(\varepsilon_t;s_t)=f_\phi^\mu(s_t)+\varepsilon_t\odot f_\phi^\sigma(s_t)
  $$
  即先从一个单位高斯分布$\mathcal{N}$采样，再把采样值乘以标准差后加上均值。这样就可以认为是从策略高斯分布采样，并且这个采样动作的过程对于策略函数来说是可导的。

  此外，最后包含Z的那一项不受$\phi$影响，可将其忽略。
  
  最终：
  $$
  J_{\pi}(\phi)=E_{s_t\sim D,\varepsilon_t\sim \mathcal{N}}\left[\alpha\log\pi_\phi(f_\phi(\varepsilon_t;s_t)|s_t)-Q_\theta(s_t,f_\phi(\varepsilon_t;s_t))\right] \tag{17}
  $$

  其中$\mathcal{N}$是单位高斯分布

  > 在初版的SAC中作者表示同时维护两个值函数可以使训练更稳定，不过在第二版中，作者引入自然调整温度系数$\alpha$，使得SAC更稳定，于是就只保留了Q函数。

  # tricks in SAC

  锦上添花的trick: double Q network,target work
  
  ## Automating Entropy Adjustment for MERL
  
  前文提到过，温度系数 $\alpha$ 作为一个超参数，可以控制MERL对熵的重视程度。但是不同的强化学习任务，甚至同一任务训练到不同时期，都各自有自己适合的  $\alpha$ ，而且这个超参数对性能的影响明显，还好，这个参数可以让SAC自己调节。实现在最优动作不确定的某个状态下，熵的取值应该大一点；而在某个最优动作比较确定的状态下，熵的取值可以小一点。

  作者将其构造为一个带约束的优化问题：最大化期望收益的同时，保持策略的熵大于一个阈值。为了自动调整熵正则化项，SAC将强化学习的目标改为一个带约束的优化问题：
  $$
  \max_{\pi} E_\pi\left[\sum_{t=0}^\infty r(s_t,a_t) \right],s.t. E_{(s_t,a_t)\sim \rho_\pi}[-\log{\pi_t(a_t|s_t)}] \ge\mathcal{H}_0 \tag{18}
  $$
  其中，$\mathcal{H}_0$是预先定义好的最小策略熵的阈值。上式即最大化期望回报，同时约束熵的均值大于$\mathcal{H}_0$。
  
  最终得到关于$\alpha$的优化的损失函数：
  $$
  J(\alpha)=E_{a_t\sim\pi_t}[-\alpha\log\pi_t(a_t|\pi_t)-\alpha \mathcal{H}_0] \tag{19}
  $$
  
  即当策略的熵低于目标值$\mathcal{H}_0$时，训练目标$L(\alpha)$会使$\alpha$的值增大，进而在上述最小化损失函数$L_\pi(\theta)$的过程中增加了策略熵对应项的重要性；而当策略的熵高于目标值$\mathcal{H}_0$时，训练目标$L(\alpha)$会使$\alpha$的值减小，进而使得策略训练时更专注于价值提升。
  
  ## squashed Gaussian Trick
  
  action从正态分布中得到，则动作的值$u\in(-\infty,+\infty)$。如果动作值范围有界限，比如$(-1,1)$，就需要对抽样得到的u进行转换，$\mu(u|s)$是对应的概率密度,用tanh映射到$a\in (-1,1)$，随机变量就换元了，计算$\log(\pi_\psi(a_t|s_t))$也要有相应的变换。
  $$
  a=\tan{u}\\
  \tan'{u}=1-\tan^2{u}\\
  $$
  有：
  $$
  \pi(a|s)=\mu(u|s)\bigg|\det{(\frac{da}{du})}\bigg|^{-1} \Rightarrow\log{\pi(a|s)}=\log{\mu(a|s)}-\sum_{i=1}^D \log{(1-\tanh^2(u_i))} \tag{20}
  $$
  其中D是U的维度，det是求行列式。
  
  但在实际的运算中，使用的计算式为：$2*(\log{2}-\pi_a-softplus(-2\pi_a))$，是对Tanh squashing correction公式更加数值稳定的替代。
  
  > 证明：
  > $$
  > \begin{aligned}
  > \log{(1-\tanh^2{u})}&=\log{sech^2(u)}\\
  > &=2\log{sech(u)}\\
  > &=2\log{(\frac{2}{\exp{(u_i)}+\exp{(-u_i)}})}\\
  > &=2\left(\log{2}-\log{(\exp{(u_i)}+\exp{(-u_i)})} \right)\\
  > &=2\left(\log{2}-\log{(\exp{u_i}\cdot(1+\exp{(-2u_i)}))} \right)\\
  > &=2(\log{2}-u_i-\log{(1+\exp{(-2u_i)})})\\
  > &=2(\log{2}-u_i-softplus(-2u_i))
  > \end{aligned}
  > $$
  > 
  
  
  
  # 算法流程
  
  > 用随机的网络参数$\omega_1、\omega_2和\theta$分别初始化Critic网络$Q_{\omega_1}(s,a)和Q_{\omega_2}(s,a)$和Actor网络$\pi_\theta(s)$
  >
  > 复制相同的参数$\omega_1^-\leftarrow \omega_1、\omega_2^-\leftarrow \omega_2和\theta^- \leftarrow\theta$，分别初始化目标网络$Q_{\omega_1^-}(s,a)、Q_{\omega_2^-}(s,a)$和$\pi_{\theta^-}(s)$
  >
  > 初始化经验回放池D
  >
  > for 序列e=1 to E do:
  >
  > ​		获取环境初始状态$s_1$
  >
  > ​		for 时间步 t=1 to T do:
  >
  > ​				$a_t=\pi_\theta(s_t)$
  >
  > ​				$s_{t+1},r_t=env.step(action)$
  >
  > ​				D.append(($s_t,a_t,r_t,s_{t+1}$))
  >
  > ​				for 训练轮次 k=1 to K do:
  >
  > ​						从D中取N个元组样本$\{(s_i,a_i,r_i,s_{i+1})\}_{i=1,\dots,N}$
  >
  > ​						对每个元组，用目标网络计算$y_i=r_i+\gamma\min_{j=1,2}Q_{\omega_j^-}(s_{i+1},a_{i+1})-\alpha\log\pi_\theta(a_{i+1}|s_{i+1}),其中a_{i+1}\sim \pi_\theta(\cdot|s_{i+1})$
  >
  > ​						对两个Critic网络都进行如下更新：对j=1,2，最小化损失函数$L=\frac{1}{N}\sum_{i=1}^N(y_i-Q_{\omega_j}(s_i,a_i))^2$
  >
  > ​						用重参数化技巧采样动作$\tilde{a_i}$，然后用以下损失函数更新当前Actor网络：
  > $$
  > L_\pi(\theta)=\frac{1}{N}\sum_{i=1}^N(\alpha\log\pi_\theta(\tilde{a_i}|s_i)-\min_{j=1,2}Q_{\omega_j}(s_i,\tilde{a_i}))
  > $$
  > ​						更新熵正则项的系数$\alpha$
  >
  > ​						更新目标网络：
  > $$
  > \omega_1^-\leftarrow \tau\omega_1+(1-\tau)\omega_1^-\\
  > \omega_2^-\leftarrow \tau\omega_2+(1-\tau)\omega_2^-
  > $$
  > ​						end for
  >
  > ​				end for
  >
  > end for

# 参考资料

#[Feliks](https://www.zhihu.com/people/zhu-jin-59-34-82) SAC(Soft Actor-Critic)阅读笔记 https://zhuanlan.zhihu.com/p/85003758

张伟楠 沈键 俞勇 《动手学强化学习》 人民邮电出版社

Pytorch深度强化学习4. SAC中的Squashed Gaussian Trick - 0xAA的文章 - 知乎 https://zhuanlan.zhihu.com/p/138021330