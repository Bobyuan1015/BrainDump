# 0 简版


## 0.1 核心思想概述
SAC（Soft Actor-Critic）是基于**最大熵强化学习框架**的off-policy算法，核心创新在于通过**熵正则化**自动平衡探索与利用：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \big( r(s_t, a_t) + \alpha H(\pi(\cdot | s_t)) \big) \right]
$$

- $H(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$：策略熵
- $\alpha > 0$：温度系数（自动调整）

> **与常规RL的区别**：常规RL通过$\epsilon$-greedy等外部机制探索，SAC通过内生熵正则化自动实现探索-利用平衡。

## 0.2. 理论基础：玻尔兹曼分布
### 0.2.1 定义与形式
策略建模为玻尔兹曼分布：
$$ \pi(a|s) = \frac{1}{Z(s)} \exp \left( \frac{Q(s,a)}{\alpha} \right) $$
其中配分函数：
$$ Z(s) = \int \exp \left( \frac{Q(s,a)}{\alpha} \right) da $$

### 0.2.2 温度参数 $\alpha$ 的作用
- $\alpha \to 0$：趋近贪婪策略（纯利用）
- $\alpha \to \infty$：趋近均匀随机策略（纯探索）
- **关键特性**：高Q值动作概率更高，但低Q值动作仍有非零概率（保证探索性）

### 0.2.3 玻尔兹曼分布的优势
1. **自然平衡探索与利用**：通过可学习的$\alpha$自动调节
2. **贝尔曼一致性**：软值函数有解析解
   $$ V(s) = \alpha \log \int \exp(Q(s,a)/\alpha) da $$
3. **多模态支持**：能同时捕获多个高价值动作
4. **对Q值误差鲁棒**：避免被错误Q值过度引导

## 0.3. 变分推断视角的SAC推导
### 0.3.1 优化目标建模
通过变分法建立策略优化与分布匹配的等价关系：

$$
\begin{aligned}
&\min_\theta D_{KL} \left( \pi_\theta \big\| \frac{\exp(Q^\pi/\alpha)}{Z} \right) \\
= &\min_\theta \mathbb{E}_{a\sim\pi_\theta} \left[ \log\pi_\theta - \frac{Q^\pi}{\alpha} \right] + \text{const}
\end{aligned}
$$

### 0.3.2 拉格朗日乘子法求解
带约束优化问题：
$$ \max_\pi \mathbb{E}_{a\sim\pi} \left[ Q^\pi(s,a) - \alpha\log\pi \right] \quad \text{s.t.} \quad \int\pi da = 1 $$

构造拉格朗日函数：
$$
\mathcal{L} = \int \pi Q da - \alpha \int \pi \log\pi da + \lambda \left( 1 - \int \pi da \right)
$$

泛函求导得最优策略：
$$ \frac{\delta\mathcal{L}}{\delta\pi} = Q - \alpha(\log\pi + 1) - \lambda = 0 $$
$$ \Rightarrow \pi^* = \exp \left( \frac{Q}{\alpha} - 1 - \lambda \right) $$

归一化处理得最终形式：
$$ \boxed{\pi^*(a|s) = \frac{\exp(Q^\pi(s,a)/\alpha)}{\int \exp(Q^\pi(s,a)/\alpha) da}} $$

## 0.4 SAC目标函数详解
### 0.4.1 策略网络目标（Actor Loss）
$$
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}, a_t \sim \pi_\phi} \left[ \alpha \log \pi_\phi(a_t|s_t) - Q_\theta(s_t,a_t) \right]
$$

**推导过程**：
1. 基于最大熵目标函数：
   $$ J(\pi) = \mathbb{E} \left[ \sum_t \gamma^t (r_t + \alpha H(\pi)) \right] $$
2. 等价于最小化KL散度：
   $$ J_\pi(\phi) = \mathbb{E}_{s_t} \left[ D_{KL} \left( \pi_\phi \big\| \frac{\exp(Q_\theta/\alpha)}{Z} \right) \right] $$

**重参数化技巧**（高斯策略）：
$$ a_t = \mu_\phi(s_t) + \sigma_\phi(s_t) \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,1) $$

### 0.4.2 Q网络目标（Critic Loss）
双Q网络设计：
$$ J_Q(\theta_i) = \mathbb{E}_{(s_t,a_t,r_t,s_{t+1}) \sim \mathcal{D}} \left[ (Q_{\theta_i}(s_t,a_t) - y_t)^2 \right],  i=1,2 $$

目标值计算：
$$
y_t = r_t + \gamma \mathbb{E}_{a_{t+1} \sim \pi_\phi} \left[ \min_{j=1,2} Q_{\bar{\theta}_j}(s_{t+1},a_{t+1}) - \alpha \log \pi_\phi(a_{t+1}|s_{t+1}) \right]
$$

**技术细节**：
- $\min$操作：减少Q值过估计
- 目标网络$\bar{\theta}$：稳定训练（软更新：$\bar{\theta} \leftarrow \tau \theta + (1-\tau)\bar{\theta}$）

### 0.4.3 温度参数目标（Entropy Regularization）
$$ J(\alpha) = \mathbb{E}_{s_t \sim \mathcal{D}, a_t \sim \pi_\phi} \left[ -\alpha (\log \pi_\phi(a_t|s_t) + \bar{\mathcal{H}}) \right] $$

**推导过程**：
1. 带熵约束的优化：
   $$ \max_\pi \mathbb{E}[\sum \gamma^t r_t] \quad \text{s.t.} \quad H(\pi) \geq \bar{\mathcal{H}} $$
2. 拉格朗日函数：
   $$ \mathcal{L} = \mathbb{E}[r_t + \alpha(H(\pi) - \bar{\mathcal{H}})] $$
3. 优化对数空间参数$\tilde{\alpha}=\log\alpha$：
   $$ \nabla_{\tilde{\alpha}} J = -\alpha \mathbb{E}[\log\pi + \bar{\mathcal{H}}] $$

## 0.5 网络架构与训练流程
### 0.5.1 网络组成
| 网络类型         | 数量 | 参数           | 功能                     |
| ---------------- | ---- | -------------- | ------------------------ |
| 策略网络 (Actor) | 1    | $\phi$         | 输出动作分布$\mu,\sigma$ |
| Q网络 (Critic)   | 2    | $\theta$       | 估计Q值                  |
| 目标Q网络        | 2    | $\bar{\theta}$ | 稳定目标值计算           |
| 温度参数         | 1    | $\alpha$       | 调节熵正则化强度         |

###0. 5.2 训练流程
1. **采样**：从经验池$\mathcal{D}$采样$(s_t,a_t,r_t,s_{t+1})$
2. **Critic更新**：最小化$J_Q(\theta_i)$
3. **Actor更新**：最大化$J_\pi(\phi)$（重参数化）
4. **温度更新**：最小化$J(\alpha)$
5. **目标网络更新**：$\bar{\theta} \leftarrow \tau \theta + (1-\tau)\bar{\theta}$

### 0.5.3 关键实现技术
1. **重参数化**：解决策略梯度高方差问题
   $$ a_t = \mu_\phi(s_t) + \sigma_\phi(s_t) \odot \varepsilon $$
2. **双Q网络**：缓解值函数过估计
   $$ Q = \min(Q_{\theta_1}, Q_{\theta_2}) $$
3. **目标网络**：稳定训练过程
4. **自动熵调节**：$\alpha$的动态优化

------



# 1. SAC（Soft Actor-Critic）

论文：
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor    https://arxiv.org/pdf/1801.01290



**Hard版本不加熵？**

在常规强化学习中，探索通常通过外部机制（如随机初始化、*ϵ*-greedy、动作空间噪声）实现，而SAC通过**内生的熵正则化**自动平衡探索与利用。Hard版本的目标是纯粹最大化奖励，因此不需要在值函数中引入熵项。





## 1.1 玻尔兹曼分布的定义与特性

玻尔兹曼分布(Boltzmann Distribution)最初来源于统计物理学，用于描述粒子在热平衡状态下的能量分布。在强化学习中，它被推广为一种基于能量的概率分布，其形式为:

$$ p(x)\propto\exp\left(\frac{E(x)}{T}\right), $$

其中:

$ \bullet $E(x)是状态或动作的能量(在强化学习中通常用负Q值表示，即$ E(x)=-Q(s,a) $);

$ \bullet $T是温度参数(在SAC中记为a)，控制分布的"平坦程度"。

在SAC中，策略$ \pi(a|s) $被建模为玻尔兹曼分布:

$$ \pi(a|s)\propto\exp\left(\frac{Q(s,a)}{a}\right). $$



### 1.1.1 玻尔兹曼分布的关键特性

1. 能量（Q值）与概率的关系：
   - 高Q值动作概率更高；动作概率呈指数增长。
   - 低Q值动作仍有非零概率：即使某些动作的Q值较低，它们仍可能被选择，保证探索性。

2. 温度参数 α 的作用：
   - α → 0：分布趋近于 贪婪策略（只选最优动作）。
   - α → ∞：分布趋近于 均匀随机策略（完全探索）。
   - 在SAC中，α是可学习的，能自动调节探索与利用的平衡。

3. 归一化与配分函数：
   - 珂尔曼分布需要配分函数 Z = ∫ exp(Q(s, a)/α) da 进行归一化，但在SAC中，由于策略优化通过采样实现，无需显式计算 Z。

4. 多模态支持：
   - 如果Q函数有多个峰值（即多个高奖励动作），珂尔曼分布会自动分配概率给这些动作，而斯托克斯策略等单峰分布可能无法捕捉这种特性。

### 1.1.2 珂尔曼分布中 p(x) 与 E(x) 的关系

在基于能量的模型（Energy-Based Models, EBMs）中，珂尔曼分布描述了状态或动作的概率 p(x) 与其能量 E(x) 之间的关系：

$$
p(x) = \frac{1}{Z} \exp \left( -\frac{E(x)}{T} \right),
$$

其中：
- \( p(x) \)：状态或动作 x 的概率分布。
- \( E(x) \)：能量函数，表示 x 的“不稳定性”（能量越高，概率越低）。
- \( T \)：温度参数（在SAC中 α），控制分布的温度。
- \( Z \)：配分函数（归一化常数），确保 \( \int p(x) dx = 1 \)，定义为：

$$
Z = \int \exp \left( -\frac{E(x)}{T} \right) dx.
$$

**关键关系与条件**

1. 能量与概率的关系
   - 能量 \( E(x) \) 越高，p(x) 越低（高能量状态更不稳定，概率小）。
   - 在强化学习中，通常将 Q 值视为能量，即 \( E(x) = -Q(s, a) \)，因此：

     $$
     p(a|s) \propto \exp \left( \frac{Q(s, a)}{T} \right).
     $$

     这样高 Q 值动作的概率更高。

2. 温度 \( T \) 的作用
   - \( T -> 0 \)：分布趋近于 Dirac delta 函数（即选择最大的状态存活，贪婪策略）。
   - \( T -> ∞ \)：分布趋近于均匀分布（完全探索）。

   在SAC中，温度 α 是可学习的，用于动态调节的权重。






## 1.2为什么珂尔曼分布适合SAC?

- 其中 $Z(s)=\int\exp (\frac{Q(s,a)}{\alpha })da$ 是配分函数(partition function)，用于归一化。

- 为什么用$\frac{Q(s,a)}{\alpha }$? 因为$\alpha$作为温度参数，控制了分布的平滑程度。当$\alpha$较大时，分布更均匀(更随机，鼓励探索)；当$\alpha$较小时，分布更集中于高Q值的动作(更倾向于利用)。**指数形式$\exp (\frac{Q(s,a)}{\alpha })$确保了动作的概率**与**Q值**成正比，而$\alpha$调节了这种关系的强度。

- 通过最小化KL散度$\mathrm{KL}(\pi(\cdot|s)||p(\cdot|s))$，可以推导出最优策略的形式：



**自然平衡探索与利用**

   - SAC的核心思想是最大化强化学习的累积奖励，而**珂尔曼分布**通过**温度参数 α** 直接控制**探索**的大小：
     - 高温（大α）鼓励探索，避免策略过早收敛；
     - 低温（小α）偏向利用，专注高Q值动作。

**与贝尔曼过程的一致性**

   - SAC的Soft Q函数和Soft值函数定义如下：

     $$Q(s, a) = r(s, a) + \gamma \mathbb{E}_{s'}[V(s')], \quad V(s) = \mathbb{E}_{a \sim \pi}[Q(s, a) - \alpha \log \pi(a|s)].$$

     当策略 π 是珂尔曼分布时，值函数 V(s) 可解析表示为：

     $$V(s) = \alpha \log \int \exp(Q(s, a)/\alpha) da,$$

     这种形式与统计物理中的自由能（Free Energy）一致，确保了理论自治。

6. 策略优化的闭式解
   - 通过最小化KL散度 $D_{KL}(\pi || \exp(Q/\alpha)/Z)$，策略 π 的最优解就是珂尔曼分布，使得优化过程更加高效稳定。

7. 对Q函数计算误差的鲁棒性
   - 传统策略（如DQN）对Q函数估计非常敏感，而珂尔曼分布会为动作保留一定概率，避免策略被错误的Q值引导。



# 2. 完整推导（brief）

1. **软Q值Bellman方程的原始形式**

首先回顾强化学习中的软Q值Bellman方程：

$$
Q^\pi(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim p} [V^\pi(s_{t+1})]
$$

其中软状态值函数定义为：

$$
V^\pi(s_t) = \mathbb{E}_{a \sim \pi} [Q^\pi(s_t, a_t) - \alpha \log \pi(a_t|s_t)]
$$

2. **策略优化的基本目标**

强化学习的目标是**最大化带有熵正则化**的累积回报：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha H(\pi(a_t | s_t)) \right) \right]
$$

这里：
- $H(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ 是策略熵
- $\alpha > 0$ 是温度系数



3. **从值函数到目标函数的转化**

**为什么不能直接用Bellman方程优化?**

   - 计算不可行性：连续动作空间下$\mathbb{E}_{a\sim \pi}[Q^\pi(s, a)]$的精确积分分难以计算。
   - 采样效率：通过目标函数的期望形式，只需要动作分布采样即可估计梯度。
   - 策略参数化：目标函数允许用神经网络直接参数化策略，而Bellman方程仅更新值函数。

   这种定义方式完美接轨了值函数学习和策略优化，是SAC算法能够高效处理连续问题的核心设计。



**通过以下步骤建立关系**：

- 递归展开：将$V^\pi(s)$展开为无限时域的期望：

$$
V^\pi(s_0) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha \log \pi(a_t|s_t) \right) \right]
$$

这是$J(\pi)$的表达式。

- 策略梯度推导：为其求梯度：

$$
\nabla_\theta J(\pi) = \mathbb{E}_{s_t \sim p_{\pi}, a_t \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \left( Q^\pi(s_t, a_t) - \alpha \log \pi_\theta(a_t|s_t) \right) \right]
$$

- 最优策略的解析形式：通过变分法求解$\arg\max_\pi J(\pi)$，得到：

$$
\pi^*(a|s) = \frac{\exp(Q^\pi(s, a)/\alpha)}{Z(s)}
$$

其中$Z(s)$是归一化常数。



# 3. 变分推断在SAC中推导

1. 问题建模：从策略优化到分布匹配

在强化学习中，我们希望找到最优策略$\pi^*(a|s)$最大化累积奖励。引入熵正则化后，目标变为：

$$
\pi^* = \arg\max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha H(\pi(a_t | s_t)) \right) \right]
$$

其中$H(\pi) = -\mathbb{E}_{a\sim \pi}[\log \pi(a|s)]$。

2. 目标分布的构造

通过软Q函数构造目标分布：

$$
p(a|s) = \frac{\exp(Q^\pi(s, a)/\alpha)}{Z(s)}, \quad Z(s) = \int \exp(Q^\pi(s, a)/\alpha) da
$$

3. KL散度最小化的详细推导

我们需要最小化策略$\pi(a|s)$目标分布$p(a|s)$的KL散度：

$$
D_{KL}(\pi_\theta || p) = \mathbb{E}_{a\sim \pi_\theta} \left[ \log \pi_\theta(a|s) - \log p(a|s) \right]
= \mathbb{E}_{a\sim \pi_\theta} \left[ \log \pi_\theta(a|s) - \frac{Q^\pi(s, a)}{\alpha} + \log Z(s) \right]
= \mathbb{E}_{a\sim \pi_\theta} \left[ \log \pi_\theta(a|s) - \frac{Q^\pi(s, a)}{\alpha} \right] + \text{const}
$$

关键步骤：

1. 展开$p(a|s)$的定义
2. $\log Z(s)$无关$α$，视为常数
3. 最小化KL散度等价于：

$$
\min_\theta \mathbb{E}_{a\sim \pi_\theta} \left[ \frac{\log \pi_\theta(a|s) - Q^\pi(s, a)}{\alpha} \right]
$$

4. 转化为最大化问题

将最小化问题转化为等效的最大化问题：

$$
\max_\theta \mathbb{E}_{a\sim \pi_\theta} \left[ \frac{Q^\pi(s, a)}{\alpha} - \log \pi_\theta(a|s) \right]
$$

令$\alpha = 1$时（一般化情形类似），得到SAC的目标函数：

$$
\max_\theta \mathbb{E}_{a\sim \pi_\theta} \left[ Q^\pi(s, a) - \log \pi_\theta(a|s) \right]
$$



## 3.2 通过变分法求极值：

### 3.2.1 拉格朗日乘子的标准形式

对于带约束的优化问题：

$$
\max_x f(x) \quad \text{s.t.} \quad g(x) = 0
$$

其拉格朗日函数为：

$$
L(x, \lambda) = f(x) - \lambda g(x)
$$

其中：
- $\lambda$ 是拉格朗日乘子（量度或向量，取决于约束数量）
- $\lambda$ 负号确保（也可用正号，但需保持一致性）

### 3.2.2 拉格朗日应用SAC

在SAC的策略优化中，我们需要：

$$
\max_\pi \mathbb{E}_{a\sim \pi} \left[ Q^\pi(s, a) - \alpha \log \pi(a|s) \right]
\quad \text{s.t.} \quad \int \pi(a|s) da = 1 \quad (\text{概率归一化约束})
$$


1. 构造拉格朗日函数：

$$
L(\pi, \lambda) = \int \pi(a|s) Q^\pi(s, a) da - \alpha \int \pi(a|s) \log \pi(a|s) da + \lambda \left( 1 - \int \pi(a|s) da \right)
$$

关键点说明：
- 奖励项：最大化期望Q值

- 熵项：$-\alpha \int \pi(a|s) \log \pi(a|s) da$ 是香农熵的连续形式，鼓励探索

- 约束项：确保$\pi(a|s)$是有效的概率分布

  

#### 3.2.2.1 变分推断求解

**1.变分法基础**

变分法是处理函数极值的工具（对比微积分处理变量极值）。核心操作是对泛函$ J[f] $求关于函数f的导数（称为泛函导数或变分导数）。

**2.具体推导步骤**

(1)对拉格朗日函数L求泛函导数：

$$ \frac{\delta\mathrm{L}}{\delta\pi}=\frac{Q^{\pi}(s, a)}{\alpha}-\log\pi(a\mid s)-1-\lambda $$

(2)令导数为零（极值必要条件）：

$$ \frac{Q^{\pi}(s, a)}{\alpha}-\log\pi(a\mid s)-1-\lambda=0 $$

(3)解这个方程得到：

$$ \pi^{*}(a\mid s)=\exp\left(\frac{Q^{\pi}(s, a)}{\alpha}-1-\lambda\right) $$

**3.归一化处理**

通过约束条件$ \int\pi da=1 $确定$ \lambda $：

$$ \begin{aligned}
\int\exp\left(\frac{Q^{\pi}}{\alpha}-1-\lambda\right) da&=1\\
\Rightarrow e^{-1-\lambda}\int\exp\left(Q^{\pi} /\alpha\right) da&=1\\
\Rightarrow\pi^{*}(a\mid s)=\frac{\exp\left(Q^{\pi} /\alpha\right)}{\int\exp\left(Q^{\pi} /\alpha\right) da}
\end{aligned} $$

**3.变分法VS微积分**

| 概念     | 传统微积分   | 变分法                               |
| -------- | ------------ | ------------------------------------ |
| 优化对象 | 变量x        | 函数f(x)                             |
| 极值条件 | $ df/dx=0 $  | $ \delta J/\delta f=0 $              |
| 典型应用 | 求函数极值点 | 最速降线问题、力学中的最小作用量原理 |





# 4. SAC 的目标函数

SAC是一种基于最大熵强化学习框架的算法，旨在通过最大化期望累积奖励并保持策略的随机性（通过熵正则化）来平衡探索与利用。其目标函数主要包括**策略网络目标函数（Actor Loss）、价值网络目标函数（Critic Loss）以及温度参数目标函数（Entropy Regularization）**。以下详细介绍每个目标函数、变量含义及推导过程。

## 4.1 策略网络目标函数（Actor Loss）

目标函数：

$$ J_{\pi}(\phi)=\mathbb{E}_{s_{t}\sim\mathcal{D}, a_{t}\sim\pi_{\phi}}\left[ Q_{\theta}\left(s_{t}, a_{t}\right)-\alpha\log\pi_{\phi}\left(a_{t}\mid s_{t}\right)\right] $$

变量含义：

- $ s_{t} $：当前状态，从经验回放池 $ \mathcal{D} $ 中采样。
- $ a_{t} $：根据策略 $ \pi_{\phi}(\cdot|s_{t}) $ 采样的动作。
- $ \pi_{\phi} $：参数为 $ \phi $ 的策略网络，通常输出动作分布（如高斯分布的均值和方差）。
- $ Q_{\theta}(s_{t}, a_{t}) $：由价值网络（参数为 $ \theta $）估计的状态-动作对的 Q 值，表示在状态 $ s_{t} $ 下采取动作 $ a_{t} $ 的期望回报。
- $ \alpha $：温度参数，控制熵正则化的权重，平衡探索与利用。
- $ \log\pi_{\phi}(a_{t}|s_{t}) $：策略在状态 $ s_{t} $ 下选择动作 $ a_{t} $ 的对数概率，代表策略熵的负值。
- $ \mathcal{D} $：经验回放池，存储历史交互数据 $ \left(s_{t}, a_{t},r_{t},s_{t+1}\right) $ 。

推导过程：

SAC的策略优化基于最大熵强化学习框架，目标是最大化期望累积奖励并同时最大化策略熵：

$$ J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^{t}\left(r\left(s_{t}, a_{t}\right)+\alpha\mathcal{H}\left(\pi\left(\cdot|s_{t}\right)\right)\right)\right] $$

其中：
- $\tau = (s_0, a_0, s_1, a_1, \dots)$: 由策略 $\pi$ 生成的轨迹。
- $r(s_t, a_t)$: 即时奖励。
- $\gamma$: 折扣因子，$0 \leq \gamma < 1$。
- $\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a_t \sim \pi(\cdot|s_t)}[\log \pi(a_t|s_t)]$: 策略熵。
- $\alpha$: 温度参数。

为了优化策略，SAC 使用 Q 函数近似期望回报。策略优化的目标可以重写为：

$$
\max_\pi \mathbb{E}_{s_t \sim D, a_t \sim \pi(\cdot|s_t)}\left[Q(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]
$$

代入定义:

$$
\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a_t \sim \pi(\cdot|s_t)}[\log \pi(a_t|s_t)]
$$

因此，目标函数为：

$$
J_\pi(\varphi) = \mathbb{E}_{s_t \sim D, a_t \sim \pi(\cdot|s_t)}\left[Q_\theta(s_t, a_t) - \alpha \log \pi_\varphi(a_t|s_t)\right]
$$

从 KL 散度的角度，策略优化等价于最小化当前策略 $\pi_\varphi$ 与最优策略分布之间的 KL 散度：

$$
J_\pi(\varphi) = \mathbb{E}_{s_t \sim D} \, \mathbb{E}_{a_t \sim \pi_\varphi}\left[ D_{KL}\left( \pi_\varphi(\cdot|s_t) \parallel \exp\left(\frac{Q_\theta(s_t, \cdot)}{\alpha}\right) / Z(s_t) \right) \right]
$$

其中 $Z(s_t) = \int \exp(Q_\theta(s_t, a_t)/\alpha)da$ 是归一化常数。展开 KL 散度:

$$
D_{KL}\left( \pi_\varphi \parallel \exp(Q_\theta/\alpha) / Z(s_t) \right) = \mathbb{E}_{a_t \sim \pi_\varphi}\left[\log \pi_\varphi(a_t|s_t) - \frac{Q_\theta(s_t, a_t)}{\alpha} + \log Z(s_t)\right]
$$

最终目标函数 $Z(s_t)$，策略优化目标转化为最大化：

$$
J_\pi(\varphi) = \mathbb{E}_{s_t \sim D, a_t \sim \pi_\varphi}\left[Q_\theta(s_t, a_t) - \alpha \log \pi_\varphi(a_t|s_t)\right]
$$
为了计算稳定性，SAC 使用重参数化技术，假设策略 πθ 输出高斯分布（均值 μφ(st)，标准差 σφ(st)），动作表示为：

$$
a_t = f_\theta(\varepsilon_t) = \mu_\phi(s_t) + \sigma_\phi(s_t) \cdot \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)
$$

目标函数的梯度为：

$$
\nabla_\theta J_\pi(\phi) \approx \mathbb{E}_\varepsilon [ Q_\theta(s_t, f_\theta(\varepsilon_t)) - \alpha \log \pi_\phi(\varepsilon_t|s_t) ]
$$

通过梯度上升优化此 φ。

## 4.2 价值网络目标函数 (Critic Loss)

目标函数：

SAC 使用两个 Q 网络（参数为 θ1, θ2）以减少 Q 值估计，目标函数为方 Bellman 误差：

$$
J_Q(\theta_i) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim \mathcal{D}} \left[ \left( Q_{\theta_i}(s_t, a_t) - y_t \right)^2 \right], \quad i=1,2
$$

其中目标值 $y_t$ 为：

$$
y_t = r_t + \gamma \mathbb{E}_{a_{t+1} \sim \pi_\phi} \left[ \min_{i=1,2} Q_{\theta_i}(s_{t+1}, a_{t+1}) - \alpha \log \pi_\phi(a_{t+1}|s_{t+1}) \right]
$$

**变量定义：**
- $Q_{\theta_i}(s_t, a_t)$: 第 $i$ 个 Q 网络的 Q 值。
- $r_t$: 即时奖励。
- $s_{t+1}$: 下一状态。
- $Q_{\theta_i}(s_{t+1}, a_{t+1})$: 目标网络（参数为 $\theta_i$）的 Q 值，目标网络是主 Q 网络的延迟更新版本。
- $\min_{i=1,2} Q_{\theta_i}(s_{t+1}, a_{t+1})$: 获取两个 Q 网络的最小值，减少过估计。
- $\alpha \log \pi_\phi(a_{t+1}|s_{t+1})$: 熵正则化项。

**推导过程：**
Q 函数基于最大熵框架的 Bellman 方程推导：
$$
Q(s_t, a_t) = r_t + \gamma E_{s_{t+1}, a_{t+1} \sim \pi} [Q(s_{t+1}, a_{t+1}) - \alpha \log \pi(a_{t+1} | s_{t+1})]
$$

目标值 $y_t$ 是 Bellman 方程的单步估计，使用目标 Q 网络和当前策略：

$$
y_t = r_t + \gamma E_{a_{t+1} \sim \pi} [\min_{i=1,2} Q_{\theta_i} (s_{t+1}, a_{t+1}) - \alpha \log \pi(a_{t+1} | s_{t+1})]
$$

Q 网络通过最小化 Bellman 误差更新：

$$
J_Q(\theta_i) = E_{(s_t, a_t, r_t, s_{t+1}) \sim D} [(Q_{\theta_i} (s_t, a_t) - y_t)^2]
$$

梯度为：

$$
\nabla_{\theta_i} J_Q(\theta_i) = \nabla_{\theta_i} (Q_{\theta_i} (s_t, a_t) - y_t)^2
$$

## 4.3 温度参数a目标函数 (Entropy Regularization)

目标函数：

SAC 动态调整 $\alpha$，使策略趋近目标熵 $\mathcal{H}$：

$$
J(\alpha) = E_{s_t \sim D, a_t \sim \pi} [-\alpha (\log \pi_{\phi}(a_t | s_t) + \mathcal{H})]
$$

变更含义：

- $\mathcal{H}$: 目标熵，通常设为 $-\dim(A)$
- $\mathcal{H}(\pi_{\phi}(\cdot | s_t)) = -E_{a_t \sim \pi} [\log \pi_{\phi}(a_t | s_t)]$: 实际熵。

**推导过程：**

SAC 的优化问题包含熵约束：

$$
\max_{\pi} E_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right] \text{ s.t. } \mathcal{H}(\pi(\cdot | s_t)) \geq \mathcal{H}
$$

使用拉格朗日乘子法，构造拉格朗日函数：
$$
L(\pi, \alpha) = E_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha \left( \mathcal{H}(\pi(\cdot|s_t)) - \overline{\mathcal{H}} \right) \right) \right]
$$

分离 $\alpha$ 的优化部分：

$$
J(\alpha) = E_{s_t \sim D, a_t \sim \pi_{\tau}} \left[ \alpha \left( \mathcal{H}(\pi(\phi|\cdot|s_t)) - \overline{\mathcal{H}} \right) \right]
$$

代入 $\mathcal{H}(\pi_{\phi}) = -E_{a_t \sim \pi_{\tau}} \left[ \log \pi_{\phi}(a_t|s_t) \right]$：

$$
J(\alpha) = E_{s_t \sim D, a_t \sim \pi_{\tau}} \left[ -\alpha \left( \log \pi_{\phi}(a_t|s_t) + \overline{\mathcal{H}} \right) \right]
$$

计算梯度：

$$
\nabla_{\alpha} J(\alpha) = E_{s_t \sim D, a_t \sim \pi_{\tau}} \left[ - \left( \log \pi_{\phi}(a_t|s_t) + \overline{\mathcal{H}} \right) \right] = \mathcal{H}(\pi_{\phi}) - \overline{\mathcal{H}}
$$

优化 $\tilde{\alpha} = \log \alpha$：

$$
\nabla_{\tilde{\alpha}} J(\tilde{\alpha}) = -\alpha E_{s_t \sim D, a_t \sim \pi_{\tau}} \left[ \log \pi_{\phi}(a_t|s_t) + \overline{\mathcal{H}} \right]
$$

更新公式：

$$
\tilde{\alpha} \leftarrow \tilde{\alpha} - \eta_{\alpha} \nabla_{\tilde{\alpha}} J(\tilde{\alpha}) = -\tilde{\alpha} + \eta_{\alpha} \alpha E_{s_t \sim D, a_t \sim \pi_{\tau}} \left[ \log \pi_{\phi}(a_t|s_t) + \overline{\mathcal{H}} \right]
$$

关于 $\alpha$ 的角色：

- 虽然优化 $J(\alpha)$ 借鉴了拉格朗日乘子法，$\alpha$ 更像是温度参数，而非严格的拉格朗日乘子，因为 SAC 使用软约束（$\mathcal{H}(\pi_{\phi}) \approx \overline{\mathcal{H}}$）而非硬约束。
- $\alpha$ 作为温度参数，出现在哪些策略优化的指标分布中：

$$
\pi^*(a_t|s_t) \propto \exp \left( Q_{\theta}(s_t, a_t) / \alpha \right)
$$

通过 KL 散度最小化，策略目标为：

$$
J_{\pi}(\phi) = E_{s_t \sim D} \left[ D_{KL} \left( \pi(\cdot|s_t) \parallel \frac{\exp(Q_{\theta}(s_t, \cdot)/\alpha)}{Z(s_t)} \right) \right]
$$


$\alpha$ 控制目标分布的平滑性，动态调整确保策略随时间迁移。

## 4.4 训练的网络（核心）

SAC 训练以下网络：

### 4.4.1 策略网络 (Actor)

   - 功能：输出动作分布的参数（如高斯分布的均值和方差）。
   - 参数：$\phi$。
   - 更新方式：通过梯度上升最大化 $J_{\phi}(\phi)$，使用重参数化技巧：

$$
   \nabla_{\phi} J_{\phi}(\phi) \approx \nabla_{\phi} \left[ Q_{\theta}(s_t, f_{\phi}(\epsilon_t, s_t)) - \alpha \log \pi_{\phi}(f_{\phi}(\epsilon_t, s_t) | s_t) \right]
$$

### 4.4.2 两个 Q 网络 (Critic)：

   - 功能：估计状态-动作对的 Q 值，分别为 $Q_{\theta_1}$, $Q_{\theta_2}$。
   - 参数：$\theta_1, \theta_2$。
   - 更新方式：通过梯度下降最小化 $J_Q(\theta_i)$：

$$
   \nabla_{\theta_i} J_Q(\theta_i) = \nabla_{\theta_i} \left[ Q_{\theta}(s_t, a_t) - y_t \right]^2
$$

### 4.4.3 两个目标 Q 网络 (Target Critic)

   - 功能：提供稳定的 Q 值估计。
   - 参数：$\theta_1, \theta_2$。
   - 更新方式：软更新：



软更新的标准形式应为：

$$ \theta_{\text{target}, i} \leftarrow \tau \theta_{i}+(1-\tau) \theta_{\text{target}, i} \quad(i=1,2) $$

其中 $\tau$ 是更新系统的平滑因子（如 0.005）。

#### 4.4.3.1 软更新（Polyak Averaging）

在SAC中，目标网络的参数 $\bar{\theta}$ 通过主Q网络参数 $\theta$ 的指数移动平均更新：

$$ \bar{\theta} \leftarrow \tau\theta + (1 - \tau)\bar{\theta} $$

其中：
- $\tau \in (0, 1)$ 是平滑系数（通常很小，如0.005）。
- 不涉及梯度计算，直接按权重混合参数。

特点：
- 每一步都轻微调整目标网络，训练更稳定。
- 是SAC、DDPG等算法的默认选择。

### 4.4.4 温度参数a

- 功能：动态调整 $\alpha$ 以控制熵。
- 参数：$\hat{\alpha} = \log \alpha$。 从而保证a为正数
- 更新方式：通过梯度下降最小化 $J(\alpha)$：

$$
\hat{\alpha} \leftarrow \hat{\alpha} + \eta_\alpha \alpha \mathbb{E}_{s_t \sim D, a_t \sim \pi_{\phi}} \left[ \log \pi_{\phi}(a_t | s_t) + \hat{H} \right]
$$

### 4.4.5 参数更新流程

1. 采样数据：从经验回放池 $D$ 采样 $(s_t, a_t, r_t, s_{t+1})$。
2. 更新 Q 网络：计算目标值 $y_t$，最小化 $J_Q(\theta_1), J_Q(\theta_2)$。
3. 更新策略网络：使用重参数化技巧，最大化 $J_{\pi}(\phi)$。
4. 更新目标 Q 网络：软更新 $\theta_1, \theta_2$。
5. 更新温度参数：最小化 $J(\alpha)$，调整 $\hat{\alpha}$。
### 4.4.6 α 的深入分析

- $\alpha$ 更新公式的推导：

$$
目标函数 $J(\alpha) = \mathbb{E} \left[ - \alpha \left( \log \pi_{\phi}(a_t | s_t) + \hat{H} \right) \right] \text{来自于约束} H(\pi_{\phi}) \geq \hat{H}
$$

- 梯度计算：

$$
\nabla_{\alpha} J(\alpha) = \mathbb{E} \left[ - (\log \pi_{\phi}(a_t | s_t) + \hat{H}) \right] = H(\pi_{\phi}) - \hat{H}
$$



- **优化公式 $\hat{\alpha} = \log \alpha$ 确保 $\alpha > 0$**，更新公式正确反映了映射偏差。

- 负号符号来自于 $-\alpha$，推导表明，负号来自 $\log \pi_{\phi}(a_t | s_t) + \hat{H}$ 保持符号，因为目标使得 $H(\pi_{\phi}) = \mathbb{E}[- \log \pi_{\phi}(a_t | s_t)] \approx \hat{H}$。

#### 4.4.6.1 $\alpha$ 的统计学性质：
- SAC 通过引入拉格朗日乘子构造 $J(\alpha)$，但 $\alpha$ 不是严格的拉格朗日目标子，因为 SAC 使用软约束（$H(\pi_{\phi}) \geq \hat{H}$）而非硬约束。

- $\alpha$ 作为温度参数，控制策略分布的随机性，出现在指示分布 $\pi^*(a_t | s_t) \propto \exp(Q_{\theta}(s_t, a_t) / \alpha)$。其动态调整通过 $J(\alpha)$ 实现，类似积分方法。

#### 4.4.6.2 $\alpha$ 与 KL 散度的关系：
- $\alpha$ 被用于控制最小目标分布 $ \exp(Q_{\theta}(s_t, a_t) / \alpha) / Z(s_t) $，控制分布平滑性。

- 优化 $J(\alpha)$ 确保 $H(\pi_{\phi}) \approx \hat{H}$，间接影响 KL 散度中的目标分布。

## 4.5 总结

**目标函数**
- 策略网络：最大化 $J_{\pi}(\phi) = \mathbb{E} \left[ Q_{\theta}(s_t, a_t) - \alpha \log \pi_{\phi}(a_t | s_t) \right]$。
- Q 网络：最小化 $J_Q(\theta) = \mathbb{E} \left[ Q_{\theta}(s_t, a_t) - y_t \right]^2$。
- 温度参数：最小化 $J(\alpha) = \mathbb{E} \left[ -\alpha \log \pi_{\phi}(a_t | s_t) + \hat{H} \right]$。

**训练网络**
- 一个策略网络，两个 Q 网络，两个目标 Q 网络，温度参数 $\alpha$。

**参数更新**
- Q 网络和目标 Q 网络通过梯度下降更新，策略网络通过梯度上升更新，目标 Q 网络通过软更新。

- $\alpha$ 作为温度参数，控制策略随机性，自动调节目标 $H(\pi_{\phi}) \approx \hat{H}$，控制 KL 散度中目标分布的平滑度。



# 5.实现中的近似处理

实际应用中：

1.避免计算$ Z(s) $：通过梯度下降直接优化参数化策略

2.重参数化技巧：对高斯策略$ \pi_{\theta}(a|s)=\mathrm{N}\left(\mu_{\theta}(s),\sigma_{\theta}(s)\right) $，采用：

$$ a=\mu_{\theta}(s)+\sigma_{\theta}(s)\cdot\epsilon,\quad\epsilon\sim\mathrm{N}(0,1) $$

3.双Q网络：防止过估计，使用两个Q函数取最小值：

$$ Q=\min\left(Q_{\phi_{1}}, Q_{\phi_{2}}\right) $$

## 5.1完整推导链路

<img src="assets/image-20250709113158549.png" width="300">

## 5.2关键点总结

1.变分推断的作用：将策略优化转化为可计算的分布匹配问题

2.温度系数$ \alpha $：自动调整探索-利用权衡（可通过梯度下降优化）

3.与传统PG的区别：

·普通策略梯度：$ \nabla_{\theta} J\propto\mathrm{E}[Q\nabla\log\pi] $

·SAC策略梯度：$ \nabla_{\theta} J\propto\mathrm{E}[(Q-\alpha\log\pi)\nabla\log\pi] $

这种基于变分推断的方法使得SAC既能处理连续动作空间，又能保持策略的探索能力，是其优于传统策略梯度方法的核心所在。
