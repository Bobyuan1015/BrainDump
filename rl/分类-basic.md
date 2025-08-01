# 1、强化学习的目标

强化学习的目标是通过学习一个策略（policy）或价值函数（value function），使智能体在环境中获得的期望累计奖励最大化。常见的优化目标包括：

1. **策略优化目标**：直接优化策略 $\pi(a|s)$，使期望累积奖励 $J(\pi)$ 最大化。
2. **价值函数优化目标**：优化状态价值函数 $V(s)$ 或动作价值函数 $Q(s, a)$，通过价值迭代或策略评估实现。
3. **混合目标**：结合策略和价值函数，例如在Actor-Critic方法中同时优化策略和价值估计。




## 1.1 状态价值函数（V函数）

**定义**：在策略 $\pi$ 下，从状态 $s_t$ 开始获得的长期期望回报。

$$
V^{\pi}(s_t)
=
\mathbb{E}_{\tau \sim \pi}
\left[
\sum_{k=0}^{\infty} \gamma^{k} r_{t+k}
\mid s_t
\right]
$$

---

## 1.2 动作价值函数（Q函数）

**定义**：在策略 $\pi$ 下，从状态 $s_t$ 执行动作 $a_t$ 后，获得的长期期望回报。

$$
Q^{\pi}(s_t, a_t)
=
\mathbb{E}_{\tau \sim \pi}
\left[
\sum_{k=0}^{\infty} \gamma^{k} r_{t+k}
\mid s_t, a_t
\right]
$$

---

## 1.3 V函数与Q函数的比较

| 特性     | V函数 $V^{\pi}(s_t)$ | Q函数 $Q^{\pi}(s_t, a_t)$ |
| -------- | --------------------- | -------------------------- |
| **输入变量** | 状态 $s_t$ | 状态 $s_t$ + 动作 $a_t$ |
| **评估对象** | 策略 $\pi$ 在状态 $s_t$ 的长期价值 | 策略 $\pi$ 下执行 $a_t$ 后的长期价值 |
| **关系式** | $ V^{\pi}(s_t) = \mathbb{E}_{a_t \sim \pi} \left[ Q^{\pi}(s_t, a_t) \right] $ | $ Q^{\pi}(s_t, a_t) = r_t + \gamma \mathbb{E}_{s_{t+1}} \left[ V^{\pi}(s_{t+1}) \right] $ |

---

## 1.4 策略梯度定理推导

### 1.4.1 目标函数定义

定义策略参数 $\theta$ 下的目标函数为：

$$
J(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
\sum_{t=0}^{\infty} \gamma^t r_t
\right]
$$

其中，轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots)$ 服从策略 $\pi_{\theta}$。

---

### 1.4.2 梯度形式展开

根据期望定义：

$$
\nabla_{\theta} J(\theta)
=
\nabla_{\theta}
\int P(\tau | \theta) R(\tau) d\tau
$$

其中：

- $P(\tau | \theta)$：轨迹概率
- $R(\tau)$：轨迹回报

---

### 1.4.3 使用对数技巧（Likelihood Ratio Trick）

由于：

$$
\nabla_{\theta} P(\tau | \theta)
=
P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta)
$$

因此有：

$$
\nabla_{\theta} J(\theta)
=
\int P(\tau | \theta)
\nabla_{\theta} \log P(\tau | \theta)
R(\tau) d\tau
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
\nabla_{\theta} \log P(\tau | \theta)
R(\tau)
\right]
$$

---

### 1.4.4 轨迹概率的分解

轨迹概率可分解为：

$$
P(\tau | \theta)
=
p(s_0)
\prod_{t=0}^{\infty}
\pi_{\theta}(a_t | s_t)
p(s_{t+1} | s_t, a_t)
$$

其中：

- $p(s_0)$: 初始状态分布
- $\pi_{\theta}(a_t | s_t)$: 策略选择动作概率
- $p(s_{t+1} | s_t, a_t)$: 环境状态转移概率

---

取对数：

$$
\log P(\tau | \theta)
=
\log p(s_0)
+
\sum_{t=0}^{\infty}
\left[
\log \pi_{\theta}(a_t | s_t)
+
\log p(s_{t+1} | s_t, a_t)
\right]
$$

---

由于：

- $\log p(s_0)$ 与参数 $\theta$ 无关
- $\log p(s_{t+1} | s_t, a_t)$ 也与参数 $\theta$ 无关

所以：

$$
\nabla_{\theta} \log P(\tau | \theta)
=
\sum_{t=0}^{\infty}
\nabla_{\theta}
\log \pi_{\theta}(a_t | s_t)
$$

---

### 1.4.5 将梯度代入目标函数梯度表达式

因此：

$$
\nabla_{\theta} J(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
R(\tau)
\sum_{t=0}^{\infty}
\nabla_{\theta}
\log \pi_{\theta}(a_t | s_t)
\right]
$$

---

### 1.4.6 引入折扣因子

由于：

$$
R(\tau)
=
\sum_{t=0}^{\infty}
\gamma^t r_t
$$

代入上式：

$$
\nabla_{\theta} J(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
\left(
\sum_{t=0}^{\infty}
\gamma^t r_t
\right)
\left(
\sum_{k=0}^{\infty}
\nabla_{\theta}
\log \pi_{\theta}(a_k | s_k)
\right)
\right]
$$

---

### 1.4.7 引入因果关系（Causality Correction）

由于动作 $a_k$ 只影响 $t \ge k$ 的奖励，可将求和顺序调整为：

$$
\nabla_{\theta} J(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
\sum_{k=0}^{\infty}
\nabla_{\theta}
\log \pi_{\theta}(a_k | s_k)
\left(
\sum_{t=k}^{\infty}
\gamma^t r_t
\right)
\right]
$$

---

### 1.4.8 定义动作价值函数 Q

定义：

$$
Q^{\pi}(s_k, a_k)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
\sum_{t=k}^{\infty}
\gamma^t r_t
\mid s_k, a_k
\right]
$$

---

### 1.4.9 策略梯度定理最终形式

将上述表示为 Q 函数，得到 **策略梯度定理**：

$$
\nabla_{\theta} J(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
\sum_{t=0}^{\infty}
\nabla_{\theta}
\log \pi_{\theta}(a_t | s_t)
Q^{\pi}(s_t, a_t)
\right]
$$

---


# 2、求导的分类

在强化学习中，求导主要用于优化目标函数，根据方法的不同，求导可以分为以下几类：

## 2.1 策略梯度方法（Policy Gradient Methods）

策略梯度方法直接对策略的参数进行优化，目标是最大化期望累积奖励 $J(\theta)$，其中 $\theta$ 是策略 $\pi_{\theta}(a|s)$ 的参数。

- **目标函数**：

  $$
  J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
  $$

  其中，$r_t$ 是时间步 $t$ 的奖励，$\gamma \in [0,1)$ 是折扣因子。

- **求导公式**（策略梯度定理）：

  $$
  \nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{\infty} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) Q^{\pi_{\theta}}(s_t, a_t) \right]
  $$

  其中，$Q^{\pi_{\theta}}(s_t, a_t)$ 是动作价值函数，表示在状态 $s_t$ 采取动作 $a_t$ 后的期望累积奖励。

- **说明：**
  
  - 策略梯度通过对策略的对数概率求导，利用动作价值函数 $Q$ 来调整策略。
  - 实际中，常用蒙特卡洛采样或时序差分（TD）方法估计 $Q$ 值。
  - 常见算法：REINFORCE、TRPO、PPO。

| 算法      | 改进点                | 优化方式           | 适用场景                         |
| --------- | --------------------- | ------------------ | -------------------------------- |
| REINFORCE | 引入基线减方差        | 蒙特卡洛梯度上升   | 简单任务，理论验证               |
| TRPO      | 信任域约束+二阶优化   | 带约束的优化问题   | 高维连续动作空间（如机器人控制） |
| PPO       | 裁剪目标函数+一阶优化 | 无约束优化（Adam） | 通用场景，工业级应用             |

## 2.2 价值函数优化（Value-Based Methods）

价值函数优化的目标是最小化价值函数的预测误差，常见于Q-learning或SARSA等方法。

- **目标函数**：对于Q-learning，目标是最小化时序差分误差（TD error）：

  $$
  L(\theta) = \mathbb{E} \left[ \left( r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta) - Q(s_t, a_t; \theta) \right)^2 \right]
  $$

  其中，$Q(s, a; \theta)$ 是参数化的Q函数，$r_t$ 是即时奖励，$\gamma$ 是折扣因子。

- **求导公式**：

  $$
  \nabla_{\theta} L(\theta) = \mathbb{E} \left[ 2 \left( r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta) - Q(s_t, a_t; \theta) \right) \nabla_{\theta} Q(s_t, a_t; \theta) \right]
  $$

- **说明：**
  - 求导的目的是调整参数 $\theta$，使Q函数逼近真实的动作价值。
  - 常用于深度Q网络（DQN），通过神经网络参数化 $Q(s, a; \theta)$。

## 2.3 Actor-Critic方法

Actor-Critic方法结合策略梯度和价值函数优化，分为Actor（策略）和Critic（价值函数）两部分。

- **Actor的目标函数**（与策略梯度相同）：

  $$
  J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
  $$

  求导公式：

  $$
  \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) A(s_t, a_t) \right]
  $$

  其中，$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$ 是优势函数（Advantage Function），用于降低方差。

- **Critic的目标函数**：Critic的目标是最小化价值函数的预测误差：

  $$
  L(\phi) = \mathbb{E} \left[ \left( r_t + \gamma V(s_{t+1}; \phi) - V(s_t; \phi) \right)^2 \right]
  $$

  求导公式：

  $$
  \nabla_\phi L(\phi) = \mathbb{E} \left[ 2 \left( r_t + \gamma V(s_{t+1}; \phi) - V(s_t; \phi) \right) \nabla_\phi V(s_t; \phi) \right]
  $$

- **说明：**
  - Actor优化策略，Critic优化价值函数，二者相互配合。
  - 常见算法：A2C、A3C、DDPG、SAC。















# 3. 各算法Actor优化目标分析

## 3.1 DDPG（Deep Deterministic Policy Gradient）

- **Actor优化目标**：最大化Q值。

  - DDPG使用**确定性策略** $$ \mu(s|\theta^\mu) $$，Actor直接输出动作$$ a = \mu(s|\theta^\mu) $$。

  - 优化目标是：
    $$
    J(\theta^\mu) = \mathbb{E}_{s \sim \rho^\mu} \left[ Q(s, \mu(s|\theta^\mu)|\theta^Q) \right]
    $$

  - 梯度为：
    $$
    \nabla_{\theta^\mu} J \approx \mathbb{E}_{s \sim \rho^\mu} \left[ \left. \nabla_a Q(s, a|\theta^Q) \right|_{a = \mu(s|\theta^\mu)} \nabla_{\theta^\mu} \mu(s|\theta^\mu) \right]
    $$

  - **特点**：Actor直接通过Critic的Q值梯度优化策略，目标是使输出的动作**最大化Q值**。

  - **是否最大化Q值**：是，直接最大化$$ Q(s, a) $$。
  
  

---

## 3.2 Actor-Critic（经典Actor-Critic）

- **Actor优化目标**：最大化期望回报，通常通过Q值或优势函数。

  - 经典Actor-Critic使用**随机策略** $$ \pi(a|s, \theta) $$，输出动作的概率分布。

  - 优化目标是：
    $$
    J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} \left[ Q(s, a) \right]
    $$

  - 梯度为：
    $$
    \nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} \left[ \nabla_\theta \log \pi(a|s, \theta) Q(s, a) \right]
    $$

  - **特点**：Actor通过对动作概率分布的log似然加权Q值进行优化，**间接最大化Q值**。相比DDPG，经典Actor-Critic的随机策略需要采样动作，优化过程基于概率分布。

  - **是否最大化Q值**：是，但通过对动作行为加权Q值实现，而不是直接优化确定动作的Q值。

  - **差异**：DDPG直接优化确定性动作的Q值，而经典Actor-Critic通过调整动作分布优化期望Q值。
  
  

## 3.3 A2C（Advantage Actor-Critic）

- **Actor优化目标**：最大化优势函数（Advantage Function）。

  - A2C是经典Actor-Critic的改进，使用**随机策略** $$ \pi(a|s, \theta) $$，并引入优势函数 $$ A(s, a) = Q(s, a) - V(s) $$ 来降低梯度方差。

  - 优化目标是：
    $$
    J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} \left[ \log \pi(a|s, \theta) A(s, a) \right]
    $$

  - 梯度为：
    $$
    \nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} \left[ \nabla_\theta \log \pi(a|s, \theta) A(s, a) \right]
    $$

  - **特点**：Actor优化的是优势函数，而不是直接的Q值。优势函数表示动作相对于平均值的优势，降低了梯度估计的方差。

  - **是否最大化Q值**：间接最大化Q值，通过优势函数 $$ A(s, a) $$ 优化策略，但目标不是直接最大化 $$ Q(s, a) $$，而是调整动作分布以提高期望回报。

  - **差异**：A2C使用优势函数而非直接Q值，适用于随机策略，且更适合离散或连续动作空间。
  
  

---

## 3.4 SAC（Soft Actor-Critic）

- **Actor优化目标**：最大化期望回报与熵的组合。

  - SAC使用**随机策略** $$ \pi(a|s, \theta) $$，并引入熵正则化以鼓励探索。

  - 优化目标是：
    $$
    J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} \left[ Q(s, a) + \alpha \mathcal{H}(\pi(\cdot|s)) \right]
    $$

    其中，$$ \mathcal{H}(\pi) = \mathbb{E}_{a \sim \pi}[-\log \pi(a|s)] $$ 是策略的熵，$$ \alpha $$ 是熵正则化系数。

  - 梯度为：
    $$
    \nabla_\theta J(\theta) \approx \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} \left[ \nabla_\theta \log \pi(a|s, \theta) \left( Q(s, a) - \alpha \log \pi(a|s, \theta) \right) \right]
    $$

  - **特点**：SAC不仅最大化Q值，还通过熵项鼓励策略的多样性（探索）。这与DDPG的确定性策略不同，SAC的随机策略通过熵正则化平衡探索与利用。

- **是否最大化Q值**：部分是，最大化Q值的同时考虑熵正则化，目标是 $$ Q(s, a) + \alpha \mathcal{H} $$，而非单纯的Q值。

- **差异**：SAC的Actor优化目标包含熵项、鼓励探索，而DDPG通过外加噪声实现探索。



---

## 3.5 PPO（Proximal Policy Optimization）

- **Actor优化目标**：最大化剪切比率加权的优势函数。

  - PPO使用**随机策略** $$ \pi(a|s, \theta) $$，通过剪切比率（clipped ratio）限制策略更新，增强稳定性。

  - 优化目标是：
    $$
    J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} \left[ \min\left( \frac{\pi(a|s, \theta)}{\pi_{\text{old}}(a|s)} A(s, a), \; \text{clip}\left( \frac{\pi(a|s, \theta)}{\pi_{\text{old}}(a|s)}, 1 - \epsilon, 1 + \epsilon \right) A(s, a) \right) \right]
    $$

  - **特点**：PPO通过剪切比率限制策略更新，优化的是优势函数 $$ A(s, a) $$，而不是直接Q值。剪切机制确保策略不会变化过大，提高训练稳定性。

  - **是否最大化Q值**：间接通过优势函数优化期望回报，但不直接最大化 $$ Q(s, a) $$。

  - **差异**：PPO关注优势函数和策略稳定，而DDPG直接优化确定性动作的Q值。
  
  

---

## 3.6 TRPO（Trust Region Policy Optimization）

- **Actor优化目标**：最大化约束下的期望优势函数。

  - TRPO使用**随机策略** $$ \pi(a|s, \theta) $$，通过信任区域（trust region）约束策略更新，优化目标为：
    $$
    J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} \left[ \frac{\pi(a|s, \theta)}{\pi_{\text{old}}(a|s)} A(s, a) \right]
    $$

    受限于KL散度约束：
    $$
    \mathbb{E}_{s \sim \rho^\pi} \left[ \text{KL}(\pi_{\text{old}}(\cdot|s) \| \pi(\cdot|s, \theta)) \right] \leq \delta
    $$

  - **特点**：TRPO通过KL散度限制策略更新幅度，优化优势函数而非直接Q值，确保策略更新在信任区域内。

  - **是否最大化Q值**：间接通过优势函数优化期望回报，但不直接最大化 $$ Q(s, a) $$。

  - **差异**：TRPO使用信任区约束和优势函数，区别于DDPG的确定性策略更注重策略的稳定性。
  
  

## 3.7 TD3（Twin Delayed Deep Deterministic Policy Gradient）

- **Actor优化目标**：最大化Q值（与DDPG相同）。

  - TD3是DDPG的改进，依然使用**确定性策略** $$ \mu(s|\theta^\mu) $$，优化目标与DDPG相同：

    $$
    J(\theta^\mu) = \mathbb{E}_{s \sim \rho^\mu} \left[ Q(s, \mu(s|\theta^\mu)|\theta^Q) \right]
    $$

  - 梯度为：

    $$
    \nabla_{\theta^\mu} J \approx \mathbb{E}_{s \sim \rho^\mu} \left[ \left. \nabla_a Q(s, a|\theta^Q) \right|_{a = \mu(s|\theta^\mu)} \nabla_{\theta^\mu} \mu(s|\theta^\mu) \right]
    $$

- **特点**：TD3通过双Q网络（Twin Q-Networks）、延迟更新（Delayed Update）和目标策略平滑（Target Policy Smoothing）改进DDPG，但Actor的优化目标与DDPG一致。

- **是否最大化Q值**：是，与DDPG完全相同，直接最大化 $$ Q(s, a) $$。

- **差异**：TD3与DDPG的Actor优化目标相同，但通过改进Critic和更新策略提高稳定性。



### 3. 比较总结

| 算法         | 策略类型 | Actor优化目标                                | 是否直接最大化Q值 | 关键差异                                             |
| ------------ | -------- | -------------------------------------------- | ----------------- | ---------------------------------------------------- |
| DDPG         | 确定性   | 最大化 $$ Q(s, \mu(s)) $$                    | 是                | 直接优化确定性动作的Q值，探索靠外加噪声。            |
| Actor-Critic | 随机性   | 最大化 $$ \mathbb{E}_{\sim \pi} [Q(s, a)] $$ | 是（间接）        | 通过动作分布加权Q值优化，适合离散或连续动作空间。    |
| A2C          | 随机性   | 最大化 $$ \mathbb{E}_{\sim \pi} [A(s, a)] $$ | 否（优势函数）    | 使用优势函数降低方差，适合随机策略。                 |
| SAC          | 随机性   | 最大化 $$ Q(s, a) + \alpha \mathcal{H} $$    | 否（Q值+熵）      | 引入熵正则化鼓励探索，优化目标平衡回报和策略多样性。 |
| PPO          | 随机性   | 最大化剪切比率加权的 $$ A(s, a) $$           | 否（优势函数）    | 通过剪切比率限制更新幅度，注重稳定性。               |
| TRPO         | 随机性   | 最大化约束下的 $$ A(s, a) $$                 | 否（优势函数）    | 使用KL散度约束信任区域，优化稳定性。                 |
| TD3          | 确定性   | 最大化 $$ Q(s, \mu(s)) $$                    | 是                | 与DDPG相同，但通过双Q网络和延迟更新提高稳定性。      |





# 4.TD基于Q、V、对比



**TD（Temporal Difference）Error** 的核心思想是：

> 当前的估值与基于下一状态的估值之间的差异。

它是一种 **Bootstrapping** 机制。

- TD Error 是 Bellman 残差的估计。

- 基于 V 还是 Q，取决于算法估计的是哪个函数。

- 两者形式类似，但含义不同：
  - V函数：状态价值，估计"状态好不好"
  - Q函数：状态+动作的价值，估计"在某状态下，某动作好不好"

- 在 Actor-Critic 中使用 $$ \delta_V $$，是为了获得近似 Advantage。



| 方法                                      | 使用的TD error                               | 使用的值函数           |
| ----------------------------------------- | -------------------------------------------- | ---------------------- |
| Q-Learning / DQN                          | $$ \delta = Q_{\text{target}} - Q $$         | Q函数                  |
| SARSA                                     | $$ \delta = r + \gamma Q(s', a') - Q(s,a) $$ | Q函数                  |
| Actor-Critic                              | $$ \delta = r + \gamma V(s') - V(s) $$       | V函数                  |
| Advantage Actor-Critic (A2C)              | $$ \delta = r + \gamma V(s') - V(s) $$       | V函数（用于advantage） |
| Deep Deterministic Policy Gradient (DDPG) | Critic 用Q函数更新，Actor用Q的梯度           | Q函数                  |

| 使用                           | 使用 TD error 形式                                  | 说明                            |
| ------------------------------ | --------------------------------------------------- | ------------------------------- |
| Policy Gradient (Actor-Critic) | $$\delta = r + \gamma V(s') - V(s)$$                | Critic估计V，TD error≈Advantage |
| DQN / Q-learning               | $$\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a)$$ | 目标是直接学习Q                 |
| DDPG / SAC / TD3               | $$\delta = r + \gamma Q(s',\pi(s')) - Q(s,a)$$      | 使用目标策略获得下一步动作      |

