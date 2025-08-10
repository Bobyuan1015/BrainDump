# 1.TD3论文

- **文题目**：Addressing Function Approximation Error in Actor-Critic Methods
- **作者**：Scott Fujimoto, Herke van Hoof, David Meger
- **年份**：2018
- **会议/期刊**：International Conference on Machine Learning (ICML)
- **链接**：https://arxiv.org/abs/1802.09477


TD3的成功启发了后续改进算法，以下是几个代表性算法：

1. **SAC (Soft Actor-Critic)**：
   - 引入熵正则化，鼓励探索，适用于连续动作空间。
   - 与TD3的区别：SAC是随机策略（输出动作分布），而TD3是确定性策略。





## 1.1 背景

强化学习(Reinforcement Learning, RL)是一种通过与环境交互来学习最优策略的机器学习方法，广泛应用于机器人控制、游戏AI等领域。在深度强化学习(Deep RL)兴起后，结合深度神经网络的算法如DQN(Deep Q-Network)在离散动作空间取得了成功，但对于连续动作空间(如机器人控制中的角度或速度)，DQN表现不佳。

为此，Deep Deterministic Policy Gradient (DDPG)算法被提出，作为一种基于Actor-Critic架构的深度强化学习算法，适用于连续动作空间。DDPG结合了确定性策略梯度(DPG)和深度神经网络，能够处理高维连续动作空间。然而，DDPG存在以下问题：

- Q值过估计：由于函数逼近误差，DDPG中的Q函数(价值函数)容易高估动作价值，导致次优策略被选择。
- 训练不稳定：DDPG对超参数敏感，训练过程容易发散。
- 探索不足：DDPG的噪声添加方式可能导致探索效率低下。

## 1.2 动机

TD3(Twin Delayed Deep Deterministic Policy Gradient)由Fujimoto等人在2018年提出，旨在解决DDPG的上述问题。TD3的动机是提高连续动作空间强化学习算法的稳定性和性能，通过以下方式：

1. 减少Q值过估计问题，提升价值估计的准确性。
2. 提高训练稳定性，使算法对超参数不敏感。
3. 优化探索策略，平衡探索与利用。



## 1.3 解决的问题及其创新点

**要解决的问题**

- **Q值过估计**：DDPG中的Q函数由于最大化操作(max操作)容易高估动作价值，导致策略学习偏向次优解。

- **训练不稳定性**：DDPG的训练过程对超参数(如学习率、噪声参数)敏感，容易导致策略发散或性能波动。

- **探索与利用的平衡**：DDPG的噪声添加方式(如Ornstein-Uhlenbeck噪声)可能导致探索不足，陷入局部最优。

**创新点**

TD3在DDPG基础上引入了三项关键改进，统称为"三重技巧":

- **Clipped Double Q-Learning(双重Q学习剪切)**:
   - 使用两个Q函数(Twin Q-Networks)来估计价值，取较小的Q值作为目标，减少过估计偏差。

- **Delayed Policy Updates(延迟策略更新)**:
   - 降低策略网络的更新频率(通常每两次Critic更新后更新一次Actor)，使Q函数更稳定，减少策略震荡。

- **Target Policy Smoothing(目标策略平滑化)**:
   - 在目标动作上添加噪声，平滑目标Q值的估计，防止策略过度拟合尖锐的Q值峰值。

这些创新点使TD3在性能和稳定性上显著优于DDPG，尤其在高维连续动作空间任务中。





# 2.数学原理与推导过程

**强化学习基础**

在强化学习中，智能体（Agent）与环境（Environment）交互，目标是最大化累积期望回报。环境建尔可夫决策过程（MDP），定义为五元组$(S,A,P,R,\gamma)$，其中：

- $S$：状态空间。
- $A$：动作空间（TD3处理连续动作空间）。
- $P(s'|s,a)$：状态转移概率。
- $R(s,a)$：奖励函数。
- $\gamma\in[0,1]$：折扣因子。

智能体的策略$\pi:S\rightarrow A$是一个从状态到动作的映射，目标是最大化期望累积回报：

$$J(\pi)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^{t}R\left(s_{t},a_{t}\right)\right]$$

其中，$a_{t}\sim\pi(s_{t})$，$s_{t+1}\sim P(\cdot|s_{t},a_{t})$。

**Q函数与Bellman方程**

Q函数（动作价值函数）表示在状态s下执行动作a，然后按策略π行动的期望回报：

$$Q^{\pi}(s,a)=\mathbb{E}_{\pi}[R(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim P}[Q^{\pi}(s^{\prime},\pi(s^{\prime}))]]$$

DDPG和TD3基于Actor-Critic框架：

- Actor：策略网络$\pi_{\phi}(s)$输出确定性动作。
- Critic：Q网络$Q_{\theta}(s,a)$估计动作价值。



## 2.1 数学原理

TD3在DDPG基础上引入了三项改进，下面逐一推导其数学原理。

### 2.1.1 Clipped Double Q-Learning

DDPG中的Q值更新基于Bellman方程：

$$ y=r+\gamma Q_{\theta}\left(s^{\prime},\pi_{\phi}\left(s^{\prime}\right)\right) $$

其中，y是目标Q值，r是即时奖励，$s^{\prime}$是下一状态。然而，由于神经网络的函数逼近误差，$\max_{a}Q_{\theta}\left(s^{\prime}, a\right)$容易过估计。

TD3引入两个Q网络$Q_{\theta_{1}}$和$Q_{\theta_{2}}$，分别计算目标Q值，并取较小值：

$$ y=r+\gamma\min\left(Q_{\theta_{1}}\left(s^{\prime},\pi_{\phi}\left(s^{\prime}\right)\right), Q_{\theta_{2}}\left(s^{\prime},\pi_{\phi}\left(s^{\prime}\right)\right)\right) $$

#### 2.1.1 .1推导过程：

- 假设单一Q函数$Q_{\theta}$存在估计误差$\epsilon$，即$Q_{\theta}(s,a)=Q^{\pi}(s,a)+\epsilon$。在DDPG中，最大化操作会导致$\epsilon$累积，产生过估计。

- 使用两个独立的Q网络，误差$\epsilon_{1}$和$\epsilon_{2}$相互独立，取最小值可有效降低正偏差：

$$ \min\left(Q_{\theta_{1}}, Q_{\theta_{2}}\right) \approx Q^{\pi}+\min\left(\epsilon_{1}, \epsilon_{2}\right) $$

因为$\min\left(\epsilon_{1}, \epsilon_{2}\right)$通常比单一$\epsilon$小，过估计问题得到缓解。

### 2.1.2 Delayed Policy Updates

DDPG中，Actor和Critic同时更新，可能导致Q函数未收敛时策略更新过快，引起震荡。TD3通过延迟策略更新解决此问题：

- Critic网络每更新d次（如$d=2$），Actor网络更新一次。

- 数学上，这确保$Q_{\theta_{1}}$和$Q_{\theta_{2}}$在更新策略时更接近真实Q值，减少误差传播。

### 2.1.3 Target Policy Smoothing

为了避免Q函数对尖锐峰值（高方差动作）的过度拟合，TD3在目标动作上添加噪声：

$$ a^{\prime}=\pi_{\phi}(s^{\prime})+\epsilon,\quad\epsilon\sim\operatorname{clip}(\mathcal{N}(0,\sigma),-c, c) $$

目标Q值变为：

$$ y=r+\gamma\min\left(Q_{\theta_{1}}\left(s^{\prime}, a^{\prime}\right), Q_{\theta_{2}}\left(s^{\prime}, a^{\prime}\right)\right) $$

推导过程：

- 尖锐的Q值峰值可能导致策略过度拟合某一动作。添加噪声$\epsilon$相当于对目标动作进行平滑，类似正则化：

$$ Q_{\text{smoothed}}\left(s^{\prime}, a\right)\approx\mathbb{E}_{\epsilon}\left[Q\left(s^{\prime},\pi_{\phi}\left(s^{\prime}\right)+\epsilon\right)\right] $$

- 这使得Q值更平滑，策略对小扰动更鲁棒。

**变量含义**

- $s,s^{\prime}$：当前状态和下一状态。
- $a,a^{\prime}$：当前动作和目标动作。
- $r$：即时奖励。
- $\gamma$：折扣因子。
- $Q_{\theta_{1}}, Q_{\theta_{2}}$：两个Critic网络，参数分别为$\theta_{1},\theta_{2}$。
- $\pi_{\phi}$：Actor网络，参数为$\phi$。
- $\epsilon$：目标策略平滑化的噪声，服从截断正态分布。
- $\sigma,c$：噪声的标准差和截断范围。





## 2.2 目标函数、梯度推导与训练过程



### 2.2.1 Critic目标函数

Critic网络通过最小化时间差分误差(TD Error)更新:

$$ L\left(\theta_{i}\right)=\mathbb{E}_{\left(s, a,r,s^{\prime}\right)\sim\mathcal{D}}\left[\left(Q_{\theta_{i}}(s, a)-y\right)^{2}\right],\quad i=1,2 $$

其中，目标值y为：

$$ y=r+\gamma\min\left(Q_{\theta_{1}^{\prime}}\left(s^{\prime}, a^{\prime}\right), Q_{\theta_{2}^{\prime}}\left(s^{\prime}, a^{\prime}\right)\right),\quad a^{\prime}=\pi_{\phi^{\prime}}\left(s^{\prime}\right)+\epsilon $$

- $\mathcal{D}$：经验回放缓冲区，存储$\left(s, a,r,s^{\prime}\right)$元组。
- $\theta_{1}^{\prime}, \theta_{2}^{\prime}$：目标Critic网络参数。
- $\phi^{\prime}$：目标Actor网络参数。
- $\epsilon\sim\operatorname{clip}(\mathcal{N}(0,\sigma),-c, c)$：平滑噪声。

### 2.2.2 Actor目标函数

Actor网络通过最大化Q值更新：

$$ J(\phi)=\mathbb{E}_{s\sim\mathcal{D}}\left[Q_{\theta_{1}}\left(s,\pi_{\phi}(s)\right)\right] $$

注意：只使用$Q_{\theta_{1}}$更新策略，避免引入额外复杂性。

## 2.3 梯度推导

### 2.3.1 Critic梯度

对于$L\left(\theta_{i}\right)$，梯度为：

$$ \nabla_{\theta_{i}} L\left(\theta_{i}\right)=\mathbb{E}_{\left(s, a,r,s^{\prime}\right)\sim\mathcal{D}}\left[2\left(Q_{\theta_{i}}(s, a)-y\right)\nabla_{\theta_{i}} Q_{\theta_{i}}(s, a)\right] $$

- $Q_{\theta_{i}}(s, a)$：Critic网络输出。

- $y$：目标Q值，固定在梯度计算时。

- $ \nabla_{\theta_{i}} Q_{\theta_{i}}(s, a) $ : Critic网络对参数 $ \theta_{i} $ 的梯度，通过反向传播计算。

### 2.3.2 Actor梯度：

  根据确定性策略梯度定理，Actor梯度为：

  $$ \nabla_{\phi} J(\phi)=\mathbb{E}_{s\sim\mathcal{D}}\left[\left.\nabla_{a} Q_{\theta_{1}}(s, a)\right|_{a=\pi_{\phi}(s)} \cdot \nabla_{\phi} \pi_{\phi}(s)\right] $$

- $ \nabla_{a} Q_{\theta_{1}}(s, a) $ : Q函数对动作的梯度。

- $ \nabla_{\phi} \pi_{\phi}(s) $ :策略网络对参数 $ \phi $ 的梯度。

-  通过链式法则，梯度从Q值传递到策略参数。

## 2.4 网络参数更新

### 2.4.1 Critic更新：

  使用梯度下降更新 $ \theta_{1}, \theta_{2} $ :

  $$ \theta_{i} \leftarrow \theta_{i}-\alpha \nabla_{\theta_{i}} L\left(\theta_{i}\right), \quad i=1,2 $$

  其中，$ \alpha $ 为Critic学习率。

### 2.4.2 Actor更新：

  使用梯度上升更新 $ \phi $ :

  $$ \phi \leftarrow \phi+\beta \nabla_{\phi} J(\phi) $$

  其中，$ \beta $ 为Actor学习率。注意，Actor每 d 次Critic更新后更新一次。

### 2.4.3 目标网络更新：

  使用软更新 (Polyak Averaging) 更新目标网络：

  $$ \theta_{i}^{\prime} \leftarrow \tau \theta_{i}+(1-\tau) \theta_{i}^{\prime}, \quad i=1,2 $$

  $$ \phi^{\prime} \leftarrow \tau \phi+(1-\tau) \phi^{\prime} $$

- $ \tau $ : 软更新系数，通常取小值（如0.005），使目标网络缓慢跟踪主网络。

  

  
  
  
  
## 2.5 训练全过程

  TD3涉及以下网络：

  - **主网络**：
    - Actor网络 $$ \pi_{\phi}(s) $$
    - 两个Critic网络 $$ Q_{\theta_{1}}(s, a),Q_{\theta_{2}}(s,a) $$
  
  - **目标网络**：
    - 目标Actor网络 $$ \pi_{\phi^{\prime}}(s) $$
    - 两个目标Critic网络 $$ Q_{\theta_{1}^{\prime}}(s,a),Q_{\theta_{2}^{\prime}}(s,a) $$

### 2.5.1 训练步骤：

  1. 初始化主网络参数 $$ \theta_{1},\theta_{2},\phi $$ 和目标网络参数 $$ \theta_{1}^{\prime}=\theta_{1},\theta_{2}^{\prime}=\theta_{2},\phi^{\prime}=\phi $$ 。
  
  2. 初始化经验回放缓冲区 $$ \mathcal{D} $$。
  
  3. 循环：
  
     - 与环境交互，采集 $$ \left(s, a,r,s^{\prime}\right) $$，存入 $$ \mathcal{D} $$ 。
    
     - 从 $$ \mathcal{D} $$ 中采样小批量数据。
    
     - 计算目标Q值 $$ y $$，更新Critic网络 $$ \theta_{1},\theta_{2} $$ 。
    
     - 每 $$ d $$ 次Critic更新后，更新Actor网络 $$ \phi $$ 。
    
     - 软更新目标网络 $$ \theta_{1}^{\prime},\theta_{2}^{\prime},\phi^{\prime} $$。

  





