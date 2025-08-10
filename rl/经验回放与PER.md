

# 1. 经验回放 算法应用
## 1.1 标准经验回放（Uniform Experience Replay）

- **DQN** (Deep Q-Network)
- **Double DQN**
- **Dueling DQN**
- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed DDPG)
- **SAC** (Soft Actor-Critic)



## 1.2 优先级经验回放（Prioritized Experience Replay, PER）

- **Prioritized DQN** (PER-DQN)
- **Rainbow DQN** (结合PER及其他改进)
- **PER-DDPG** (DDPG + PER)
- **PER-SAC** (SAC + PER)
- **Ape-X DQN** (分布式PER)
- **R2D2** (Recurrent Replay Distributed DQN, 使用PER)





# 2.经验回放 (Experience Replay)

经验回放是深度强化学习 (DRL) 中的一个关键技术，主要用于离线策略 (Off-Policy) 算法（如DQN、DDPG、SAC等）。其核心思想是将智能体与环境交互产生的历史经验（状态、动作、奖励等）存储在一个经验池中，并从中随机采样进行训练，从而打破数据间的相关性，提高样本利用率。





## 1.1为什么需要经验回放？

**解决的问题：**

- **数据相关性：**
    - 连续采样得到的经验（如状态转移序列）具有强时间相关性，直接训练综合会导致神经网络过拟合局部数据。

- **样本效率：**
    - 在线策略 (On-Policy) 算法（如A3C、PPO）需要新数据立即更新，丧失旧数据，导致数据利用率低。

- **训练稳定性：**
    - 直接使用近期经验可能导致策略震荡（例如DQN中目标Q值的剧烈变化）。

**类比理解：**

类比于人类通过“记忆”学习：不是仅仅重复最近经验调整行为，而是从过去的多样经验中随机回顾学习。

## 1.2 经验回放的实现方式

**核心组件：**

- **回放缓冲区 (Replay Buffer)：**
    - 存储经验的队列或数组大小的缓冲区，每条记录通常为 $(s, a, r, s', done)$。

- **采样策略：**
    - 随机均匀采样 (DQN) 或优先采样 (Prioritized Experience Replay)。



# 3.优先级经验回放 (Prioritized Experience Replay, PER)

**改进点：**

- **关键思想：** 优先回放“更有学习价值”的经验（如TD误差较大的样本）。
  
- **实现方式：**
    - 为每条经验分配优先级，如 $p_i = |\delta_i| + \epsilon$，其中 $\delta_i$ 是TD误差。
    - 使用SumTree数据结构高效采样。

**优点：**
- 加速收敛，提高关键样本利用率。

**缺点：**
- 需要调整重要性采样权重 (IS weight) 以避免偏差。



## 3.1 Prioritized Experience Replay (PER) 实现详解

Prioritized Experience Replay (PER) 通过优先采样高价值（高TD误差）的经验来加速收敛，其核心包括：

1. **优先级分配 (基于TD误差)**
2. **高效采样 (SumTree数据结构)**
3. **偏差校正 (重要性采样权重)**

以下是具体算法流程及公式说明：

### 3.1.1 优先级分配 (Priority Assignment)

每条经验 $(s_t, a_t, r_t, s_{t+1}, done)$ 的优先级 $p_i$ 定义为：

$$
p_i = |\delta_i| + \epsilon
$$

- $\delta_i$: TD误差 (Temporal Difference Error)，衡量当前样本的“学习价值”。
    - 对于DQN: $\delta_i = r_t + \gamma \max_a Q_{\text{target}}(s_{t+1}, a') - Q(s_t, a_t)$
    - 对于Actor-Critic: $\delta_i = r_t + \gamma V_{\text{target}}(s_{t+1}) - V(s_t)$
  
- $\epsilon$: 极小常数（如 $10^{-5}$），避免优先级为0的样本完全不被采样。

**注：** 初始时所有样本的优先级设为最大值（如 $p_i = 1$），随后在训练中动态更新。

### 3.1.2 采样概率 (Sampling Probability)

样本 $i$ 的采样概率 $P(i)$ 由其优先级 $p_i$ 决定：

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

- $\alpha \in [0, 1]$: 控制优先级的强度。
    - $\alpha = 0$: 均匀采样（避免过多采样Replay Buffer）。
    - $\alpha = 1$: 完全优先级采样。
  
- 通常选择 $\alpha = 0.6$（较好的探索与利用平衡）。



### 3.1.3 偏差校正 (Importance Sampling Weight)

优先采样会引入偏差（高优先级样本被过度使用），需要通过重要性采样权重 $w_i$ 校正：

$$
w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta
$$

- $N$: 回放缓冲区大小。
- $\beta$ （$\beta \in [0, 1]$）：控制校正强度。
    - $\beta = 0$：无校正。
    - $\beta = 1$：完全校正。
    - 通常选择 $\beta = 0.4$ 逐步增加到 1.0。

最终损失函数（以DQN为例）：

$$
L = \frac{1}{B} \sum_i w_i \cdot (\delta_i)^2
$$

- $B$: 批次大小 (Batch Size)。
- 权重 $w_i$ 需要归一化（除去 $max_i w_i$，避免梯度爆炸）。




#### 3.1.3.1 权重（偏差校正）公式推导 
**(1) 问题描述**

- 目标：在均匀采样分布 $P_{\text{uniform}}(i) = \frac{1}{N}$ 下，优化期望损失：

$$
E_{\nu \sim \text{uniform}}[\delta^2_i] = \frac{1}{N} \sum_{i=1}^{N} \delta^2_i.
$$

- 实际采样分布：PER 的采样分布为 $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$，与均匀分布不一致。

**(2) 重要性采样 (Importance Sampling)**
为了在非均匀分布 $P(i)$ 下估计均匀分布的期望，引入权重 $w_i$：

$$
E_{\nu \sim \text{uniform}}[\delta^2_i] = E_{\nu \sim P(i)} \left[ P_{\text{uniform}}(i) \cdot \frac{\delta^2_i}{P(i)} \right].
$$

代入 $P_{\text{uniform}}(i) = \frac{1}{N}$，得到理想权重：

$$
w^*_i = \frac{1}{N \cdot P(i)}.
$$

**(3) 引入超参数 $\beta$**

- 完全校正 ($\beta = 1$)：直接使用 $w^*_i$，但可能导致高方差（尤其训练初期 $P(i)$ 不准确时）。
- 部分校正 ($\beta < 1$)：通过 $\beta$ 平滑权重，平衡偏差与方差：

$$
w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta.
$$

- $\beta = 0$：无校正 ($w_i = 1$)，退化为普通PER。
- $\beta = 1$：渐进完全校正。



### 3.1.4 完整算法流程

1. **存储经验：**
    - 新经验 $(s_t, a_t, r_t, s_{t+1}, done)$ 存入经验池，初始优先级 $p_t = \max_i p_i$。

2. **采样批次：**
    - 根据 $P(i)$ 从SumTree中采样B条经验。

3. **计算权重：**
    - 对每条样本计算权重 $w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta$。

4. **更新网络：**
    - 计算TD误差 $\delta_i$，更新网络或策略网络。

5. **更新优先级：**
    - 用新的 $\delta_i$ 更新SumTree中对应节点的优先级。





