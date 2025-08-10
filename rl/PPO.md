

论文：**Proximal Policy Optimization Algorithms** https://arxiv.org/pdf/1707.06347



# 1. 解决的问题



PPO（Proximal Policy Optimization）由OpenAI团队在2017年提出，旨在：

- 设计简单、高效、稳定的强化学习算法
- 适用于多种任务，同时保持易于实现和调参的特点
- 结合策略梯度方法和信任区域方法的优点
- 通过限制策略更新的幅度确保训练稳定性
- 保持较高的样本效率



在强化学习中，策略梯度方法（如REINFORCE）虽然理论上成立，但在实际中存在高方差、收敛慢、更新不稳定等问题。为了解决这些问题，人们提出了TRPO（Trust Region Policy Optimization），通过限制策略变化的范围，提高了稳定性。

但是，TRPO实现复杂、计算开销大（需要二阶导数、共轭梯度），这限制了其广泛应用。



PPO（Proximal Policy Optimization，近端策略优化）算法的命名反映了其核心设计理念和优化方式。以下是命名的来源和原因：

1. **Proximal（近端）**：

   - “Proximal”指的是PPO在更新策略时限制新策略与旧策略之间的差异，保持更新的“接近性”。这通过裁剪概率比率实现，限制在$$ [1-\epsilon,1+\epsilon] $$范围内。

   - 这种限制受到TRPO（Trust Region Policy Optimization，信任区域策略优化）的启发，TRPO使用KL散度约束来限制策略更新幅度，而PPO用更简单、计算效率更高的裁剪机制来实现类似效果。因此，“Proximal”强调了这种“近端”约束，防止策略更新过于激进。

     

## 1.1 问题分析

- **稳定性问题**：策略梯度方法（如REINFORCE）对超参数和初始条件敏感，容易导致训练不稳定

- **效率问题**：TRPO（Trust Region Policy Optimization）虽然稳定，但计算复杂，难以在大型神经网络上高效实现

- **样本效率低**：早期的强化学习算法需要大量样本，训练成本高

- 如何有效控制策略更新的步长

- 如何提升策略优化的稳定性与收敛速度

  

## 1.2 创新点

· Clipped Objective：不再使用复杂的Kullback-Leibler（KL）约束，而是通过"剪切"目标函数直接限制策略变化。

· 易于实现：相比TRPO复杂的二阶优化，PPO只需一阶梯度下降，可以直接用在现成的深度学习框架中。

· 样本效率高：可以重复使用多个batch数据进行多次更新。



# 2. PPO的数学原理与推导过程

PPO的核心是基于策略梯度方法的优化，通过引入裁剪代理目标函数来限制策略更新。以下逐步推导其数学原理。

## 2.1 策略梯度基础

强化学习的目标是最大化期望回报：

$$ J(\theta)=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \gamma^{t} r\left(s_{t}, a_{t}\right)\right] $$

其中：
- $\pi_{\theta}(a|s)$ 是参数化为θ的策略，表示在状态s下选择动作a的概率
- $\tau$是轨迹（状态-动作-奖励序列）
- $r(s_{t},a_{t})$ 是奖励函数
- $\gamma\in(0,1)$ 是折扣因子

策略梯度方法通过梯度上升更新策略参数：

$$ \nabla_{\theta} J(\theta)=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) A\left(s_{t}, a_{t}\right)\right] $$



## 2.2 优势函数
其中，$A\left(s_{t}, a_{t}\right)$ 是优势函数，表示动作$a_{t}$在状态$s_{t}$下的优越性，常用如下估计：

$$ A\left(s_{t}, a_{t}\right)=Q\left(s_{t}, a_{t}\right)-V\left(s_{t}\right) $$

或通过**广义优势估计(GAE):**

$$ A_{t}=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l},\quad\delta_{t}=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right) $$

其中$\lambda$是GAE的折扣参数。



## 2.3 对比TRPO

TRPO通过约束策略更新到"信任区域"内来保证稳定性，优化目标为：

$$ \max_{\theta} \mathbb{E}_{s,a \sim \pi_{\theta_{\text{old}}}} \left[\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A(s,a)\right] $$

约束条件为：

$$ \mathbb{E}\left[\mathrm{KL}\left(\pi_{\theta_{\text{old}}} \| \pi_{\theta}\right)\right] \leq \delta $$

其中KL是Kullback-Leibler散度，用于限制新旧策略的差异。然而，TRPO需要计算Hessian矩阵的逆，计算复杂。





## 2.4 PPO的裁剪代理目标

PPO改进了TRPO，提出了一种简单的代理目标函数，避免复杂约束。PPO的优化目标为：

$$ L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right] $$

其中：

- $r_{t}(\theta)=\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text{old}}}\left(a_{t} \mid s_{t}\right)}$ 是新旧策略的概率比
- $\hat{A}_{t}$ 是优势估计
- $\operatorname{clip}(x, a, b)$ 是裁剪函数，将$x$限制在$[a, b]$范围内
- $\epsilon$ 是一个超参数（如0.2），控制裁剪范围

- 当$\hat{A}_t>0$时，限制$r_t(\theta)$过大，防止过度乐观更新
- 当$\hat{A}_t<0$时，限制$r_t(\theta)$过小，防止过度悲观更新



### 2.4.1 值函数损失
PPO同时优化值函数：

$$ L^{\mathrm{VF}}(\theta)=\mathbb{E}_t\left[\left(V_\theta(s_t)-V_t^{\mathrm{target}}\right)^2\right] $$

其中$V_t^{\mathrm{target}}$通过回报或GAE计算。



### 2.4.1 总损失函数

PPO的最终损失函数结合策略损失和值函数损失：

$$ L(\theta)=L^{\mathrm{CLIP}}(\theta)-c_{1} L^{\mathrm{VF}}(\theta)+c_{2} S\left[\pi_{\theta}\right]\left(s_{t}\right) $$

其中：
- $c_{1},c_{2}$ 是超参数，平衡各部分损失
- $S[\pi_{\theta}](s_{t})$ 是熵正则化项，促进探索

**数学意义**

- **裁剪的意义**  
   裁剪机制等效于在信任区域内优化策略，但避免了TRPO的复杂约束

- **优势估计**  
   通过GAE，PPO平衡了偏差和方差，提供了更稳定的优势估计

- **熵正则化**  
   防止策略过早收敛到局部最优，促进探索

#### 2.4.1.1 熵项的定义

策略熵(Policy Entropy)定义为动作概率分布的负对数期望:

$$ H\left(\pi_{\theta}\left(\cdot \mid s_{t}\right)\right)=-\mathrm{E}_{a_{t} \sim \pi_{\theta}}\left[\log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right] $$

物理意义:熵值越大，策略的随机性越强，探索性越高。

#### 2.4.1.2 熵项梯度推导

我们需要计算熵对策略参数$ \theta $的梯度$ \nabla_{\theta} H $。以下是详细推导步骤:

**步骤1：展开期望表达式**

将期望展开为对动作概率的求和(离散动作空间)或积分(连续动作空间):

$$ H\left(\pi_{\theta}\right)=-\sum_{a} \pi_{\theta}\left(a \mid s_{t}\right) \log \pi_{\theta}\left(a \mid s_{t}\right) $$

或连续情况:

$$ H\left(\pi_{\theta}\right)=-\int\pi_{\theta}\left(a\mid s_{t}\right)\log\pi_{\theta}\left(a\mid s_{t}\right) da $$

**步骤2: 直接对$ \theta $求导**

对$ H\left(\pi_{\theta}\right) $直接求梯度:

$$ \nabla_{\theta} H=-\nabla_{\theta}\left(\sum_{a} \pi_{\theta}\left(a \mid s_{t}\right) \log \pi_{\theta}\left(a \mid s_{t}\right)\right) $$

根据乘积法则，梯度分为两部分:

$$ \nabla_{\theta} H=-\sum_{a}\left[\nabla_{\theta} \pi_{\theta}\left(a \mid s_{t}\right) \cdot \log \pi_{\theta}\left(a \mid s_{t}\right)+\pi_{\theta}\left(a \mid s_{t}\right) \cdot \nabla_{\theta} \log \pi_{\theta}\left(a \mid s_{t}\right)\right] $$

**步骤3: 简化梯度项**

利用对数导数恒等式$ \nabla_{\theta} \log \pi_{\theta}\left(a \mid s_{t}\right)=\frac{\nabla_{\theta} \pi_{\theta}\left(a \mid s_{t}\right)}{\pi_{\theta}\left(a \mid s_{t}\right)} $,第二项可化简为:

$$ \pi_{\theta}\left(a \mid s_{t}\right) \cdot \nabla_{\theta} \log \pi_{\theta}\left(a \mid s_{t}\right)=\nabla_{\theta} \pi_{\theta}\left(a \mid s_{t}\right) $$

$$ \nabla_{\theta} H=-\sum_{a} \nabla_{\theta} \pi_{\theta}\left(a \mid s_{t}\right)\left(\log \pi_{\theta}\left(a \mid s_{t}\right)+1\right) $$

**转换为期望形式**
利用期望的梯度估计（对数导数恒等式），将求和转换为期望：

$$ \nabla_{\theta} H=-\mathrm{E}_{a_{t} \sim \pi_{\theta}}\left[\left(\log \pi_{\theta}\left(a_{t} \mid s_{t}\right)+1\right) \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right] $$





#### 2.4.1.4  物理意义与实现

**梯度更新增大动作概率的机制**

1. **梯度方向分析**：
   
   - 梯度表达式中的关键项：$(\log\pi_\theta +1)\nabla_\theta\log\pi_\theta$
   - 当$\pi_\theta(a_t|s_t)$较小时：
     * $\log\pi_\theta$为较大负值（如$\pi=0.1→\log\pi≈-2.3$）
     * $(\log\pi+1)$为负（$-2.3+1=-1.3$）
     * 前有总体负号：$-(-1.3)=+1.3$ → 正梯度方向
   - 更新效果：$\theta \leftarrow \theta + \alpha\cdot$(正梯度) → $\pi_\theta$增大
   
2. **熵提升原理**：
   
   - 熵$H$在均匀分布时最大
   - 梯度更新使小概率动作概率增大：
     * 减少"概率集中"现象
     * 使分布趋向均匀 → 直接提升$H$
     
     
     
     
     


## 2.5 参数更新方式

PPO的参数更新分为两部分：策略网络参数θ和价值网络参数φ的更新。两者都通过梯度下降（或其变种，如Adam优化器）进行优化。

### 2.5.1 策略网络参数θ的更新

目标：优化策略网络以最大化期望的累积奖励，同时通过约束避免策略更新过于激进。

更新方式：PPO使用Clipped Surrogate Objective（裁剪代理目标）来更新策略网络参数。更新基于以下目标函数：

$$ L^{\text{CLIP}}(\theta)=\mathbb{E}_{t}\left[\min\left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}(r_{t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{t}\right)\right] $$

其中：

$$ r_{t}(\theta)=\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text{old}}}\left(a_{t} \mid s_{t}\right)} $$

是新旧策略的概率比率。

$ \hat{A}_{t} $是优势函数（Advantage Function），表示在状态$ s_{t} $采取动作$ a_{t} $相对于平均策略的优劣。

$ \epsilon $是超参数（如0.2），控制裁剪范围，限制策略更新的幅度。

clip(x,a,b)将x限制在[a,b]范围内。

梯度计算：策略网络的参数θ通过对$ L^{\text{CLIP}}(\theta) $求梯度来更新：

$$ \nabla_{\theta} L^{\text{CLIP}}(\theta) \approx \nabla_{\theta}\left[\min\left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}(r_{t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{t}\right)\right] $$

### 2.5.2 价值网络参数$ \phi $的更新

- 目标：使价值网络$ V_{\phi}(s) $尽可能准确地估计真实的状态价值函数$ V(s) $。

- 更新方式：价值网络通过最小化均方误差（MSE）损失函数更新：

$$ L^{\text{VALUE}}(\phi)=\mathbb{E}_{t}\left[\left(V_{\phi}(s_{t})-R_{t}\right)^{2}\right] $$

其中$ R_{t} $是目标回报（target return），通常通过以下方式计算：

- 使用TD（Temporal Difference）目标：$ R_{t}=r_{t}+\gamma V_{\phi}(s_{t+1}) $，其中$ r_{t} $是即时奖励，$ \gamma $是折扣因子。

- 或者使用GAE（Generalized Advantage Estimation）的回报估计：

$$ R_{t}=\sum_{k=0}^{T-t}(\gamma\lambda)^{k}\delta_{t+k},\quad\text{其中}\quad\delta_{t}=r_{t}+\gamma V_{\phi}(s_{t+1})-V_{\phi}(s_{t}) $$

这里的$ \lambda $是GAE的超参数，用于平衡偏差和方差。

- 梯度计算：

$$ \nabla_{\phi} L^{\text{VALUE}}(\phi)=\nabla_{\phi}\left[\left(V_{\phi}(s_{t})-R_{t}\right)^{2}\right] $$



### 2.5.3 总体损失函数



- **监督学习**：通常用“损失函数”（Loss Function），因为目标是最小化误差（如分类错误、回归误差）。  

- **强化学习**：通常用“目标函数”（Objective Function），因为目标是最大化回报（即使某些部分看起来像“损失”，如价值函数误差）。

  

在实践中，PPO通常将策略目标和价值目标组合成一个联合损失函数，并添加熵正则化以鼓励探索：

$$ L(\theta,\phi)=L^{\text{Clip}}(\theta)-c_{1} L^{\text{Value}}(\phi)+c_{2} S[\pi_{\theta}](s_{t}) $$

其中：

- 裁剪代理目标，用于优化策略以最大化期望回报

- 值函数损失（均方误差），用于拟合状态值函数

- 熵正则化项，其中：
  - $c_{2}>0$ 是超参数
  - $S\left[\pi_{\theta}\right](s)=-\sum_{a\in\mathcal{A}}\pi_{\theta}(a|s)\log\pi_{\theta}(a|s)$ 是策略熵

参数$ \theta $和$ \phi $通常通过多次迭代（小批量梯度下降）更新，每次更新使用采集到的轨迹数据（状态、动作、奖励等）。



○ 策略目标是直接最大化  
○ 价值函数误差是最小化，但在目标函数中前面有负号（$-c_{1}L_{t}^{Value}$），所以整体仍然是最大化  
○ 熵是加，因为我们要最大化熵（鼓励探索）



#### 2.5.3.1 最大熵强化学习的理论背景

最大熵强化学习(Maximum Entropy RL)是一种强化学习框架，目标是最大化期望回报的同时，保持策略的熵尽可能高。其优化目标可以写为：

$$ J(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta}}\left[\sum_{t=0}^{T}\gamma^{t} r\left(s_{t}, a_{t}\right)+\alpha S\left[\pi_{\theta}\right]\left(s_{t}\right)\right] $$

其中：

- $\mathbb{E}_{\tau\sim\pi_{\theta}}\left[\sum_{t=0}^{T}\gamma^{t} r\left(s_{t}, a_{t}\right)\right]$: 期望回报，衡量策略的性能
- $\alpha S\left[\pi_{\theta}\right]\left(s_{t}\right)$: 熵正则化项，$\alpha>0$是权重
- $S\left[\pi_{\theta}\right]\left(s_{t}\right)=-\sum_{a\in\mathcal{A}}\pi_{\theta}\left(a\mid s_{t}\right)\log\pi_{\theta}\left(a\mid s_{t}\right)$: 策略在状态$s_{t}$下的熵


**熵正则化项说明**

熵正则化项基于策略$\pi_{\theta}(a|s)$的熵（entropy），表示策略在给定状态$s$下动作分布的随机性。数学上，策略的熵定义为：

$$ S[\pi_{\theta}](s)=-\sum_{a\in\mathcal{A}}\pi_{\theta}(a|s)\log\pi_{\theta}(a|s) $$


熵值越大，策略的动作分布越均匀（随机性高）；熵值越小，策略倾向于确定性地选择某些动作（随机性低）。





​     


### 2.5.4  实现示例（PyTorch）

```python
# 输入：动作概率分布probs (batch_size, action_dim)
log_probs = torch.log(probs)
entropy = -torch.sum(probs * log_probs, dim=-1).mean()  

# 梯度计算（自动微分）
entropy_loss = -c2 * entropy  # 系数c2控制强度
total_loss = policy_loss + value_loss + entropy_loss
total_loss.backward()  # 自动计算梯度并更新
```



### 2.5.6 核心流程


PPO是一种基于Actor-Critic架构的强化学习算法，结合策略网络（Actor）和值函数网络（Critic）进行优化。其核心流程如下：

1. **初始化**：初始化策略网络 $$ \pi_\theta $$、值函数网络 $$ V_\phi $$、环境以及超参数（如裁剪参数 $$ \epsilon=0.2 $$、学习率、优化步数等）。
2. **收集数据**：使用当前策略 $$ \pi_{\theta_{\text{old}}} $$ 与环境交互，收集轨迹数据，包括状态 $$ s_t $$、动作 $$ a_t $$、奖励 $$ r_t $$ 和下一状态 $$ s_{t+1} $$。
3. **计算优势**：通过广义优势估计（GAE）计算优势函数 $$ \hat{A}_t $$，以评估动作的优越性。
4. **优化目标**：
   - 计算裁剪代理目标 $$ L^{\text{CLIP}} $$，限制策略更新幅度。
   - 计算值函数损失 $$ L^{\text{VF}} $$，优化状态值估计。
   - 可选：添加熵正则化 $$ S[\pi_\theta] $$，促进策略探索。
5. **更新参数**：使用一阶优化器（如Adam）更新策略参数 $$ \theta $$ 和值函数参数 $$ \phi $$。
6. **重复迭代**：重复步骤2-5，直到策略收敛或达到最大迭代次数。



PPO的目标是最大化一个“代理目标函数”(surrogate objective)，同时限制策略更新的幅度。定义概率比：

$$
r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

最初的代理目标函数是：

$$
L_{\text{CLIP}}(\theta) = \mathbb{E}_t[r_t(\theta) \hat{A}_t]
$$

但直接最大化这个目标可能导致$ r_t(\theta) $偏离1太多，因此PPO引入裁剪机制：

$$
L_{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min \left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right)\right]
$$

其中$\epsilon$是超参数(通常为0.1-0.2)，clip(x, a, b)表示将x裁剪到[a, b]区间。

这个裁剪操作保证了：

- 当$r_t(\theta) > 1 + \epsilon$时，使用$(1 + \epsilon) \hat{A}_t$
- 当$r_t(\theta) < 1 - \epsilon$时，使用$(1 - \epsilon) \hat{A}_t$
- 否则使用$r_t(\theta) \hat{A}_t$

从而限制了策略更新的幅度。






# 3. 训练过程


Algorithm 1 PPO, Actor-Critic Style

for iteration=1,2,... do
    for actor=1,2,...,N do
        Run policy $ \pi_{\theta_{\text{old}}} $ in environment for $ T $ timesteps
        Compute advantage estimates $ \hat{A}_{1},\ldots,\hat{A}_{T} $
    end for
    Optimize surrogate $ L $ wrt $ \theta $, with $ K $ epochs and minibatch size $ M\leq NT $
    $ \theta_{\text{old}}\leftarrow\theta $
end for



## 3.1 迭代
算法通过多个**迭代**来更新策略。每个迭代中，有多个**演员**（actor）并行地与环境交互，收集数据。具体步骤如下：

1. **迭代（Iteration）**：
   - 算法会进行多次迭代，每次迭代都进行策略更新。
   
2. **Actor**：
   - 对于每个演员，执行以下操作：
     - **运行策略（Run Policy）**：
       演员使用当前策略 在环境中执行 \( T \) 步，收集数据。
     - **计算优势估计（Compute Advantage Estimates）**：
       基于收集的数据计算每个时间步的优势估计，用于衡量当前策略比基准策略的优越程度。

3. **优化过程（Optimize Surrogate）**：
   - 在所有演员完成任务后，使用收集的数据来优化代理目标函数 \( L \)，并更新策略参数 \( \theta \)。

### 3.1.1 N 和 T 的含义

- **N**：表示演员的数量，即有多少个并行的进程或工作者在环境中执行当前的策略。多个演员可以加速算法的学习过程。
  
- **T**：表示每个演员在一次迭代中与环境交互的时间步数。每个演员会执行 \( T \) 步，并生成相应的数据。



### 3.1.2 "wrt" 的含义

在PPO算法中，"wrt" 是 "with respect to" 的缩写，意思是“关于”或“相对于”。



### 3.1.3 代码实现

```python
初始化策略网络 π_θ 和价值网络 V_θ
初始化超参数：ε, c1, c2, γ, λ, 最大迭代次数 T, 每次采样的步数 N, 小批量大小 M
对于迭代次数 t = 1, 2, ..., T 做：
    # 1. 收集轨迹数据
    使用当前策略 π_θ_old 采样 N 步轨迹 {s_t, a_t, r_t, s_{t+1}}
    计算回报 R_t 和优势估计 A_t（使用 GAE）
    
    # 2. 更新策略和价值网络
    对于 K 次小批量更新（K 为超参数）：
        随机采样 M 个数据点
        计算概率比 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        计算剪切目标函数 L^CLIP(θ)
        计算价值损失 L^VF(θ)
        计算熵正则项 S[π_θ](s_t)
        计算总损失 L(θ)
        使用优化器（如 Adam）更新参数 θ
    
    # 3. 更新旧策略
    将 π_θ 复制到 π_θ_old
```
```python
def update(self, states, actions, rewards, next_states):
    # 计算回报和优势
    values = self.value(states).squeeze().detach()
    next_value = self.value(next_states[-1]).squeeze().detach()
    advantages = self.compute_gae(rewards, values, next_value)
    advantages = torch.tensor(advantages)
    returns = advantages + values

    # 计算概率比和策略损失
    old_probs = self.policy(states).gather(1, actions).detach()
    new_probs = self.policy(states).gather(1, actions)
    ratio = new_probs / old_probs
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 计算价值损失
    value_pred = self.value(states).squeeze()
    value_loss = ((value_pred - returns) ** 2).mean()

    # 计算熵正则化项（可选）
    dist = torch.distributions.Categorical(self.policy(states))
    entropy = dist.entropy().mean()
    
    # 总损失函数
    c1 = 0.5  # 价值损失权重
    c2 = 0.01  # 熵正则化权重
    total_loss = policy_loss + c1 * value_loss - c2 * entropy

    # 联合优化策略和价值网络
    self.policy_optimizer.zero_grad()
    self.value_optimizer.zero_grad()
    total_loss.backward()
    self.policy_optimizer.step()
    self.value_optimizer.step()
```





# 4. TRPO vs PPO 对比

## 4.1 TRPO

- **目标**  
  TRPO通过信任区域(trust region)优化策略，最大化期望优势函数，同时约束新旧策略之间的KL散度。

- **更新方式**  
  TRPO的目标函数为:
  $$ \max_{\theta}\mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\hat{A}_t\right], \quad \text{s.t.} \quad \mathbb{E}\left[ \text{KL}(\pi_{\theta_{\text{old}}},\pi_\theta) \right] \leq \delta $$
  其中$ \hat{A}_t $是优势函数，$ \delta $是KL散度的约束阈值。

- **优化过程**  
  TRPO使用二阶优化(如共轭梯度法)近似求解信任区域约束优化问题，计算复杂且需要估计Fisher信息矩阵来逼近KL散度约束。

- **价值网络更新**  
  价值网络通过最小化均方误差更新:
  $$ L^{\text{VALUE}}(\phi)=\mathbb{E}_t\left[\left(V_\phi(s_t)-R_t\right)^2\right] $$
  其中$ R_t $是目标回报(如通过TD或Monte Carlo估计)。

- **特点**  
  TRPO通过严格的KL散度约束保证策略更新的稳定性，但计算开销大，超参数(如$ \delta $)调节复杂。



- 策略梯度：TRPO的策略梯度基于信任区域优化的目标：
  $$ \nabla_\theta \mathbb{E}_{\pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right] $$
- 但受KL散度约束$ \mathbb{E}\left[ \text{KL}(\pi_{\theta_{old}},\pi_\theta) \right] \leq \delta $限制，实际梯度通过近似二阶方法（如共轭梯度法）计算：
  $$ \nabla_\theta L^{\text{TRPO}}(\theta) \approx g - \alpha(F^{-1} g) $$
- 其中$ g $是原始策略梯度，$ F $是Fisher信息矩阵，$ \alpha $是步长。

- 价值梯度：价值网络的梯度为：
  $$ \nabla_\phi L^{\text{VALUE}}(\phi) = \nabla_\phi \left[ (V_\phi(s_t) - R_t)^2 \right] $$

- 特点：策略梯度计算复杂，涉及二阶信息（如Hessian矩阵的逆），需要额外的KL散度估计。



## 4.2 PPO 

- **目标**  
  PPO简化TRPO的信任区域约束，使用裁剪代理目标(Clipped Surrogate Objective)来限制策略更新幅度。

- **更新方式**  
  PPO的目标函数(以PPO-Clip为例)为:
  $$ L^{\text{CLIP}}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right] $$
  其中$ r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $是概率比率，$ \epsilon $是裁剪范围(如0.2)。

- **优化过程**  
  PPO使用一阶优化(如Adam优化器)，直接对裁剪目标函数进行梯度下降，计算简单且高效。

- **价值网络更新**  
  与TRPO类似，PPO通过最小化均方误差更新价值网络:
  $$ L^{\text{VALUE}}(\phi)=\mathbb{E}_t\left[\left(V_\phi(s_t)-R_t\right)^2\right] $$



**总体对比总结**

- **网络结构**：TRPO和PPO的网络结构基本一致（策略网络+价值网络+旧策略网络），差异在于实现细节和优化算法的复杂性。
- **参数更新**：
  - TRPO使用KL散度约束和二阶优化，计算复杂但稳定性强。
  - PPO使用裁剪概率比率和一阶优化，计算简单，超参数调节更友好。
- **更新梯度**：
  - TRPO的策略梯度涉及二阶信息和KL散度计算，复杂且耗时。
  - PPO的策略梯度基于裁剪目标，依赖一阶优化，计算效率高。
- **实际效果**：
  - TRPO在理论上更稳定，适合复杂任务，但实现难度大，计算成本高。
  - PPO在稳定性和效率之间取得平衡，适用于广泛任务，是许多实际应用的首选。
