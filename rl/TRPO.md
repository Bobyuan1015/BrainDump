# Trust Region Policy Optimization (TRPO)

**论文**：[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

TRPO 是一种强化学习中的策略优化算法，通过引入**信任区域约束**，限制新旧策略之间的差异，确保策略更新稳定且性能逐步提升。其核心思想是利用**平均KL散度**约束策略变化，避免过大更新导致性能下降。

---

## 1. 核心概念与优化目标

TRPO 的目标是最大化策略的期望累计奖励，同时限制新旧策略之间的偏差。其优化问题形式化为：

$$
\max_{\theta} L_{\pi_{\theta_{\text{old}}}}(\theta) \quad \text{s.t.} \quad \bar{D}_{KL}^{\pi_{\theta_{\text{old}}}}(\pi_{\theta_{\text{old}}}, \pi_{\theta}) \leq \delta
$$

- **代理目标函数 (Surrogate Objective)**：
  $$
  L_{\pi_{\theta_{\text{old}}}}(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}}\left[ \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A^{\pi_{\theta_{\text{old}}}}(s, a) \right]
  $$
  其中，$A^{\pi_{\theta_{\text{old}}}}(s, a)$ 是旧策略下的优势函数，表示动作 $a$ 在状态 $s$ 下的相对优越性。

- **KL散度约束**：
  $$
  \bar{D}_{KL}(\pi_{\theta_{\text{old}}} \| \pi_{\theta}) = \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}}\left[ D_{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s)) \right] \leq \delta
  $$
  确保新策略 $\pi_{\theta}$ 与旧策略 $\pi_{\theta_{\text{old}}}$ 在状态分布 $\rho_{\pi_{\theta_{\text{old}}}}$ 上的平均KL散度不超过阈值 $\delta$。

---

## 2. 平均KL散度的定义与分解

### 2.1 平均KL散度公式

平均KL散度衡量新旧策略在状态分布上的总体差异：

$$
\bar{D}_{KL}(\pi_{\theta_{\text{old}}} \| \pi_{\theta}) = \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}}\left[ D_{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s)) \right]
$$

- **状态分布** $\rho_{\pi_{\theta_{\text{old}}}}(s)$：旧策略 $\pi_{\theta_{\text{old}}}$ 在环境中运行时，状态 $s$ 的访问概率。
- **KL散度** $D_{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s))$：状态 $s$ 下新旧策略动作分布的KL散度。

### 2.2 KL散度的计算

对于状态 $s$，KL散度定义为：

- **连续动作空间**：
  $$
  D_{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s)) = \int_{a} \pi_{\theta_{\text{old}}}(a \mid s) \log \left( \frac{\pi_{\theta_{\text{old}}}(a \mid s)}{\pi_{\theta}(a \mid s)} \right) da
  $$

- **离散动作空间**：
  $$
  D_{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s)) = \sum_{a} \pi_{\theta_{\text{old}}}(a \mid s) \log \left( \frac{\pi_{\theta_{\text{old}}}(a \mid s)}{\pi_{\theta}(a \mid s)} \right)
  $$

平均KL散度通过状态分布加权：

$$
\bar{D}_{KL} = \int_{s} \rho_{\pi_{\theta_{\text{old}}}}(s) \cdot D_{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s)) ds
$$

### 2.3 为什么使用平均KL散度？

- **单点KL散度不足**：仅约束某一状态的KL散度可能导致其他状态策略剧烈变化，引发不稳定。
- **全局约束**：平均KL散度考虑所有状态的加权平均，确保策略全局稳定。

---

## 3. 实际计算中的近似方法

由于真实状态分布 $\rho_{\pi_{\theta_{\text{old}}}}$ 难以直接获得，TRPO 使用蒙特卡洛方法近似：

1. **采样状态**：用旧策略 $\pi_{\theta_{\text{old}}}$ 与环境交互，收集状态样本 $\{s_1, s_2, \dots, s_N\}$。
2. **计算单状态KL散度**：对每个状态 $s_i$，通过动作采样估计KL散度：
   $$
   D_{KL}^{(i)} \approx \frac{1}{M} \sum_{j=1}^{M} \log \left( \frac{\pi_{\theta_{\text{old}}}(a_j \mid s_i)}{\pi_{\theta}(a_j \mid s_i)} \right), \quad a_j \sim \pi_{\theta_{\text{old}}}(\cdot \mid s_i)
   $$
3. **估计平均KL散度**：
   $$
   \bar{D}_{KL} \approx \frac{1}{N} \sum_{i=1}^{N} D_{KL}^{(i)}
   $$

---

## 4. 策略优化的数学推导

### 4.1 期望累计奖励

策略梯度方法的目标是最大化期望累计奖励：

$$
J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \sum_{t} \gamma^{t} r(s_t, a_t) \right]
$$

直接使用策略梯度更新可能导致步长过大，TRPO 通过信任区域约束解决这一问题。

### 4.2 代理目标函数的推导

TRPO 使用旧策略的状态分布近似新策略的期望奖励：

$$
J(\pi_{\theta}) \approx J(\pi_{\theta_{\text{old}}}) + \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}, a \sim \pi_{\theta}}\left[ A^{\pi_{\theta_{\text{old}}}}(s, a) \right]
$$

通过**重要性采样**，将动作采样从新策略 $\pi_{\theta}$ 转换为旧策略 $\pi_{\theta_{\text{old}}}$：

$$
\mathbb{E}_{a \sim \pi_{\theta}(a \mid s)}\left[ A^{\pi_{\theta_{\text{old}}}}(s, a) \right] = \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}(a \mid s)}\left[ \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A^{\pi_{\theta_{\text{old}}}}(s, a) \right]
$$

得到代理目标函数：

$$
L_{\pi_{\theta_{\text{old}}}}(\theta) = \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}, a \sim \pi_{\theta_{\text{old}}}}\left[ \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A^{\pi_{\theta_{\text{old}}}}(s, a) \right]
$$

### 4.3 重要性采样的作用

重要性采样通过权重 $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ 校正分布差异，允许使用旧策略样本评估新策略性能：

$$
\mathbb{E}_{x \sim q}[f(x)] = \mathbb{E}_{x \sim p}\left[ \frac{q(x)}{p(x)} f(x) \right]
$$

在 TRPO 中，$p = \pi_{\theta_{\text{old}}}(a \mid s)$，$q = \pi_{\theta}(a \mid s)$，$f(x) = A^{\pi_{\theta_{\text{old}}}}(s, a)$。

---

## 5. 优化问题的求解

### 5.1 拉格朗日乘子法

TRPO 的优化问题为：

$$
\max_{\theta} L_{\pi_{\theta_{\text{old}}}}(\theta) \quad \text{s.t.} \quad \bar{D}_{KL}(\pi_{\theta_{\text{old}}} \| \pi_{\theta}) \leq \delta
$$

使用拉格朗日乘子法，转化为无约束问题：

$$
\mathcal{L}(\theta, \lambda) = L_{\pi_{\theta_{\text{old}}}}(\theta) - \lambda \left( \bar{D}_{KL}(\pi_{\theta_{\text{old}}} \| \pi_{\theta}) - \delta \right)
$$

### 5.2 函数的泰勒展开

为简化优化，TRPO 对目标函数和约束进行泰勒展开近似。

#### 5.2.1 一元函数泰勒展开

对于函数 $f(x)$ 在点 $a$ 处，泰勒展开为：

$$
f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2}(x - a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x - a)^n + \cdots
$$

- **一阶展开**：仅保留线性项，近似函数的局部变化：
  $$
  f(x) \approx f(a) + f'(a)(x - a)
  $$
- **二阶展开**：加入二次项，捕获函数曲率：
  $$
  f(x) \approx f(a) + f'(a)(x - a) + \frac{f''(a)}{2}(x - a)^2
  $$
- **麦克劳林级数**（$a = 0$）：
  $$
  f(x) = f(0) + f'(0)x + \frac{f''(0)}{2}x^2 + \cdots
  $$
  示例：
  - $e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$
  - $\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$

#### 5.2.2 多元函数泰勒展开

对于多元函数 $f(\theta)$ 在点 $\theta_0$ 处，泰勒展开为：

$$
f(\theta) \approx f(\theta_0) + \nabla f(\theta_0)^T (\theta - \theta_0) + \frac{1}{2} (\theta - \theta_0)^T H(\theta_0) (\theta - \theta_0)
$$

- **梯度** $\nabla f(\theta_0)$：函数在 $\theta_0$ 处的一阶偏导数向量。
- **Hessian 矩阵** $H(\theta_0)$：二阶偏导数矩阵，捕获函数曲率：
  $$
  H(f) = \begin{bmatrix}
  \frac{\partial^2 f}{\partial \theta_1^2} & \frac{\partial^2 f}{\partial \theta_1 \partial \theta_2} & \cdots \\
  \frac{\partial^2 f}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 f}{\partial \theta_2^2} & \cdots \\
  \vdots & \vdots & \ddots
  \end{bmatrix}
  $$
  Hessian 矩阵为对称矩阵（当二阶偏导数连续时），反映参数空间的局部几何特性。

#### 5.2.3 泰勒展开示例

**目标函数**：
$$
f(\theta) = \theta_1^2 + 3\theta_1\theta_2 + 2\theta_2^2
$$

**展开点**：$\theta_0 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

1. **函数值**：
   $$
   f(1, 2) = 1^2 + 3 \cdot 1 \cdot 2 + 2 \cdot 2^2 = 1 + 6 + 8 = 15
   $$

2. **梯度**：
   $$
   \nabla f(\theta) = \begin{bmatrix} 2\theta_1 + 3\theta_2 \\ 3\theta_1 + 4\theta_2 \end{bmatrix}, \quad \nabla f(1, 2) = \begin{bmatrix} 2 \cdot 1 + 3 \cdot 2 \\ 3 \cdot 1 + 4 \cdot 2 \end{bmatrix} = \begin{bmatrix} 8 \\ 11 \end{bmatrix}
   $$

3. **Hessian 矩阵**：
   $$
   H = \begin{bmatrix} \frac{\partial^2 f}{\partial \theta_1^2} & \frac{\partial^2 f}{\partial \theta_1 \partial \theta_2} \\ \frac{\partial^2 f}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 f}{\partial \theta_2^2} \end{bmatrix} = \begin{bmatrix} 2 & 3 \\ 3 & 4 \end{bmatrix}
   $$

4. **二阶展开**：
   设 $\delta \theta = \theta - \theta_0 = \begin{bmatrix} \Delta_1 \\ \Delta_2 \end{bmatrix}$，则：
   $$
   f(\theta) \approx 15 + \begin{bmatrix} 8 & 11 \end{bmatrix} \begin{bmatrix} \Delta_1 \\ \Delta_2 \end{bmatrix} + \frac{1}{2} \begin{bmatrix} \Delta_1 & \Delta_2 \end{bmatrix} \begin{bmatrix} 2 & 3 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} \Delta_1 \\ \Delta_2 \end{bmatrix}
   $$

### 5.3 目标函数与KL散度的近似

- **目标函数一阶展开**：
  $$
  L(\theta) \approx L(\theta_{\text{old}}) + g^T (\theta - \theta_{\text{old}})
  $$
  其中，$g = \nabla_{\theta} L(\theta_{\text{old}})$ 是策略梯度。

- **KL散度二阶展开**：
  $$
  \bar{D}_{KL}(\theta) \approx \frac{1}{2} (\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}})
  $$
  其中，$F$ 是 Fisher 信息矩阵：
  $$
  F = \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}, a \sim \pi_{\theta_{\text{old}}}}\left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s)^T \right]
  $$
  Fisher 矩阵捕获策略参数空间的曲率，KL散度在 $\theta_{\text{old}}$ 处一阶导数为零，二阶近似有效。

### 5.4 自然梯度更新

优化问题简化为：

$$
\max_{\theta} g^T (\theta - \theta_{\text{old}}) \quad \text{s.t.} \quad \frac{1}{2} (\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}}) \leq \delta
$$

解析解为：

$$
\theta^* = \theta_{\text{old}} + \sqrt{\frac{2 \delta}{g^T F^{-1} g}} F^{-1} g
$$

此更新方向称为**自然梯度**，通过 Fisher 矩阵 $F$ 调整梯度方向，适应参数空间几何特性。

### 5.5 共轭梯度法

直接计算 $F^{-1}$ 成本高，TRPO 使用共轭梯度法近似求解 $F^{-1} g$，并通过线搜索确保满足KL散度约束。

---

## 6. 直观示例

### 6.1 网格世界示例

假设一个网格世界任务：
- **状态**：网格位置 $(x, y)$。
- **动作**：上下左右移动。
- **旧策略** $\pi_{\theta_{\text{old}}}$：动作概率 $[0.4, 0.3, 0.2, 0.1]$。
- **新策略** $\pi_{\theta}$：动作概率 $[0.5, 0.2, 0.2, 0.1]$。

**单状态KL散度**：

$$
D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_{\theta})|_s = 0.4 \log \frac{0.4}{0.5} + 0.3 \log \frac{0.3}{0.2} + 0.2 \log \frac{0.2}{0.2} + 0.1 \log \frac{0.1}{0.1}
$$

**平均KL散度**：对所有状态按 $\rho_{\pi_{\theta_{\text{old}}}}(s)$ 加权平均。

### 6.2 代理目标函数计算

样本数据：

| 状态 | 动作 | $\pi_{\theta_{\text{old}}}(a \mid s)$ | $\pi_{\theta}(a \mid s)$ | $A(s, a)$ |
|------|------|-------------------------------------|-------------------------|-----------|
| $s_1$ | $a_1$ | 0.2                                 | 0.3                     | +5        |
| $s_2$ | $a_2$ | 0.5                                 | 0.6                     | -1        |

**代理目标函数**：

$$
L = \frac{0.3}{0.2} \cdot 5 + \frac{0.6}{0.5} \cdot (-1) = 7.5 - 1.2 = 6.3
$$

优化通过增加高优势动作（如 $a_1$）的概率、降低低优势动作（如 $a_2$）的概率提升性能。

---

## 7. TRPO 的核心优势

1. **稳定性**：KL散度约束避免过大更新导致性能下降。
2. **全局优化**：平均KL散度确保策略全局一致性。
3. **自然梯度**：利用 Fisher 矩阵调整更新方向，适应参数空间几何。
4. **适用性**：适合复杂环境和连续/离散动作空间。

---

## 8. 符号说明

| 符号                          | 含义                                              |
|-------------------------------|--------------------------------------------------|
| $\pi_{\theta_{\text{old}}}$   | 旧策略动作分布 $\pi_{\theta_{\text{old}}}(a \mid s)$ |
| $\pi_{\theta}$                | 新策略动作分布 $\pi_{\theta}(a \mid s)$          |
| $A(s, a)$                     | 优势函数，衡量动作相对优越性                     |
| $\rho_{\pi_{\theta_{\text{old}}}}(s)$ | 旧策略下的状态访问分布                     |
| $F$                           | Fisher 信息矩阵，捕获参数空间曲率                |
| $g$                           | 策略梯度 $\nabla_{\theta} L(\theta)$             |
| $\delta$                      | KL散度约束阈值                                   |
| $H$                           | Hessian 矩阵，二阶偏导数矩阵                     |

