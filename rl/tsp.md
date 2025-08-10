可以直接抄 https://ekimetrics.github.io/blog/2021/11/03/tsp/   2021年的这篇文章







使用强化学习最基础的算法但是要使用mlp多层神经网络，不使用指针等神经网络，来解决tsp问题。设计方案需要考虑如下内容：能否直接输入one hot编码+其他输入（其中0为未访问的城市，1为访问的城市，以访问 城市的距离为负或者倒数定义为reward，或者其他更好的方案）；问题是10-50个城市数量；tsp的路径要求回到起点；状态也需要你设计；不限制使用模型，可以用dqn，sarsa、actor-critic都可以。最后的效果要求好，也要求给出设计方案的创新点和对应解决问题。





将基于强化学习（RL）的基本算法，使用多层感知机（MLP）神经网络来解决旅行商问题（TSP）。我不会使用指针网络（Pointer Network）或其他高级神经网络结构（如Transformer），而是坚持使用简单的MLP作为策略或价值函数的近似器。方案设计考虑了您提到的点：one-hot编码 + 其他输入、reward定义（包括您建议的负距离或倒数，以及我提出的更好方案）、10-50个城市规模、路径必须返回起点、状态设计、模型选择（我选择DQN作为基础算法，但也讨论了SARSA和Actor-Critic的变体）。方案旨在实现较好的效果（针对小规模TSP，RL结合MLP可以逼近最优解，尤其在10-20城市时；对于50城市，可能需要更多训练迭代，但通过reward shaping可以提升收敛）。







使用强化学习最基础的算法，不使用指针等神经网络，来解决tsp问题，如何解决tsp问题，能否直接输入one hot编码+其他输入（其中0为未访问的城市，1为访问的城市，以访问 城市的距离为负定义为reward进行优化）。要求设计的方案效果好，不要使用q learning的方式，因为table太小。并且不限制使用的强化学习算法如dqn、sarsa、actor-critic等，并且列出创新点和解决的问题





**空间结构（城市距离）并没有直接编码到 state 中**。

那么 Q 值的更新只能“从 reward 中”间接推断出距离结构：比如：如果从 A → B 给了 -2 reward，系统逐渐学到 A → B 不如 A → C（reward=-1）。

这种方式是“黑盒记忆式”的，**效率远不如直接输入城市坐标或距离矩阵**。



每一步移动都有 reward（负距离），**不是典型的稀疏奖励问题**（例如只有终点给奖励）。

所以**reward 是稠密的**，每一步都有信息。





**捕捉空间关系**：模型需理解城市坐标 (x,y)(x, y)(x,y) 之间的几何关系（如欧几里得距离），以生成更优的巡游路径。

**捕捉序列依赖**：模型需建模访问顺序的依赖性，确保生成合法的 TSP 解（每个城市访问一次并返回起点）。







- - **空间关系**：Q-Learning 通过奖励间接学习距离关系，缺乏指针网络的结构化建模（如注意力机制捕捉全局几何）。

  - **序列依赖**：Q-Learning 的状态仅记录访问状态和当前城市，序列依赖建模较弱，指针网络通过 LSTM/注意力捕捉长期依赖。

  - **状态空间爆炸**：TSP50 的状态空间为 250⋅50 2^{50} \cdot 50 250⋅50，Q 表存储和收敛困难，指针网络通过神经网络泛化。

  - 使用基础的 Q-Learning 算法，通过 one-hot 编码（0 表示未访问，1 表示已访问）和负距离奖励，可以解决 TSP，生成合法排列。设计上通

  - ：TSP 的最优解依赖于整个路径的选择。例如，选择某个城市作为早期访问点可能影响后续路径的总长度（例如，绕过远距离城市）。

    **累计效应**：早期动作（选择城市）对后续状态和奖励有长期影响。例如，早期选择较远的城市可能导致后续路径被迫绕行，增加总长度。

    **Q-Learning 的局限**：Q-Learning 使用 one-hot 编码和当前城市表示状态，序列依赖仅通过当前状态间接建模，难以捕捉早期决策对整个路径的影响。

    **对比神经网络**：Bello et al. (2016) 的指针网络通过 LSTM 和注意力机制捕捉长期依赖（例如，Decoder 记住已访问序列的历史），生成更优路径（TSP50 平均长度 6.09，接近最优 5.69）。

  - **历史路径信息**：在状态中加入已访问城市的顺序信息，例如：

    - 记录最近 k k k 个访问城市的索引：ht=[πt−k,…,πt−1] h_t = [\pi_{t-k}, \ldots, \pi_{t-1}] ht=[πt−k,…,πt−1]。
    - 或将已访问城市的距离均值加入状态：mean([Dπi,πi+1∣i∈visited]) \text{mean}([D_{\pi_i, \pi_{i+1}} \mid i \in \text{visited}]) mean([Dπi,πi+1∣i∈visited])。

    **分层策略**：将 TSP 分解为子区域，分别优化子路径，减少长期依赖的复杂性。

    **问题本质**：虽然 TSP 的目标是访问顺序，但优化高质量顺序需要考虑全局路径结构，长期依赖影响路径的整体效率



TSP 是一个 NP 难的組合优化问题，解空间大小为 n! n! n!（n 个城市的排列数）。直接使用策略梯度优化（不结合结构化模型）需要定义一个固定的动作空间，难以处理：

- **变长动作空间**：城市数量 n n n 可变，动作空间（选择下一个城市）随实例变化。



问题：

## TSP 一定要建模序列依赖，不建模顺序就等于不能解决问题

> TSP reward 是**序列结构依赖型**，而迷宫 reward 是**目标状态依赖型**



**负 reward 全局偏小 → Q 值会变得很负，收敛慢：**

- Q 更新步步都是负的，容易导致 Q 值下降到极低，反向传播效果变差（梯度微弱）。

**策略收敛慢的根因：组合空间大**

- 即使 reward 不稀疏，但状态空间是 `O(N × 2^N)`，导致要探索大量 episode。
- 特别在贪心策略收敛前，容易被次优路径困住。

方案：

使用 reward normalization 或 reward scaling（如除以最大距离，或使用 softmax Q 更新）。

添加回到起点的额外奖励，鼓励完成回路。





就像不给你地图，只让你走，走多了才知道哪条路近。Q-learning 是凭经验积累地图，而不是看地图做决策。

- ✅ 可以间接学到城市间关系（通过 reward 驱动）。
- ❌ 但效率低、不泛化、不稳健。
- 若想加快学习效果，可考虑：将城市对之间的距离作为一部分 state 或特征辅助学习（即 state = 当前城市 + visited 状态 + 到所有城市的距离向量）。





分层策略

**reward shaping** refers to the technique of modifying or augmenting the reward function in reinforcement learning (RL) to guide the agent's learning process more effectively. The goal is to provide additional intermediate rewards or penalties that help the agent learn desired behaviors faster or avoid poor strategies, without altering the optimal policy of the original task. It is often used to address sparse reward problems (where meaningful rewards are rare) by adding denser, domain-specific feedback signals



**Dense reward** refers to a reward function in reinforcement learning that provides frequent, fine-grained feedback to the agent at each step or state transition. Unlike sparse rewards (where feedback is given only upon task completion or rare milestones), dense rewards guide the agent more continuously by shaping behavior incrementally. This can accelerate learning but requires careful design to avoid unintended biases or suboptimal policies.





**State aliasing** **状态混淆**occurs in reinforcement learning (RL) when two or more distinct environmental states are mapped to the same internal representation by the agent's perception system (e.g., due to limited sensors or function approximation). This can lead to the agent treating different states as identical, causing suboptimal or even dangerous decisions. State aliasing is a key challenge in **partial observability** and can contribute to perceptual aliasing (where different observations appear the same).









## 一、对比指标设计

目标是围绕两大研究维度：

- **不同状态表示的相对贡献**（贡献度分析）
- **泛化性分析**（zero-shot能力）

我们将综合使用以下指标进行对比：

### 1. 收敛性指标

| 指标                                      | 定义                                    | 用途           |
| ----------------------------------------- | --------------------------------------- | -------------- |
| **收敛速度 (Convergence Speed)**          | 达到 `95% 最优路径长度` 所需 episode 数 | 衡量学习效率   |
| **收敛成功率 (Convergence Success Rate)** | 达到收敛条件的实验比例                  | 衡量算法鲁棒性 |
| **收敛episode标准差**                     | 不同运行之间的 episode 数波动           | 衡量稳定性     |



### 2. 性能指标

| 指标                           | 定义                                                         | 用途                     |
| ------------------------------ | ------------------------------------------------------------ | ------------------------ |
| **最终路径长度均值**           | 最后100 episode 的平均路径长度                               | 衡量最终解质量           |
| **最优性Gap (Optimality Gap)** | (path_length−optimal)/optimal×100%(path\_length - optimal) / optimal × 100\%(path_length−optimal)/optimal×100% | 衡量解的接近程度         |
| **路径效率奖励均值**           | `path_efficiency_reward = optimal / path_length`             | 替代路径长度，统一归一化 |



### 3. 稳定性与鲁棒性

| 指标                      | 定义                               | 用途                 |
| ------------------------- | ---------------------------------- | -------------------- |
| **变异系数 (CV)**         | `std / mean`，对路径长度等指标计算 | 衡量不同运行的稳定性 |
| **失败率 (Failure Rate)** | `1 - 有效解数量 / 总实验数`        | 衡量极端失败风险     |
| **最差性能 (Worst-Case)** | 所有实验中最差路径长度             | 衡量下界风险控制     |



### 4. 泛化性指标

| 指标                                 | 定义                                                         | 用途                   |
| ------------------------------------ | ------------------------------------------------------------ | ---------------------- |
| **Generalization Gap**               | test optimality gap−train optimality gap\text{test optimality gap} - \text{train optimality gap}test optimality gap−train optimality gap | 衡量train-test泛化能力 |
| **Cross-Instance Test Success Rate** | `有效解比例（cross_instance + test）`                        | 衡量zero-shot泛化能力  |



------

## 二、指标计算公式

公式如下：

```
text


复制编辑
1. optimality_gap = (L_path - L_opt) / L_opt * 100
2. path_efficiency_reward = L_opt / L_path
3. convergence_speed = min{episode | path_efficiency_reward >= 0.95}
4. convergence_success_rate = count(converged) / total_runs
5. stability_cv = std(L_path) / mean(L_path)
6. failure_rate = count(valid_solution == False) / total_runs
7. generalization_gap = test_optimality_gap - train_optimality_gap
```

------

## 三、可视化方案设计

### 【研究目标1】状态表示的相对贡献分析（基于 per-instance 模式）

#### 🔹图1：收敛速度对比图

- 类型：箱线图
- x轴：状态表示（state_type）
- y轴：收敛episode数（convergence_speed）
- 分组：算法类型 + 城市数量

#### 🔹图2：最终性能热力图

- 类型：heatmap
- 坐标：x = 状态表示，y = 算法
- 数值：最终路径长度均值 / 最优路径长度

#### 🔹图3：贡献雷达图（各指标综合）

- 类型：雷达图
- 指标：收敛速度、Gap、失败率、CV、最优奖励
- 说明：展示“full”与“ablation_x”的相对差距

------

### 【研究目标2】泛化性分析（基于 cross-instance 模式）

#### 🔹图4：train/test optimality gap 对比柱状图

- 类型：分组柱状图
- x轴：state_type
- y轴：optimality gap
- hue：train/test

#### 🔹图5：Generalization Gap 分布

- 类型：小提琴图 / 箱线图
- y轴：generalization_gap
- x轴：state_type
- 分组：城市数 + 算法

#### 🔹图6：zero-shot成功率趋势线

- 类型：折线图
- x轴：城市数量
- y轴：test成功率
- 线：不同状态表示（state_type）

------

## 四、推荐分析策略（博士级）

1. **显著性检验**：
   - 每个state_type之间使用 `paired t-test` / `Mann-Whitney U检验` 分析指标显著差异
2. **贡献分析量化**：
   - 计算每个state分量缺失后对主要指标的平均退化量（Delta Gap, Delta Speed）
3. **主成分分析**（PCA）：
   - 将所有指标合并，对比full与ablation后的分布差异
4. **交叉泛化矩阵**：
   - 行为训练状态表示，列为测试状态表示，记录每组组合的test成功率（分析跨状态迁移性能）