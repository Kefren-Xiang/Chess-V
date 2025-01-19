# Chess-V训练过程与模型结构

## 1. 核心参数

在每个节点（即每个棋局状态）上，有以下几个关键参数是固定的，不受模型和结构的影响：

- **N**：节点的访问次数 + 1（初始为 1）。
- **W**：节点的总得分。
- **Q**：节点的平均得分，等于 `W / N`，通常在区间 `[-1, 1]` 内。
- **P**：策略网络的输出，表示同一父节点下子节点的 N 值反馈情况。举例来说，如果父节点有三个子节点 A、B、C，且它们的 N 值分别为 400、200、200，则 P 值为 `[0.5, 0.25, 0.25]`。
- **U**：节点的欠搜索程度，计算公式为 `k * P * sqrt(N') / N`，其中 `N'` 是同一父节点下同层子节点 N 值的总和，`k` 是一个调整参数，随着训练次数的增加而减小。
- **V**：模型认为该节点的单次得分或价值。

## 2. 训练过程

训练过程分为两个阶段：

### 第一阶段：深度优先搜索

1. **目标**：通过深度优先的方式进行对局，大约进行 100,000 轮对局，更新节点的 `W` 值（每次更新为 +1 或 -1）。
   
2. **理由**：由于缺乏关于 `V` 的客观记录，采用深度优先搜索以确保训练的节点能被反复更新。每个节点的 `Q` 值会尽可能准确地反映该节点的实际得分。

3. **参数更新**：在第一阶段，节点的 `N`、`W`、`Q`、`P`、`U` 值都会被计算和更新，而 `V` 值将与 `Q` 保持一致。

4. **结束条件**：经过 100,000 轮对局后，第一阶段结束。此时，可以根据 `Q` 值来训练模型，生成独立的 `V` 值，进入第二阶段。

### 第二阶段：模型自我生成 V 值

1. **目标**：此阶段训练没有固定上限，训练次数越多效果越好。AI 选择节点的依据是 `V + U` 最大的节点。

2. **节点更新**：
   - **中间节点**：若选中的节点是中间节点（没有明确的胜/负/平结果），则更新路径上所有节点的 `W` 和 `N` 值（`W` 更新方式为若白棋走为+V，若黑棋走为-V）。
   - **叶子节点**：若选中的节点是叶子节点（有明确的胜/负/平结果），则更新路径上所有节点的 `W` 和 `N` 值（`W` 更新方式为 +1、0、-1，交替进行）。

3. **多线程处理**：为了提高训练效率，采用多线程处理，每个线程的选择需要添加随机噪声，避免完全一致的搜索路径。

4. **死锁保护**：若一个线程在更新节点时，其他线程需要等待，避免死锁。

5. **欠搜索调整**：随着训练次数的增加，欠搜索的重要性逐渐减小。`k` 参数会随着训练次数的增加逐步减小，可使用反比例或指数函数来调整。

6. **数据流与记录**：每一轮对局结束后，记录结果并以 `.csv` 文件形式保存，方便后续的训练和数据分析。

### 训练流程总结

- **阶段一**：通过深度优先搜索更新节点的 `W`、`N`、`Q`、`P`、`U` 值，`V` 值与 `Q` 保持一致。
- **阶段二**：模型自我训练，生成 `V` 值，并选择 `V + U` 最大的节点。多线程与死锁保护确保训练效率。

## 3. CNN模型与输入格式

- **输入数据**：使用 12 根管道表示棋局状态，每根管道是 8x8 的二维矩阵，表示棋盘上的不同棋子。最终输入为一个形状为 `8x8x12` 的三维矩阵。
  
- **V 和 P 的处理**：在训练过程中，`V` 和 `P` 使用两套不同的卷积核进行处理。父节点通过子节点的状态计算出对应的 `V` 和 `P` 值，然后选出 `V + P` 最大的子节点。

## 4. 参数调整

- **k 的调整**：`k` 参数用于调整欠搜索的重要性，随着训练进程的推进，需要逐渐减小。k = 3 / ((x % 10000) ^ 1.5 + 0.5) + 1，其中 `x` 代表训练次数。

## 5. 总结

通过深度优先的搜索和模型自我生成 `V` 值的方式，结合 CNN 来处理输入数据和更新参数，本模型能够不断改进并生成越来越准确的棋局评估与策略输出。通过多线程和噪声干扰，提升训练效率，保证了模型的多样性与稳定性。
