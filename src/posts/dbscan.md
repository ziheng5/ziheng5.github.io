---
title: DBSCAN 密度聚类分析算法
date: 2025-04-15 00:11:22
tags:
    - 机器学习
categories: 
    - 机器学习
description: |
    📚 记录了小生最近学的新聚类算法
---
## 1. DBSCAN 算法简介
---
DBSCAN 是一种基于密度的聚类算法，旨在发现任意形状的**簇**，并且对**噪声点**（**outliers**）具有鲁棒性。

DBSCAN 通过在数据空间中找到高密度区域，将这些区域作为簇，同时把孤立点（密度低的点）归为**噪声**。

DBSCAN 的基本思想是：
- 在某个点的**邻域半径**（$\epsilon$，**epsilon**）内，如果有足够多的点（超过一个**阈值 minPts**），就认为这个区域是一个高密度区域，可以扩展成一个簇。
- 一个簇通过密度相连（density-connected）的点进行扩展。
- 无法归属于任何簇的点被认为是噪声点。

> 可以前往 [DBSCAN 数据可视化网站](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)来了解执行的流程。

![dbscan_example](/images/machine_learning/dbscan1.png)

## 2. DBSCAN 的基本概念
---
- $\epsilon$-邻域（Epsilon-neighborhood）
    对于某个点 $P$，以半径 $\epsilon$ 为边界的区域内所有的点称为该点的 $\epsilon$ 邻域。
- **核心点**（**Core Point**）
    如果一个点 $P$ 的 $\epsilon$ 邻域内至少有 **minPts** 个点（包括 $P$ 自己），那么它被称为核心点。
- **边界点**（**Border Point**）
    如果一个点 $P$ 在某个核心的 $\epsilon$ 邻域内，但自身不是核心点，它被称为边界点。
- **噪声点**（**Noise Point**）
    如果一个点既不是核心点，也不属于任何核心点的领域，它被认为是噪声点。
- **密度直达**（**Directly Density-Reachable**）
    如果点 $P$ 是核心点，并且点 $Q$ 在 $P$ 的 $\epsilon$ 邻域内，那么 $Q$ 被称为 $P$ 的密度直达。
- **密度可达**（**Density-Reachable**）
    如果存在一条核心点链表（$P_1 \rightarrow P_2 \rightarrow ... \rightarrow P_n$），使得每个点从前一个点密度直达，且 $P_1=P$，$P_n=Q$，则 Q 是从 P 密度可达的。
- **密度相连**（**Density-Connected**）
    如果存在一个点 $O$，使得 $P$ 和 $Q$ 都从 $O$ 密度可达，则称 $P$ 和 $Q$ 是密度相连的。

## 3. DBSCAN 算法步骤
---
- **初始化**
    从数据集中任意选择一个点 $P$，判断它是否为核心点（即 $\epsilon$ 邻域内是否包含至少 minPts 个点）。
- **扩展簇**
    如果 $P$ 是核心点，则开始一个新簇，将 $P$ 及其邻域中的点加入簇中，并不断对新的核心点的邻域进行扩展。
- **处理噪声点**
    如果一个点既不在任何簇中，也不满足成为核心点的条件，则将其标记为噪声点。
- **重复处理**
    继续检查所有未访问的点，直到所有点都被访问为止。

## 4. DBSCAN 伪代码
---
```python
DBSCAN(D, epsilon, minPts):
    C = 0   # 初始化簇标签
    for each unvisited point P in dataset D:
        mark P as visited
        Neighbors = getNeighbors(P, epsilon)    # 获取邻域内的所有点
        if size(Neighbors) < minPts:
            mark P as NOISE # 认为该点是噪声
        else:
            C = C + 1   # 创建新簇
            expandCluster(P, Neighbors, C, epsilon, minPts)

expandCluster(P, Neighbors, C, epsilon, minPts):
    add P to cluster C
    for each point Q in Neighbors:
        if Q is not visited:
            mark Q as visited
            NeighborsQ = getNeighbors(Q, epsilon)
            if size(NeighborsQ) >= minPts:
                Neighbors = Neighbors U NeighborsQ  # 扩展簇
        if Q is not yet assigned to any cluster:
            add Q to cluster C
```

## 5. DBSCAN 的时间复杂度分析
---
- **邻域查询**：在每次扩展时，需要查找一个点的 $\epsilon$ 邻域。如果使用 KD-Tree 或 Ball-Tree 等空间索引结构，这个操作的复杂度为 $O(\log n)$。
- **总体复杂度**：如果对每个点进行邻域查询，算法的时间复杂度为 $O(n·\log n)$。如果不使用索引结构，最坏情况下是 $O(n^2)$。

## 6. DBSCAN 的 Python 实现
---

### 方法一：使用 scikit-learn
![cluster](/images/machine_learning/scatter_plot.png)

聚类代码如下

```python
from sklearn.cluster import DBSCAN
import numpy as np

# 生成示例数据
X = np.array([[1, 2], [2, 2], [2, 3],[8, 7], [8, 8], [25, 80]])

# 初始化 DBSCAN 模型
db = DBSCAN(eps=3, min_samples=2).fit(X)

# 获取聚类标签
labels = db.labels_

print("Cluster labels:", labels)
```

输出：

```bash
Cluster labels: [ 0 0 0 1 1 -1]
```

其中：
- 标签为 `-1` 的点表示**噪声点**
- 其他标签表示该点属于的**簇**

### 方法二：手动实现
```python
import numpy as np
from sklearn.metrics import pairwise_distances

class DBSCAN:
    def __init__(self, eps=1.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(self, X):
        # 计算所有点之间的距离矩阵
        distances = pairwise_distances(X)
        
        # 初始化标签：0表示未访问，-1表示噪声
        labels = np.zeros(X.shape[0], dtype=int)
        cluster_id = 0
        
        for i in range(X.shape[0]):
            if labels[i] != 0:  # 已访问过的点跳过
                continue
                
            # 找到当前点的所有邻居
            neighbors = self._find_neighbors(i, distances)
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # 标记为噪声
            else:
                cluster_id += 1
                self._expand_cluster(i, neighbors, labels, cluster_id, distances)
        
        self.labels_ = labels
        return self
    
    def _find_neighbors(self, point_idx, distances):
        """找到给定点的所有ε-邻域内的邻居"""
        return np.where(distances[point_idx] <= self.eps)[0]
    
    def _expand_cluster(self, point_idx, neighbors, labels, cluster_id, distances):
        """从核心点扩展聚类"""
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            current_point = neighbors[i]
            
            if labels[current_point] == -1:  # 如果是噪声，改为边界点
                labels[current_point] = cluster_id
            elif labels[current_point] == 0:  # 如果是未访问点
                labels[current_point] = cluster_id
                
                # 找到当前点的所有邻居
                current_neighbors = self._find_neighbors(current_point, distances)
                
                if len(current_neighbors) >= self.min_samples:
                    # 将新邻居加入列表（合并邻居）
                    neighbors = np.concatenate([neighbors, current_neighbors])
            
            i += 1

# 测试代码
if __name__ == "__main__":
    # 使用你提供的示例数据
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    
    # 创建并运行DBSCAN
    dbscan = DBSCAN(eps=3, min_samples=2)
    dbscan.fit(X)
    
    # 打印结果
    print("聚类结果:", dbscan.labels_)
    
    # 可视化结果
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i in range(len(X)):
        if dbscan.labels_[i] == -1:
            plt.scatter(X[i, 0], X[i, 1], c='black', marker='x', s=100, label='Noise' if i == 0 else "")
        else:
            plt.scatter(X[i, 0], X[i, 1], c=colors[dbscan.labels_[i] - 1], 
                       label=f'Cluster {dbscan.labels_[i]}' if i == 0 or dbscan.labels_[i] != dbscan.labels_[i-1] else "")
    
    plt.title('DBSCAN Clustering Result')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 7. DBSCAN 的优缺点
---
- 优点：
  - 可以发现任意形状的簇。
  - 不需要预先指定簇的数量。
  - 对噪声有鲁棒性。
- 缺点
  - 当簇的密度差异较大时，效果不佳。
  - **高维数据**（超过 10 维）中的性能较差，非常消耗 CPU 和 GPU 以及内存性能。
  - 需要合理选择 $\epsilon$ 和 minPts 参数。