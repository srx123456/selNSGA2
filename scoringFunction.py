import numpy as np

# 假设我们有5个设备，每个设备有3个特性（mem，cpu，带宽）
# 0：已使用cpu 1：已使用mem 2：剩余bw  
# 0.47 0.38 0.15 cpu mem  bw
n_samples = 5
n_features = 3

# 这是你的设备数据，你需要将它替换为实际的数据
X = np.random.rand(n_samples, n_features)
print(X)

# 计算每个设备的得分 - 关键在这里，这个参数哪里来的。。。
scores = -0.38 * X[:, 0] - 0.47 * X[:, 1] + 0.15 * X[:, 2]

# 选择得分最高的前k个节点
k = 3
top_k_indices = np.argsort(scores)[-k:]
print(top_k_indices)

# 在这些节点中随机选择一个
selected_index = np.random.choice(top_k_indices)
print("选择的节点: ", selected_index)
