import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成随机视觉特征和文本特征
torch.manual_seed(42)
visual_features = torch.rand(196, 512, device=device)  # 196个视觉特征
text_features = torch.rand(171, 512, device=device)    # 171个文本特征

# **将特征归一化到单位球面上**
def normalize_to_sphere(features):
    return features / features.norm(dim=1, keepdim=True)

visual_features = normalize_to_sphere(visual_features)
text_features = normalize_to_sphere(text_features)

# **计算球面上的余弦相似度**
def cosine_similarity_matrix(A, B):
    return torch.matmul(A, B.T)  # 余弦相似度计算

similarity_matrix = cosine_similarity_matrix(visual_features, text_features)

# **转换为距离矩阵**（因为OT最小化传输距离）
cost_matrix = 1 - similarity_matrix  # 1 - 余弦相似度，确保最小化传输成本

# **定义均匀分布权重**
a = torch.ones(196, device=device) / 196  # 视觉特征的均匀权重
b = torch.ones(171, device=device) / 171  # 文本特征的均匀权重

# **Sinkhorn-Knopp OT**
def sinkhorn(a, b, cost_matrix, epsilon=0.01, n_iter=50):
    K = torch.exp(-cost_matrix / epsilon)  # 计算 kernel 矩阵
    u = torch.ones_like(a) / len(a)  # 初始化 u
    v = torch.ones_like(b) / len(b)  # 初始化 v

    for _ in range(n_iter):
        u = a / (K @ v)  # 更新 u
        v = b / (K.T @ u)  # 更新 v

    transport_matrix = torch.diag(u) @ K @ torch.diag(v)  # 计算最终传输矩阵
    return transport_matrix

# 计算最优传输矩阵
optimal_transport = sinkhorn(a, b, cost_matrix)

# **获取最佳匹配**
matching_indices = torch.argmax(optimal_transport, dim=1).cpu().numpy()

# **结果展示**
matched_pairs = list(enumerate(matching_indices))  # (视觉索引, 匹配的文本索引)
print("视觉特征与文本特征的最佳匹配对：", matched_pairs)
