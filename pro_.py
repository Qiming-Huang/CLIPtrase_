import numpy as np

def slerp(p1, p2, alpha):
    """
    在球面上进行插值 (Spherical Linear Interpolation, SLERP)
    :param p1: 起始点向量 (numpy array)
    :param p2: 目标点向量 (numpy array)
    :param alpha: 插值因子，0 <= alpha <= 1
    :return: 插值后的向量
    """
    # 归一化向量，确保它们在单位球面上
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)

    # 计算夹角 θ
    dot_product = np.clip(np.dot(p1, p2), -1.0, 1.0)  # 避免数值误差
    theta = np.arccos(dot_product)

    # 计算权重
    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:  # 避免除零错误，如果夹角太小，则用 LERP 近似
        return (1 - alpha) * p1 + alpha * p2
    
    # SLERP 计算
    p_interp = (np.sin((1 - alpha) * theta) / sin_theta) * p1 + (np.sin(alpha * theta) / sin_theta) * p2
    return p_interp

# 示例: 512 维 feature embeddings
np.random.seed(42)
p4 = np.random.rand(512)  # 原始点
v3 = np.random.rand(512)  # 目标点 (希望 p4 靠近)
v4 = np.random.rand(512)  # 远离的点

# 选择 alpha 值，控制 p4 向 v3 方向移动
alpha = 0.3  # 例如让 p4 以 30% 权重向 v3 旋转

# 计算新的 p4'
p4_new = slerp(p4, v3, alpha)

# 输出
print("Original p4:", p4[:5])  # 仅显示前 5 维度
print("New p4 (moved towards v3):", p4_new[:5])
