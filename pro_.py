import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import maxflow
from skimage import graph
from skimage.segmentation import slic
from skimage.measure import regionprops
from scipy.stats import mode
from skimage import segmentation
import matplotlib.pyplot as plt
from collections import Counter

def compute_superpixel_mode(pred, superpixels):
    """
    计算每个 superpixel 的主类别（出现次数最多的类别）
    """
    unique_labels = np.unique(superpixels)
    superpixel_modes = {}

    for label in unique_labels:
        mask = superpixels == label
        superpixel_modes[label] = mode(pred[mask], axis=None)[0][0]  # 获取 mode 类别

    return superpixel_modes

# def build_rag(image, superpixels):
#     """
#     构建基于 superpixel 的 Region Adjacency Graph (RAG)
#     """
#     rag = graph.rag_mean_color(image, superpixels, mode='distance')
#     return rag


def reassign_labels_by_min_weight(rag, superpixels):
    """
    遍历 RAG 的每个节点，按照最小权重邻居的 sem_label 重新分配标签，并返回更新后的 mask。
    该函数不会修改原始的 rag，而是操作一个副本。

    参数：
    - rag: 超像素的 Region Adjacency Graph (networkx.Graph)
    - superpixels: 超像素区域索引 (224x224)，每个像素对应一个 superpixel ID

    返回：
    - updated_mask: 224x224 数组，每个像素对应更新后的 sem_label
    - rag_copy: 复制的 RAG，其中节点的 sem_label 经过更新
    """
    rag_copy = rag.copy()  # 复制原始 RAG，避免修改原始数据
    updated_labels = {}

    for node in rag_copy.nodes():
        neighbors = list(rag_copy.neighbors(node))  # 获取当前节点的所有邻居
        
        if not neighbors:  # 如果没有邻居，保持原标签
            updated_labels[node] = rag_copy.nodes[node].get('sem_label', None)
            continue
        
        # 选择权重最小的邻居
        min_weight_neighbor = min(
            neighbors, key=lambda n: rag_copy.edges[node, n].get('weight', float('inf'))
        )
        
        # 取最小权重邻居的 sem_label
        updated_labels[node] = rag_copy.nodes[min_weight_neighbor].get('sem_label', None)

    # 更新 RAG 复制体中的节点标签
    for node, new_label in updated_labels.items():
        rag_copy.nodes[node]['sem_label'] = new_label

    # 生成新的 mask
    updated_mask = np.zeros_like(superpixels, dtype=np.int32)
    unique_labels = np.unique(superpixels)

    for label in unique_labels:
        mask = superpixels == label
        updated_mask[mask] = rag_copy.nodes[label].get('sem_label', 0)  # 默认值为 0，防止缺失

    return updated_mask, rag_copy




def compute_superpixel_mode_(pred, superpixels, rag):
    """
    计算每个 superpixel 的主类别（出现次数最多的类别），并返回可视化的 mask。
    
    参数：
    - pred: 预测结果 (224x224)，类别数为 10
    - superpixels: 超像素区域索引 (224x224)，每个像素对应一个 superpixel ID
    
    返回：
    - mode_mask: 224x224 数组，每个 superpixel 取其 mode 类别
    """
    unique_labels = np.unique(superpixels)
    mode_mask = np.zeros_like(superpixels)  # 用于存储 superpixel 主要类别的 mask

    for label in unique_labels:
        mask = superpixels == label
        most_common_label = mode(pred[mask], axis=None)[0][0]  # 获取 mode 类别
        mode_mask[mask] = most_common_label  # 赋值回整个 superpixel 区域

        rag.nodes[label]['sem_label'] = most_common_label

    return mode_mask, rag

def optimize_labels_with_graph_cut(superpixels, rag, initial_labels):
    """
    使用 NetworkX 的最小割进行 Graph Cut 处理
    """
    G = nx.Graph()

    # 添加节点
    for node in rag.nodes:
        G.add_node(node, label=initial_labels[node])

    # 添加边（基于 RAG 计算的权重）
    for n1, n2, edge_data in rag.edges(data=True):
        weight = edge_data['weight']
        if weight <= 0: 
            weight = 1e-3  # 避免无效边
        G.add_edge(n1, n2, weight=weight)

    # 选择合适的源点和汇点
    sorted_labels = sorted(initial_labels.items(), key=lambda x: x[1])
    source, sink = sorted_labels[0][0], sorted_labels[-1][0]  # 选择最小和最大的类别超像素

    # 确保 source 和 sink 不是直接连接的
    if G.has_edge(source, sink):
        G[source][sink]['weight'] = np.clip(G[source][sink]['weight'], 1e-3, 50)

    # 计算最小割
    cut_value, partition = nx.minimum_cut(G, source, sink)

    # 获取优化后的 superpixel 标签
    optimized_labels = {}
    for part in partition:
        for node in part:
            optimized_labels[node] = initial_labels[node]

    return optimized_labels

def get_neighbors_with_weights(rag, thres):
    neighbors_info = {}
    nodes_labels = {}
    new_nodes_labels = {}

    reassigned_labels = {}
    
    for node in rag.nodes:
        first_layer_neighbors = list(rag.neighbors(node))
        first_layer_weights = {n: rag[node][n]['weight'] for n in first_layer_neighbors}
        
        second_layer_neighbors = set()
        second_layer_weights = {}
        
        for first_neighbor in first_layer_neighbors:
            for second_neighbor in rag.neighbors(first_neighbor):
                if second_neighbor != node and second_neighbor not in first_layer_neighbors:
                    second_layer_neighbors.add(second_neighbor)
                    second_layer_weights[second_neighbor] = rag[first_neighbor][second_neighbor]['weight']
        
        neighbors_info[node] = {
            "first_layer_neighbors": first_layer_neighbors,
            "first_layer_weights": first_layer_weights,
            "second_layer_neighbors": list(second_layer_neighbors),
            "second_layer_weights": second_layer_weights,
        }

        nodes_labels[node] = rag.nodes[node]['sem_label']


    for node in rag.nodes:
        label_set_first_order = [nodes_labels[i] for i in neighbors_info[node]['first_layer_neighbors']]
        label_set_second_order = [nodes_labels[i] for i in neighbors_info[node]['second_layer_neighbors']]

        is_first_order_same = all(label == nodes_labels[node] for label in label_set_first_order)
        is_second_order_same = all(label == nodes_labels[node] for label in label_set_second_order)

        if is_first_order_same:
            new_nodes_labels[node] = {"label": nodes_labels[node], "weight": None}
        else:
            first_layer_weights = neighbors_info[node]['first_layer_weights'].values()
            first_layer_weights = np.array(list(first_layer_weights))[label_set_first_order != nodes_labels[node]]
            
            counter = Counter(np.array(label_set_first_order)[label_set_first_order != nodes_labels[node]])
            most_common_label, most_common_count = counter.most_common(1)[0]

            new_nodes_labels[node] = {"label": most_common_label, "weight": np.mean(first_layer_weights)}

    for key, val in new_nodes_labels.items():
        if val['weight'] == None:
            reassigned_labels[key] = nodes_labels[key]
        elif val['weight'] < thres:
            reassigned_labels[key] = val['label']

    return reassigned_labels

if __name__ == "__main__":

    cluster_gts = np.load("./cluster_gts.npz")['arr_0']
    img = np.load("./image.npz")['arr_0']

    superpixels = slic(img, n_segments=1000, compactness=10)
    edges = segmentation.find_boundaries(superpixels, mode='thick')

    rag = graph.rag_mean_color(img, superpixels)

    a_labels, rag = compute_superpixel_mode_(cluster_gts, superpixels, rag)

    updated_labels, rag_copy = reassign_labels_by_min_weight(rag, superpixels)

    reassigned_labels = get_neighbors_with_weights(rag, thres=3)

    re_labeled_pred = np.zeros_like(superpixels)  # 用于存储 superpixel 主要类别的 mask

    for label, label_ in reassigned_labels.items():
        mask = superpixels == label
        re_labeled_pred[mask] = label_

    plt.subplot(1,4,1)
    plt.imshow(cluster_gts)
    plt.subplot(1,4,2)
    plt.imshow(a_labels)
    plt.subplot(1,4,3)
    plt.imshow(updated_labels) 
    plt.subplot(1,4,4)
    plt.imshow(re_labeled_pred)         


    