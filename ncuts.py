import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.segmentation import slic
from skimage.graph import rag_mean_color, cut_normalized
from skimage.color import label2rgb


def normalized_cut_with_threshold(image_path, n_segments=100, compactness=10, percentile=75):
    """
    使用 Normalized Cut (Ncut) 进行图像分割，并根据 RAG 边权重的百分位数调整阈值

    :param image_path: str, 输入图像路径
    :param n_segments: int, 预分割的超像素数量
    :param compactness: float, SLIC 超像素的紧凑性参数
    :param percentile: float, 用于设置 RAG 边权重阈值的百分位数 (0-100)
    """
    # 读取图像
    image = io.imread(image_path)

    # 将图像转换为 Lab 颜色空间
    image_lab = color.rgb2lab(image)

    # 进行 SLIC 超像素分割
    segments = slic(image_lab, n_segments=n_segments, compactness=compactness, start_label=1)

    # 构建 RAG (Region Adjacency Graph) 并计算平均颜色
    rag = rag_mean_color(image, segments)

    # 提取 RAG 的边权重
    edge_weights = [data['weight'] for _, _, data in rag.edges(data=True)]
    
    # 计算权重的百分位数阈值
    threshold = np.percentile(edge_weights, percentile)
    print(f"Edge weight threshold (percentile {percentile}%): {threshold:.4f}")

    # 进行 Normalized Cut (Ncut) 并使用计算出的阈值
    ncut_labels = cut_normalized(segments, rag, thresh=threshold)

    # 生成带有分割区域的彩色图像
    segmented_image = label2rgb(ncut_labels, image, kind='avg')

    # 显示原始图像和分割后的图像
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(segmented_image)
    ax[1].set_title(f"Ncut Segmentation (percentile={percentile}%)")
    ax[1].axis("off")

    plt.show()


# 示例调用
image_path = "/home/qiming/Desktop/datasets/coco-stuff/images/val2017/000000056127.jpg"  # 替换为您的图片路径
normalized_cut_with_threshold(image_path, n_segments=1000, compactness=10, percentile=10)
