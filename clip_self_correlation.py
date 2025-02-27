import os
from einops import rearrange
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.ndimage import median_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import InterpolationMode
NEAREST = InterpolationMode.NEAREST

# from detectron2.projects.point_rend.point_features import point_sample

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import clip_utils
from configs.dataset_cfg import dataset_info, prompt_templates
from configs.metric import scores

from skimage import graph, segmentation
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.sparse import csgraph
from collections import defaultdict

device = "cuda"

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform1(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
    ])

def _transform2():
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def gt_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=NEAREST),
        CenterCrop(n_px),
        # ToTensor(),
    ])

def get_image_name_list(dataset, image_path):
    if dataset=="CITYS19":
        city_list = os.listdir(image_path)
        image_file_list = []
        for c in city_list:
            new_image_path = image_path+'/'+c
            image_file_list = image_file_list+os.listdir(new_image_path)
    else:
        image_file_list = os.listdir(image_path)
    return image_file_list

def get_image_and_gt(dataset, image_size, image_path, gt_path, file_name):
    # file name
    gt_suffix = '.png'
    if dataset in ["ADEfull","PC459"]:
        gt_suffix = '.tif'
    if dataset == "CITYS19":
        city_name = file_name.split('_')[0]
        image_file = image_path+'/'+city_name+'/'+file_name
        gt_file = gt_path+'/'+city_name+'/'+file_name.replace('leftImg8bit.png','gtFine_labelTrainIds.png')
    else:
        image_file = image_path+'/'+file_name
        gt_file = gt_path+'/'+file_name.split('.')[0]+gt_suffix
    # load image
    img = Image.open(image_file)
    img = _transform1(image_size)(img)
    gt = gt_transform(image_size)(Image.open(gt_file))
    gt = np.array(gt)
    gt = gt.astype(np.int16) # 防止溢出
    gt = torch.tensor(gt).to(device)
    return img, gt

def get_text_features(clip,text_labels):
    text_features = []
    for qw in text_labels:
            query = clip_utils.tokenize([temp(qw) for temp in prompt_templates]).to(device)
            feature = clip.encode_text(query)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.mean(dim=0)
            feature /= feature.norm()
            text_features.append(feature.unsqueeze(0))
    text_features = torch.cat(text_features, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features # c,d

def get_visual_features(clip, images):
    visual_features, attn_weights = clip.encode_image(images)
    # b,hw+1,512     12,b,hw+1,hw+1
    visual_features = visual_features/visual_features.norm(dim=-1, keepdim=True)
    return visual_features, attn_weights

def build_color_rag(image, n_segments=500, compactness=10):
    """
    根据平均颜色构建 RAG（Region Adjacency Graph）
    :param image: 输入图像，形状为 (H, W, 3)
    :param n_segments: SLIC 超像素数量
    :param compactness: SLIC 的紧致度参数
    :return: RAG 图
    """

    labels = segmentation.slic(image, n_segments=n_segments, compactness=compactness, start_label=0)
    rag = graph.rag_mean_color(image, labels)
    
    return rag, labels

def build_rag_based_on_avg_color(image, patch_size=16):
    """
    根据平均颜色构建 RAG（Region Adjacency Graph）
    :param image: 输入图像，形状为 (H, W, 3)，假设 H 和 W 可被 patch_size 整除
    :param patch_size: 每个小块的大小
    :return: RAG 图
    """
    H, W, _ = image.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # 生成 labels，每个 16x16 小块一个标签
    labels = np.zeros((H, W), dtype=np.int32)
    label = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            labels[i:i+patch_size, j:j+patch_size] = label
            label += 1
    
    # 构建 RAG 图
    rag = graph.rag_mean_color(image, labels)
    
    return rag, labels

def build_feature_rag(vit_features, image_size=(224, 224), patch_size=16):
    """
    根据 ViT 提取的特征构建基于 Cosine Similarity 的 RAG，并返回 superpixel labels。
    :param vit_features: ViT 提取的特征，形状为 (196, 512)
    :param image_size: 输入图像的大小 (H, W)
    :param patch_size: 每个 RAG 区域的大小
    :return: RAG 图, superpixel labels
    """
    H, W = image_size
    grid_size = H // patch_size  # 14x14 网格
    
    # 生成 superpixel labels
    labels = np.arange(grid_size * grid_size).reshape((grid_size, grid_size))
    
    # 计算 Cosine Similarity 作为边权重
    similarity_matrix = cosine_similarity(vit_features.detach().cpu().numpy())
    
    # 构建 RAG，格式与 skimage.graph.rag_mean_color 类似
    rag = graph.RAG()
    for i in range(grid_size * grid_size):
        rag.add_node(i, feature=vit_features[i], labels=[i])
    
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            if j + 1 < grid_size:
                right_id = node_id + 1
                weight = 1 - similarity_matrix[node_id, right_id]  # Cosine 距离
                rag.add_edge(node_id, right_id, weight=weight)
            if i + 1 < grid_size:
                bottom_id = node_id + grid_size
                weight = 1 - similarity_matrix[node_id, bottom_id]  # Cosine 距离
                rag.add_edge(node_id, bottom_id, weight=weight)
    
    return rag, labels

def smooth_uncertainty_with_rag(uncertainty, rag, superpixel_labels, lambda_=0.5, iterations=10):
    """
    使用超像素 RAG 进行区域级别的拉普拉斯平滑，而不是单独平均后再平滑。

    参数：
    - uncertainty: (H, W) 形状的不确定性熵
    - rag: networkx Graph, 表示超像素区域邻接图
    - superpixel_labels: (H, W) 形状，每个像素的超像素索引
    - lambda_: 平滑系数 (0 < lambda_ < 1)
    - iterations: 迭代次数，控制平滑程度

    返回：
    - smoothed_uncertainty: (H, W) 形状，平滑后的不确定性熵
    """
    H, W = uncertainty.shape
    unique_labels = np.unique(superpixel_labels)
    num_superpixels = len(unique_labels)

    # 创建超像素区域的熵映射
    superpixel_uncertainty = uncertainty.copy()

    # 超像素内部进行局部平滑
    for _ in range(iterations):
        new_uncertainty = superpixel_uncertainty.copy()

        for label in unique_labels:
            mask = (superpixel_labels == label)

            # 仅在超像素内部进行平滑 (使用超像素内部均值)
            if np.any(mask):
                new_uncertainty[mask] = (1 - lambda_) * superpixel_uncertainty[mask] + lambda_ * superpixel_uncertainty[mask].mean()

        superpixel_uncertainty = new_uncertainty

    # 通过 RAG 进行超像素间的平滑
    L = nx.normalized_laplacian_matrix(rag).toarray()
    superpixel_flat = np.array([superpixel_uncertainty[superpixel_labels == label].mean() for label in unique_labels])

    # 迭代拉普拉斯平滑
    for _ in range(iterations):
        superpixel_flat = (1 - lambda_) * superpixel_flat + lambda_ * L @ superpixel_flat

    # 传播回像素级别
    smoothed_uncertainty = np.zeros_like(uncertainty)
    for idx, label in enumerate(unique_labels):
        mask = (superpixel_labels == label)
        smoothed_uncertainty[mask] = superpixel_flat[idx]

    return smoothed_uncertainty

def adjust_posterior_with_entropy(P, H_opt, lambda_=1.0, epsilon=0.1):
    """
    根据调整后的熵 H_opt 重新调整类别后验概率 P。

    参数：
    - P: (C, H, W) 形状的类别后验概率
    - H_opt: (H, W) 形状的熵
    - lambda_: 控制熵对温度的影响
    - epsilon: 防止温度过小的稳定项

    返回：
    - P_new: 重新调整后的后验概率 (C, H, W)
    """
    C, H, W = P.shape

    # 计算温度 T(x, y) = λH_opt + ε
    T = lambda_ * H_opt + epsilon  # 形状 (H, W)

    # 计算指数调整后的概率 P_c^(1/T)
    P_adj = P ** (1 / T[None, :, :])  # 广播到 (C, H, W)

    # 归一化为概率分布
    P_new = P_adj / np.sum(P_adj, axis=0, keepdims=True)

    return P_new

def compute_superpixel_weights(rag, superpixel_labels):
    """
    计算每个超像素区域的总边权重
    :param rag: Region Adjacency Graph
    :param superpixel_labels: 超像素标签图
    :return: 每个超像素区域的总边权重
    """
    # 计算每个节点的边权重之和
    node_weights = {node: sum(rag[node][nbr]['weight'] for nbr in rag.neighbors(node)) for node in rag.nodes()}
    
    # 计算每个超像素区域的总边权重
    superpixel_weights = {node: node_weights[node] for node in rag.nodes()}
    
    return superpixel_weights

def compute_superpixel_max(rag, superpixel_labels):
    """
    计算每个超像素区域的总边权重
    :param rag: Region Adjacency Graph
    :param superpixel_labels: 超像素标签图
    :return: 每个超像素区域的总边权重
    """
    # 计算每个节点的边权重之和
    node_weights = {node: max(rag[node][nbr]['weight'] for nbr in rag.neighbors(node)) for node in rag.nodes()}
    
    # 计算每个超像素区域的总边权重
    superpixel_weights = {node: node_weights[node] for node in rag.nodes()}
    
    return superpixel_weights

def visualize_superpixel_weights(image, superpixel_labels, superpixel_weights):
    """
    可视化超像素区域的边权重之和
    :param image: 原始图像
    :param superpixel_labels: 超像素标签
    :param superpixel_weights: 每个超像素的边权重之和
    """
    # 归一化超像素边权重到 0-1
    max_weight = max(superpixel_weights.values())
    min_weight = min(superpixel_weights.values())
    
    norm_weights = {sp: (superpixel_weights[sp] - min_weight) / (max_weight - min_weight) for sp in superpixel_weights}
    
    # 生成颜色映射
    cmap = plt.cm.viridis
    colored_seg = np.zeros_like(image, dtype=np.float32)

    for sp in np.unique(superpixel_labels):
        mask = superpixel_labels == sp
        color = cmap(norm_weights[sp])[:3]  # 获取 RGB 颜色
        colored_seg[mask] = color
    
    # 显示图像
    plt.figure(figsize=(10, 6))
    plt.imshow(colored_seg)
    plt.axis('off')
    plt.title("Superpixel Edge Weight Sums")
    plt.show()

def self_clip(clip, dataset, image_size=224,eps=0.7,min=3):
    # text feature
    cls_text_list = dataset_info[dataset]["labels"]
    text_features = get_text_features(clip, cls_text_list)
    C = text_features.shape[0]
    with_bg = False
    if dataset in ['COCO80_val','VOC21','PC60']: # with background
        with_bg = True
        bg_text_list = dataset_info[dataset]["background"]
        bg_features = get_text_features(clip, bg_text_list)
        BG = bg_features.shape[0]
        text_features = torch.cat([text_features,bg_features],dim=0) # C+bg, 512
    # image
    image_path = dataset_info[dataset]["image_path"]
    gt_path = dataset_info[dataset]["gt_path"]
    image_name_list = get_image_name_list(dataset,image_path)
    # calculate the semantic segmentation
    cal_pred = []
    cal_gt = []
    for pi in tqdm(range(len(image_name_list))):
        ori_images, gts = get_image_and_gt(dataset, image_size, image_path, gt_path, image_name_list[pi])
        ori_h, ori_w = gts.shape[-2:]
        images = _transform2()(ori_images).to(device)
        label_set = torch.unique(gts.reshape(-1))
        # ignore background
        if label_set[-1].float==65535:
            label_set = label_set[:-1]
        elif label_set[-1]==255:
            label_set = label_set[:-1]
        else:
            pass
        if label_set.shape[0]==0: # none gt, ignore
            continue
        # visual features
        patch_window = 16
        h,w = images.shape[-2]//patch_window, images.shape[-1]//patch_window
        visual_features, attn_weights = get_visual_features(clip,images.unsqueeze(0))
        cls_token = visual_features[0,0:1] # 1,512
        patch_tokens = visual_features[0,1:] # hw,512
        cls_weights = attn_weights[-2,0,0:1,1:] # 1,hw
        attn_weights = attn_weights[-2,0,1:,1:] # hw,hw
        # global patch denoise, can replace the dice loss denoise
        attn_weights = attn_weights-cls_weights
        attn_weights[attn_weights<0] = 0 # hw,hw
        # logits and preds
        patch_logits = patch_tokens@text_features.T * clip.logit_scale.exp() # 196, C
        patch_logits = patch_logits.permute(1,0).unsqueeze(0).reshape(1,text_features.shape[0],h,w)
        patch_logits = F.interpolate(
            patch_logits,size=(ori_h, ori_w),mode='bilinear',align_corners=False,
        )[0] # C,H,W
        if dataset=='VOC20':
            cls_logits = cls_token@text_features.T * clip.logit_scale.exp() # 1,c
            patch_logits = patch_logits*cls_logits[0].unsqueeze(1).unsqueeze(1)
        patch_preds = patch_logits.argmax(dim=0) # H,W

        if with_bg:
            patch_preds[patch_preds>=C] = C
            gts[gts==255] = C
        # clusters
        dbscan = DBSCAN(eps=eps, min_samples=min)
        labels = dbscan.fit_predict(attn_weights.detach().cpu().numpy())
        labels = torch.from_numpy(labels).to(device)
        db_label_set = torch.unique(labels)
        if db_label_set.shape[0]==1 and db_label_set[0]==-1:
            # no clusters, continue
            cal_pred.append(patch_preds.cpu().numpy())
            cal_gt.append(gts.cpu().numpy())
            # print('no clusters!')
            continue
        if db_label_set[0]==-1:
            db_label_set = db_label_set[1:]
        # clusters post process
        cluster_gts = []
        for l in range(db_label_set.shape[0]):
            temp_attn = attn_weights[labels==db_label_set[l]] # n,l
            temp_attn = temp_attn.mean(dim=0)
            cluster_gts.append(temp_attn)
        cluster_gts = torch.stack(cluster_gts,dim=0) # n,l
        cluster_gts = cluster_gts.reshape(cluster_gts.shape[0],h,w) # n,h,w
        # smooth
        ratio = 4
        cluster_gts = F.interpolate(
            cluster_gts.unsqueeze(0),size=(ratio*h,ratio*w),mode='bilinear',align_corners=False
        )[0] # n,224,224
        cluster_gts = cluster_gts.detach().cpu().numpy()
        for i in range(cluster_gts.shape[0]):
            cluster_gts[i] = median_filter(cluster_gts[i],size=ratio*2-1)
        cluster_gts = torch.from_numpy(cluster_gts)
        cluster_gts = cluster_gts.to(device)
        cluster_gts = F.interpolate(
            cluster_gts.unsqueeze(0),size=(image_size,image_size),mode='bilinear',align_corners=False
        )[0] # n,H,W

        # 基于rag的边的权重，certainty map，去修正模型预测结果的不确定性，再温度修正预测后验概率
        #############################
        rag, superpixel_labels = build_color_rag(np.array(ori_images), n_segments=1000, compactness=10)
        p_ = cluster_gts.softmax(dim=0)
        uncertainty = -torch.sum(p_ * torch.log(p_), dim=0)

        # smoothed_uncertainty = smooth_uncertainty_with_rag(uncertainty.detach().cpu().numpy(), rag, superpixel_labels, lambda_=0.5, iterations=1)
        # cluster_gts = adjust_posterior_with_entropy(p_.detach().cpu().numpy(), smoothed_uncertainty, lambda_=1, epsilon=0.1)
        # cluster_gts = torch.as_tensor(cluster_gts).to("cuda")

        superpixel_weights = compute_superpixel_weights(rag, superpixel_labels)
        max_weight = np.max(list(superpixel_weights.values()))
        min_weight = np.min(list(superpixel_weights.values()))
        
        norm_weights = {sp: (superpixel_weights[sp] - min_weight) / (max_weight - min_weight) for sp in superpixel_weights}
        
        edge_seg = np.zeros(uncertainty.shape, dtype=np.float32)
        for patch_id in range(len(np.unique(superpixel_labels))):
            mask = (superpixel_labels == patch_id)
            edge_seg[mask] = norm_weights[patch_id]

        uncertainty_weighted = uncertainty + uncertainty * torch.tensor(edge_seg).to("cuda")

        cluster_gts = adjust_posterior_with_entropy(p_.detach().cpu().numpy(), uncertainty_weighted.detach().cpu().numpy(), lambda_=1, epsilon=0.1)
        cluster_gts = torch.as_tensor(cluster_gts).to("cuda")        

        
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(uncertainty.detach().cpu().numpy(), cmap='jet')
        # plt.subplot(1,3,2)
        # plt.imshow((uncertainty * torch.tensor(edge_seg).to("cuda")).detach().cpu().numpy(), cmap='jet') 
        # plt.subplot(1,3,3)
        # plt.imshow(edge_seg, cmap='jet')
        # plt.savefig("x.png")       
        #############################

        cluster_gts = cluster_gts.argmax(dim=0)
        # vote
        for gt in range(db_label_set.shape[0]):
            mask_preds = patch_preds[cluster_gts==db_label_set[gt]] # n,
            unique_val, counts = torch.unique(mask_preds, return_counts = True)
            if counts.shape[0]==0:
                continue
            pred_label = unique_val[counts.argmax()]
            # pred_label = cluster_preds[gt]
            patch_preds[cluster_gts==db_label_set[gt]] = pred_label

        if with_bg:
            patch_preds[patch_preds>=C] = C
            gts[gts==255] = C
        cal_pred.append(patch_preds.cpu().numpy())
        cal_gt.append(gts.cpu().numpy())
    if with_bg:
        metric_scores = scores(cal_gt, cal_pred, C+1)
    else:
        metric_scores = scores(cal_gt, cal_pred, C)
    print('dataset:',dataset,'image_size:',image_size, 'eps:',eps,'min:',min)
    print('results:',metric_scores)

def self_clip_test():
    clip_type = "ViT-B/16"
    clip_model, _ = clip_utils.load(clip_type, image_size=224) # origin transforms unused
    clip_model = clip_model.to(device)
    print('load clip success!')
    datasets = ["COCO171_val"]
    # datasets = ["VOC20","VOC21","COCO80_val","COCO171_val","PC59","PC60","PC459","ADE150","ADEfull"]
    # datasets = ["VOC20","ADE150","ADEfull","COCO171_val","PC59", "PC459"]
    # datasets = ["VOC21", "COCO80_val", "PC60"]
    for d in datasets:
        self_clip(clip_model, d, image_size=224,eps=0.7,min=3)
        # self_clip(clip_model, d, image_size=336,eps=1.1,min=7)

if __name__=="__main__":
    with torch.no_grad():
        self_clip_test()
