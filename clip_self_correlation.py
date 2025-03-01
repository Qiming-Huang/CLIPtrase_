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

from skimage import graph, segmentation
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.sparse import csgraph
from collections import defaultdict
import seaborn as sns
from sklearn.cluster import KMeans
from collections import Counter
from scipy.stats import mode

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import clip_utils
from configs.dataset_cfg import dataset_info, prompt_templates
from configs.metric import scores

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
    使用超像素 RAG 进行熵的拉普拉斯平滑。

    参数：
    - uncertainty: (H, W) 形状的不确定性熵
    - rag: networkx Graph, 表示超像素区域邻接图
    - superpixel_labels: (H, W) 形状，每个像素的超像素索引
    - lambda_: 平滑系数 (0 < lambda_ < 1)
    - iterations: 迭代次数，控制平滑程度

    返回：
    - smoothed_uncertainty: (H, W) 形状，平滑后的不确定性熵
    """
    unique_labels = np.unique(superpixel_labels)
    num_superpixels = len(unique_labels)

    # 计算每个超像素区域的初始均值熵
    superpixel_uncertainty = np.zeros(num_superpixels)
    for idx, label in enumerate(unique_labels):
        mask = (superpixel_labels == label)
        superpixel_uncertainty[idx] = uncertainty[mask].mean().item()  # 计算该超像素内的熵均值

    # 获取 RAG 的邻接矩阵
    L = nx.normalized_laplacian_matrix(rag).toarray()

    # 迭代拉普拉斯平滑
    for _ in range(iterations):
        superpixel_uncertainty = (1 - lambda_) * superpixel_uncertainty + lambda_ * L @ superpixel_uncertainty

    # 传播回像素级别
    smoothed_uncertainty = np.zeros_like(uncertainty)
    for idx, label in enumerate(unique_labels):
        mask = (superpixel_labels == label)
        smoothed_uncertainty[mask] = superpixel_uncertainty[idx]

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

        # TODO superpixel-argmax operator
        ###########
        rag, superpixel_labels = build_color_rag(np.array(ori_images), n_segments=1000, compactness=10)
        
        ###########

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

        # 基于edge_map加权每一类对应的attention
        ####################################
        # rag, superpixel_labels = build_color_rag(np.array(ori_images), n_segments=1000, compactness=10)
        # superpixel_weights = compute_superpixel_weights(rag, superpixel_labels)
        # max_weight = np.max(list(superpixel_weights.values()))
        # min_weight = np.min(list(superpixel_weights.values()))
        
        # norm_weights = {sp: (superpixel_weights[sp] - min_weight) / (max_weight - min_weight) for sp in superpixel_weights}
        
        # edge_seg = np.zeros(superpixel_labels.shape, dtype=np.float32)
        # for patch_id in range(len(np.unique(superpixel_labels))):
        #     mask = (superpixel_labels == patch_id)
        #     edge_seg[mask] = 1 - norm_weights[patch_id]
        
        # edge_seg = torch.as_tensor(edge_seg).to("cuda")        
        
        # pseudo_label = torch.where(cluster_gts > 0.5, 1.0, 0.0)
        # for cls_ in range(cluster_gts.shape[0]):
        #     mask = (pseudo_label[cls_] == True)
        #     cluster_gts[cls_][mask] *= edge_seg[mask]


        ####################################

        cluster_gts = cluster_gts.argmax(dim=0).detach().cpu().numpy()

        ###############
        patch_preds = patch_preds.detach().cpu().numpy()
        rag, superpixel_labels = build_color_rag(np.array(ori_images), n_segments=1000, compactness=10)

        unique_labels = np.unique(superpixel_labels)
        mode_mask = np.zeros_like(superpixel_labels)  # 用于存储 superpixel 主要类别的 mask

        for label in unique_labels:
            mask = superpixel_labels == label
            most_common_label = mode(patch_preds[mask], axis=None)[0][0]  # 获取 mode 类别
            mode_mask[mask] = most_common_label  # 赋值回整个 superpixel 区域

            rag.nodes[label]['sem_label'] = most_common_label

        
        reassigned_labels = get_neighbors_with_weights(rag, thres=10)
        re_labeled_pred = np.zeros_like(superpixel_labels)  # 用于存储 superpixel 主要类别的 mask

        for label, label_ in reassigned_labels.items():
            mask = superpixel_labels == label
            re_labeled_pred[mask] = label_
        patch_preds = torch.from_numpy(re_labeled_pred).to("cuda")
        ###############

        # 基于cluster_gts聚类结果再进行一次patch_token 和 text feature matching, region_pred
        ################
        # region_tokens = []
        # patch_tokens_resize = F.interpolate(patch_tokens.reshape((14, 14, 512)).unsqueeze(0).permute(0, 3, 1, 2), (224, 224))
        # patch_tokens_resize = patch_tokens_resize.squeeze().permute(1,2,0)
        
        # for i in torch.unique(cluster_gts):
        #     mask = (cluster_gts == i)
        #     region_tokens.append(
        #         torch.mean(patch_tokens_resize[mask], dim=0)
        #     )
        # region_tokens = torch.stack(region_tokens)
        # region_labels = (region_tokens@text_features.T * clip.logit_scale.exp()).argmax(dim=-1) # 19

        # region_pred = torch.zeros_like(cluster_gts)
        # uni_cluster_gts = torch.unique(cluster_gts)
        # for i in range(len(uni_cluster_gts)):
        #     mask = (cluster_gts == uni_cluster_gts[i])
        #     region_pred[mask] = region_labels[i]
        # cluster_gts = region_pred
        
        # for gt in range(db_label_set.shape[0]):
        #     mask_preds = region_pred[cluster_gts==db_label_set[gt]] # n,
        #     unique_val, counts = torch.unique(mask_preds, return_counts = True)
        #     if counts.shape[0]==0:
        #         continue
        #     pred_label = unique_val[counts.argmax()]
        #     # pred_label = cluster_preds[gt]
        #     region_pred[cluster_gts==db_label_set[gt]] = pred_label

        # if with_bg:
        #     region_pred[region_pred>=C] = C
        #     gts[gts==255] = C
        # cal_pred.append(region_pred.cpu().numpy())
        # cal_gt.append(gts.cpu().numpy())        
        ################

        ################
        # rag, superpixel_labels = build_color_rag(np.array(ori_images), n_segments=1000, compactness=10)
        # # adj_ = nx.to_numpy_array(rag)
        # # adj_ = adj_ / adj_.max()
        # L = nx.normalized_laplacian_matrix(rag).toarray()
        # eigvals, eigvecs = np.linalg.eigh(L)
        # labels_l = dbscan.fit_predict(eigvecs)

        # k = 5  # 设定类别数
        # # X = eigvecs[:, 1:k+1]  # 忽略第一个特征向量 (常为全1向量)
        # X = eigvecs[:, 1:k+1]

        # # 5. 使用 k-means 聚类
        # kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        # labels_xxx = kmeans.fit_predict(X)

        # num_classes = labels_xxx.max() + 1
        # color_map = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)  # 生成 RGB 颜色

        # # 创建彩色图像 (224, 224, 3)
        # visualization = np.zeros((224, 224, 3), dtype=np.uint8)

        # # 遍历所有像素，将 superpixel 映射到类别，再映射到颜色
        # for sp in range(808):
        #     mask = (superpixel_labels == sp)  # 找到 superpixel 的像素位置
        #     class_id = labels_xxx[sp]  # 获取该 superpixel 对应的类别
        #     visualization[mask] = color_map[class_id]  # 赋予类别颜色

        # # 可视化结果
        # plt.figure(figsize=(6, 6))
        # plt.imshow(visualization)
        # plt.axis("off")
        # plt.title("Superpixel Classification Visualization")
        # plt.show()        

        # rag_cluster_gts = graph.rag_mean_color(cluster_gts.detach().cpu().numpy(), superpixel_labels)
        # graph.show_rag(superpixel_labels, rag_cluster_gts, np.stack([cluster_gts.detach().cpu().numpy() * 25] * 3, axis=-1), img_cmap=None)
        ################

        # 好像vote之前结果还不错，vote之后结果就很不行了
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

        # visualisation
        ###############
        # save_preds.append(patch_preds.detach().cpu().numpy())
        # save_gts.append(gts.detach().cpu().numpy())
        # if pi ==6:
        # num_classes = C
        # cmap = plt.cm.get_cmap('jet', num_classes)
        # plt.figure(figsize=(16,6))
        # plt.subplot(1,3,1)
        # plt.imshow(images.permute(1,2,0).detach().cpu().numpy())
        # plt.subplot(1,3,2)
        # plt.imshow(region_pred.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=num_classes-1)
        # plt.subplot(1,3,3)
        # plt.imshow(gts.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=num_classes-1)   
        # plt.tight_layout()
        # plt.savefig(f"/home/qiming/Desktop/code/iccv25/v7/CLIPtrase_/vis_/{pi}.png") 
        # plt.close()
        ###############

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
    # datasets = ["VOC20","VOC21","COCO80_val","COCO171_val","PC59","PC60","PC459","ADE150","ADEfull"]
    # datasets = ["VOC20","ADE150","ADEfull","COCO171_val","PC59", "PC459"]
    datasets = ["COCO171_val"]
    for d in datasets:
        self_clip(clip_model, d, image_size=224,eps=0.7,min=3)
        # self_clip(clip_model, d, image_size=336,eps=1.1,min=7)

if __name__=="__main__":
    with torch.no_grad():
        self_clip_test()
