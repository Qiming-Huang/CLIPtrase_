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
import seaborn as sns
from sklearn.cluster import KMeans
from collections import Counter
from scipy.stats import mode

from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from torchvision.utils import save_image
import scipy
import ot

device = "cuda"

def generate_patch_label():
    mask_size = (224, 224)
    patch_size = (16, 16)
    num_patches = (mask_size[0] // patch_size[0], mask_size[1] // patch_size[1])

    # 生成标签矩阵
    patch_mask = np.zeros(mask_size, dtype=np.int32)

    # 为每个patch分配标签
    patch_idx = 0
    for i in range(num_patches[0]):
        for j in range(num_patches[1]):
            patch_mask[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]] = patch_idx
            patch_idx += 1 

    return patch_mask    

def majority_filter(matrix, node_mask, window_size=3):
    """
    对给定矩阵应用滑动窗口滤波，
    替换中心像素为窗口内 matrix 中对应于 node_mask 中最接近中心值的位置的值。
    
    :param matrix: 输入的矩阵 (numpy array)
    :param node_mask: 具有相同大小的掩码矩阵 (numpy array)
    :param window_size: 滑动窗口的大小 (必须是奇数, 默认3)
    :return: 处理后的矩阵
    """
    if window_size % 2 == 0:
        raise ValueError("窗口大小必须是奇数！")
    
    matrix = np.array(matrix)
    node_mask = np.array(node_mask)
    rows, cols = matrix.shape
    output = matrix.copy()  # 复制矩阵
    pad = window_size // 2  # 计算填充的大小
    
    # 遍历每个像素点（忽略边界）
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            # 提取 node_mask 窗口
            mask_window = node_mask[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
            matrix_window = matrix[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
            center_value = node_mask[i, j]
            
            # 计算与中心值的绝对差值，并找到最接近的索引
            closest_idx = np.argmin(np.abs(mask_window - center_value))
            closest_value = matrix_window[closest_idx]
            
            # 替换中心像素值为 matrix 中对应位置的值
            output[i, j] = closest_value
    
    return output

def sinkhorn(a, b, cost_matrix, epsilon=0.01, n_iter=50):
    K = torch.exp(-cost_matrix / epsilon)  # 计算 kernel 矩阵
    u = torch.ones_like(a) / len(a)  # 初始化 u
    v = torch.ones_like(b) / len(b)  # 初始化 v

    for _ in range(n_iter):
        u = a / (K @ v)  # 更新 u
        v = b / (K.T @ u)  # 更新 v

    transport_matrix = torch.diag(u) @ K @ torch.diag(v)  # 计算最终传输矩阵
    return transport_matrix

def normalize_to_sphere(features):
    return features / features.norm(dim=1, keepdim=True)

def cosine_similarity_matrix(A, B):
    return torch.matmul(A, B.T)  # 余弦相似度计算


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

def slerp_torch(p1, p2, alpha):
    """
    在球面上进行插值 (Spherical Linear Interpolation, SLERP) - PyTorch 版本
    :param p1: 起始点向量 (torch tensor, shape=[d])
    :param p2: 目标点向量 (torch tensor, shape=[d])
    :param alpha: 插值因子，0 <= alpha <= 1 (float or tensor)
    :return: 插值后的向量 (torch tensor)
    """
    # 归一化向量，确保它们在单位球面上
    p1 = p1 / torch.linalg.norm(p1, dim=-1, keepdim=True)
    p2 = p2 / torch.linalg.norm(p2, dim=-1, keepdim=True)

    # 计算夹角 θ
    dot_product = torch.clamp(torch.dot(p1, p2), -1.0, 1.0)  # 避免数值误差
    theta = torch.acos(dot_product)

    # 计算权重
    sin_theta = torch.sin(theta)
    if sin_theta < 1e-6:  # 避免除零错误，如果夹角太小，则用 LERP 近似
        return (1 - alpha) * p1 + alpha * p2

    # SLERP 计算
    p_interp = (torch.sin((1 - alpha) * theta) / sin_theta) * p1 + (torch.sin(alpha * theta) / sin_theta) * p2
    return p_interp

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

    save_preds = []
    save_gts = []

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

        #######################        
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

        # attention refinement, patch
        #######################
        superpixel_labels = generate_patch_label()
        rag = graph.rag_mean_color(np.array(ori_images), superpixel_labels)

        edge_weights = [rag[u][v]['weight'] for u, v in rag.edges]

        if edge_weights:  # 确保有边
            min_w, max_w = np.min(edge_weights), np.max(edge_weights)

            # 避免除零错误
            if max_w > min_w:
                normalized_weights = [(w - min_w) / (max_w - min_w) for w in edge_weights]
            else:
                normalized_weights = [0] * len(edge_weights)  # 所有权重相等，归一化后为 0

            # 赋值回 `rag`
            for (u, v), w_ in zip(rag.edges, normalized_weights):
                rag[u][v]['weight'] = w_
        
        ############################ TODO
        # scaler = MinMaxScaler()
        # A_ = nx.to_numpy_array(rag)
        # A_n = scaler.fit_transform(A_)
        # attn_weights += torch.from_numpy(attn_weights.detach().cpu().numpy() * A_n).to("cuda")

        labels = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(attn_weights.detach().cpu().numpy()).labels_
        labels = torch.from_numpy(labels)


        patch_logits = patch_tokens@text_features.T * clip.logit_scale.exp() # 196, C
        patch_logits = patch_logits.permute(1,0).unsqueeze(0).reshape(1,text_features.shape[0],h,w)

        patch_preds_topk = torch.topk(patch_logits.squeeze(), k=5, dim=0).indices.reshape((5, 196))

        patch_logits = F.interpolate(
            patch_logits,size=(ori_h, ori_w),mode='bilinear',align_corners=False,
        )[0] # C,H,W
        if dataset=='VOC20':
            cls_logits = cls_token@text_features.T * clip.logit_scale.exp() # 1,c
            patch_logits = patch_logits*cls_logits[0].unsqueeze(1).unsqueeze(1)

        patch_preds = patch_logits.argmax(dim=0) # H,W

        # 修改特征表达在球面上的位置
        ##########################
        patch_preds_ = F.interpolate(patch_preds.unsqueeze(0).unsqueeze(0).float(), (14, 14)).squeeze().long().reshape((-1, 1))
        for node in rag.nodes:
            rag.nodes[node]['cluster_label'] = int(patch_preds_[node])
        
        for node in rag.nodes:
            neighbor_weights = {neighbor: 1 - rag[node][neighbor]['weight'] for neighbor in rag.adj[node]}
            cluster_labels = {node: rag.nodes[node]['cluster_label'] for node in neighbor_weights.keys()}

            # node_cluster_label = rag.nodes[node]['cluster_label']

            # 检查是否所有值都相同
            if len(set(cluster_labels.values())) > 1:
                for key, val in cluster_labels.items():
                    if rag.nodes[node]['cluster_label'] != val:
                        # patch_tokens[node] = slerp_torch(patch_tokens[node], text_features[val], neighbor_weights[key])
                        patch_tokens[node] = slerp_torch(patch_tokens[node], text_features[patch_preds_topk[:, node], :].mean(dim=0), neighbor_weights[key])
            else:
                pass
        ##########################


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

        # patch region matching
        #######################
        # cluster_preds = cluster_gts.argmax(dim=0).reshape((-1, 1))[:,0]

        # region_preds = torch.zeros((196), dtype=cluster_preds.dtype).to("cuda")

        # region_feats = []
        # region_labels = []
        # for c_p in torch.unique(cluster_preds):
        #     mask = (cluster_preds == c_p)
        #     region_feat = patch_tokens[mask]
        #     region_feats.append(region_feat)

        #     region_matched = (region_feat@text_features.T * clip.logit_scale.exp()).argmax(dim=-1)
        #     unique_val, counts = torch.unique(region_matched, return_counts = True)
        #     max_label = unique_val[counts.argmax()]
        #     region_labels.append(max_label)

        #     region_preds[mask] = max_label

        # region_feats[0]
        # patch_logits = region_feats@text_features.T * clip.logit_scale.exp() # 196, C
        #######################

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

        # visualisation
        ###############
        save_preds.append(patch_preds.detach().cpu().numpy())
        save_gts.append(gts.detach().cpu().numpy())
        num_classes = C
        cmap = plt.cm.get_cmap('jet', num_classes)
        plt.figure(figsize=(16,6))
        plt.subplot(1,3,1)
        plt.imshow(images.permute(1,2,0).detach().cpu().numpy())
        plt.subplot(1,3,2)
        plt.imshow(patch_preds.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=num_classes-1)
        plt.subplot(1,3,3)
        plt.imshow(gts.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=num_classes-1)   
        plt.tight_layout()
        plt.savefig(f"/home/qiming/Desktop/code/iccv25/v7/CLIPtrase_/vis_all/vis_3/{pi}.png") 
        plt.close()
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
    # datasets = ["VOC21", "COCO80_val", "PC60"]
    datasets = ["COCO171_val"]
    for d in datasets:
        self_clip(clip_model, d, image_size=224,eps=0.7,min=3)
        # self_clip(clip_model, d, image_size=336,eps=1.1,min=7)

if __name__=="__main__":
    with torch.no_grad():
        self_clip_test()
