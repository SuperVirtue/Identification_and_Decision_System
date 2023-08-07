# coding: utf-8
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torchvision.transforms as transforms
import torchvision.models as models

# 训练过的模型路径
from torch import nn

resume_path = r"D:\TJU\GBDB\set113\cross_validation\test1\epoch_0257_checkpoint.pth.tar"
# 输入图像路径
single_img_path = r'D:\TJU\GBDB\set113\CAM\temp.jpg'
# 绘制的热力图存储路径
save_path = r'D:\TJU\GBDB\set113\CAM\temp_layer4.jpg'

# 网络层的层名列表, 需要根据实际使用网络进行修改
layers_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
# 指定层名
out_layer_name = "layer4"

features_grad = 0


# 为了读取模型中间参数变量的梯度而定义的辅助函数
def extract(g):
    global features_grad
    features_grad = g


def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False, out_layer=None):
    """
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    """
    # 读取图像并预处理
    global layer2
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img).cuda()
    img = img.unsqueeze(0)  # (1, 3, 448, 448)

    # model转为eval模式
    model.eval()

    # 获取模型层的字典
    layers_dict = {layers_names[i]: None for i in range(len(layers_names))}
    for i, (name, module) in enumerate(model.features._modules.items()):
        layers_dict[layers_names[i]] = module

    # 遍历模型的每一层, 获得指定层的输出特征图
    # features: 指定层输出的特征图, features_flatten: 为继续完成前端传播而设置的变量
    features = img
    start_flatten = False
    features_flatten = None
    for name, layer in layers_dict.items():
        if name != out_layer and start_flatten is False:  # 指定层之前
            features = layer(features)
        elif name == out_layer and start_flatten is False:  # 指定层
            features = layer(features)
            start_flatten = True
        else:  # 指定层之后
            if features_flatten is None:
                features_flatten = layer(features)
            else:
                features_flatten = layer(features_flatten)

    features_flatten = torch.flatten(features_flatten, 1)
    output = model.classifier(features_flatten)

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output, 1).item()
    pred_class = output[:, pred]

    # 求中间变量features的梯度
    # 方法1
    # features.register_hook(extract)
    # pred_class.backward()
    # 方法2
    features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]

    grads = features_grad  # 获取梯度
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    print("pooled_grads:", pooled_grads.shape)
    print("features:", features.shape)
    # features.shape[0]是指定层feature的通道数
    for i in range(features.shape[0]):
        features[i, ...] *= pooled_grads[i, ...]

    # 计算heatmap
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘


# if __name__ == '__main__':
#     transform = transforms.Compose([
#         transforms.Resize(448),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#     # 构建模型并加载预训练参数
#     seresnet50 = FineTuneSEResnet50(num_class=113).cuda()
#     checkpoint = torch.load(resume_path)
#     seresnet50.load_state_dict(checkpoint['state_dict'])
#     draw_CAM(seresnet50, single_img_path, save_path, transform=transform, visual_heatmap=True, out_layer=out_layer_name)
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from ptflops import get_model_complexity_info
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from thop import profile
from predictor import VisualizationDemo
from adet.config import get_cfg
import torch
import torchvision
import detectron2.data.transforms as T
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="D:/Software/pycharm/test/BlendMaskAndYolov7/AdelaiDet-master/configs/BlendMask/Base-BlendMask.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", default="D:/Software/pycharm/datasets/Instance/coco/test2017/PNGtype/0021.png", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "D:/Software/pycharm/test/BlendMaskAndYolov7/AdelaiDet-master/training_dir/blendmask_R_50_1x_saventh/model_final.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser

def featuremap_visual(feature,
                      out_dir=None,  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=True,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      num_ch=-1,  # 显示特征图前几个通道，-1 or None 都显示
                      nrow=8,  # 每行显示多少个特征图通道
                      padding=10,  # 特征图之间间隔多少像素值
                      pad_value=1  # 特征图之间的间隔像素
                      ):
    import matplotlib.pylab as plt
    import torchvision
    import os
    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0]
    feature = feature.unsqueeze(1)

    if c > num_ch > 0:
        feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)

    plt.title(title)
    plt.imshow(images)
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
        plt.savefig(out_root)
    if show_feature:
        plt.show()


import cv2
import numpy as np


def imnormalize(img,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
                ):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return (img - mean) / std


if __name__ == '__main__':
    import matplotlib.pylab as plt
    img = cv2.imread('D:/Software/pycharm/datasets/maskrcnn/pic/0001.jpg')  # 读取图片

    img = imnormalize(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    img = img.permute(0, 3, 1, 2)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.to('cuda:0')
    model = models.mobilenet_v2(pretrained=True)
    model = model.cuda()
    out = model(img)
