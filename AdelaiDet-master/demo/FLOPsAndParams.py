# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    all_time_list = []
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
    print(args.input)
    img = read_image(args.input, format="BGR")
    original_image = img[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}

    # flops, params = get_model_complexity_info(demo.predictor.model, [inputs], as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params = profile(demo.predictor.model, ([inputs],))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    # tensor = (torch.rand(1, 3, 224, 224),)
    # flops = FlopCountAnalysis(demo.predictor.model, [inputs])
    # print("FLOPs: ", flops.total())
    # print(parameter_count_table(demo.predictor.model))



#python D:\Software\pycharm\test\BlendMaskAndYolov7\AdelaiDet-master\demo\demo.py --config-file D:\Software\pycharm\test\BlendMaskAndYolov7\AdelaiDet-master\configs\BlendMask\Base-BlendMask.yaml --input D:\Software\pycharm\datasets\Instance\coco\test2017\PNGtype\0021.png --confidence-threshold 0.35 --opts MODEL.WEIGHTS D:\Software\pycharm\test\AdelaiDet-master\training_dir\blendmask_R_50_1x\model_final.pth
#python D:\Software\pycharm\test\BlendMaskAndYolov7\AdelaiDet-master\demo\demo.py --config-file D:\Software\pycharm\test\BlendMaskAndYolov7\AdelaiDet-master\configs\BlendMask\Base-BlendMask.yaml --input D:\Software\pycharm\datasets\Instance\coco\test2017\PNGtype --output C:\Users\asus\Desktop\data\BlendShow  --confidence-threshold 0.35 --opts MODEL.WEIGHTS D:\Software\pycharm\test\BlendMaskAndYolov7\AdelaiDet-master\training_dir\blendmask_R_50_1x_fifth\model_final.pth