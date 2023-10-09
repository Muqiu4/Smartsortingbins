# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()#获取当前目录(detect.py)的(使用relsove)绝对路径,并将其赋值给变量FILE F:\yolov5-7.0\detect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory 获取上一级目录 F:\yolov5-7.0
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)# 输入图像的路径、目录、URL、glob、screen或0（webcam）转换为字符串型
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)#判断source是否是文件路径，即判断其后缀名是否在IMG_FORMATS和VID_FORMATS中。
    # IMG_FORMATS和VID_FORMATS是预定义的图像和视频格式列表。
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))# 判断source是否是URL，即判断其是否以rtsp://、rtmp://、http://或https://开头。
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)#判断source是否是摄像头，即判断其是否是数字、以.txt结尾或者是URL但不是文件
    screenshot = source.lower().startswith('screen')#判断source是否是屏幕截图，即判断其是否以screen开头。
    if is_url and is_file:
        source = check_file(source)  # download如果source是URL（网站）并且不是文件，则调用check_file()函数下载文件。

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run 将project和name拼接成一个路径，并调用increment_path()函数对路径进行增量命名，以避免覆盖之前的结果
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir创建保存推理结果的目录。如果save_txt为True，则在save_dir路径下创建一个名为labels的子目录，用于保存标签文件

    # Load model
    device = select_device(device)# 根据device参数选择设备，如果device为空，则自动选择最快的可用设备。
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)#加载模型，weights参数是模型权重文件的路径，
    # device参数是设备，dnn参数表示是否使用OpenCV DNN进行ONNX推理，data参数是数据集的yaml文件路径，fp16参数表示是否使用FP16半精度推理。
    stride, names, pt = model.stride, model.names, model.pt# 获取模型的步幅、类别名称和权重文件路径。
    imgsz = check_img_size(imgsz, s=stride)  # check image size检查输入图像的大小是否符合要求。
    # check_img_size()函数会将输入的图像大小调整为模型的步幅的倍数，并返回调整后的图像大小。

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)#使用摄像头，则设置view_img为True，表示需要显示视频流的画面
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        #加载视频流，并将其存储在dataset变量中。LoadStreams是一个自定义的函数，用于加载视频流。
        # source表示视频流的来源，img_size表示图像的大小，stride表示图像的步长，auto表示是否使用自动调整大小的模式，vid_stride表示视频流的步长。
        bs = len(dataset)#计算视频流中的帧数，并将其存储在bs变量中。len(dataset)返回视频流中的帧数。
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup,推理预热
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)#将 NumPy 数组转换为 PyTorch 张量，并将其移动到指定的设备上。
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32如果模型使用了混合精度训练（fp16），则将张量类型转换为半精度浮点数（half），否则转换为单精度浮点数（float）。
            im /= 255  # 0 - 255 to 0.0 - 1.0  归一化
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim如果张量的维度为 3，则在第 0 维上添加一个维度，以便将其扩展为批次大小为 1 的张量。

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False#根据 visualize 参数的值，决定是否将可视化结果保存到本地。这里是False
            pred = model(im, augment=augment, visualize=visualize)#调用 YOLOv5 模型的 model 方法，对输入的图像或视频进行推理，并得到目标检测结果。

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)#对模型输出的目标检测结果进行非极大值抑制（NMS）处理，去除重复的检测框
            """
            pred：模型输出的目标检测结果，是一个列表，每个元素表示一张图像的检测结果。
            
            conf_thres：置信度阈值，小于该值的检测结果将被过滤掉。
            
            iou_thres：IoU 阈值，大于该值的检测框将被过滤掉。
            
            classes：一个整数列表，表示需要保留的类别。
            
            agnostic_nms：一个布尔值，表示是否对不同类别的检测结果进行独立的 NMS 处理。
            
            max_det：一个整数，表示每个类别的最大检测框数量。
            """
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt，保存为结果txt
            s += '%gx%g ' % im.shape[2:]  # print string用做打印结果
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 获得原图的宽高大小
            imc = im0.copy() if save_crop else im0  # for save_crop,是否需要将框内东西裁剪下来保存
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))#画框init处理
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()#将坐标映射到原图上画框

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')#是否需要画框
                        annotator.box_label(xyxy, label, color=colors(c, True))#正式画框和标签
                    if save_crop:#是否需要把框内东西保存为图片
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()#画好的图片
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default="E:\yolov5-7.0\\runs\\train\exp\weights\\best.pt", help='model path or triton URL')
    parser.add_argument('--source', type=str, default="0", help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand [640,640]
    print_args(vars(opt))#打印以上所有参数信息
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))#检查运行所需的软件包及依赖库安装完成
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
