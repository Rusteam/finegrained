"""YOLO utils for training, evaluating and exporting.
"""
import shutil
from typing import List, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import nn
from torchvision.ops import batched_nms

from finegrained.utils.triton import TritonExporter

DEFAULTS = dict(iou_threshold=0.4, conf_threshold=0.25, stride=32, max_detections=100)


def _get_dim(t: torch.Tensor) -> torch.Tensor:
    """get a size without casting to python"""
    bools = t.greater_equal(0)
    return bools.sum(0, keepdim=True)


def get_tensor_size(x: torch.Tensor) -> torch.Tensor:
    """return (h,w) of a tensor as a tensor"""
    # return torch.tensor([x.size(1), x.size(2)])
    h = _get_dim(x[0, :, 0])
    w = _get_dim(x[0, 0, :])
    return torch.cat([h, w])


def apply_conf_threshold(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    label_probs: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter out low-confidence results.


    :param scores: scores to apply the threshold to
    :param threshold: a threshold value
    :return: a list of filtered tensors
    """
    above_conf = scores > threshold
    boxes = boxes[above_conf]
    scores = scores[above_conf]
    labels = labels[above_conf]
    label_probs = label_probs[above_conf]
    return boxes, scores, labels, label_probs


def scale_coords(
    img1_shape: torch.Tensor, coords: torch.Tensor, img0_shape: torch.Tensor
) -> torch.Tensor:
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    gain = (img1_shape / img0_shape).min()
    pad = (img1_shape - img0_shape * gain) / 2
    coords[:, [0, 2]] -= pad[1]  # x padding
    coords[:, [1, 3]] -= pad[0]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes: torch.Tensor, shape: torch.Tensor) -> None:
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    boxes[:, 0].clamp_(0, shape[1])  # x1
    boxes[:, 1].clamp_(0, shape[0])  # y1
    boxes[:, 2].clamp_(0, shape[1])  # x2
    boxes[:, 3].clamp_(0, shape[0])  # y2


def xyxy2xywhn(x: torch.Tensor, w: int = 640, h: int = 640) -> torch.Tensor:
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    normalized where xy1=top-left, xy2=bottom-right"""
    y = x.clone()
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def rescale_boxes(
    boxes: torch.Tensor, input_size: torch.Tensor, img_size: torch.Tensor
) -> torch.Tensor:
    """Scale detected boxes to orig image size
    and normalize to 0-1 range in the xywh format.
    """
    boxes = scale_coords(img1_shape=input_size, coords=boxes, img0_shape=img_size)
    h, w = img_size[0], img_size[1]
    boxes = xyxy2xywhn(boxes, w=w, h=h)
    return boxes


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    where xy1=top-left, xy2=bottom-right"""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def parse_yolov5_prediction(
    prediction: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse raw predictions into boxes, scores, label probs and indices."""
    xywh = prediction[:, :4]
    obj_scores = prediction[:, 4]
    label_probs = prediction[:, 5:]
    clf_scores, labels = label_probs.max(dim=1)
    boxes = xywh2xyxy(xywh)
    scores = obj_scores * clf_scores
    return boxes, scores, labels, label_probs


class Letterbox(nn.Module):
    """Letterbox transformation for object detection.

    Resize larger size to new_shape if size is bigger than new_shape
    or pad to the closest divisible of stride if less than new_shape.
    Resize smaller size with the same ratio and
    pad it to the closest divisible by stride.

    :param new_shape: new max size, will result in (<=new_shape, <=new_shape)
    :param stride: both dimensions will be divisible by this number
    """

    def __init__(self, new_shape: int, stride: int):
        super(Letterbox, self).__init__()
        self.new_shape = nn.Parameter(torch.tensor(new_shape), requires_grad=False)
        self.stride = nn.Parameter(torch.tensor(stride), requires_grad=False)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        shape: torch.Tensor = get_tensor_size(sample)
        scale_ratio: torch.Tensor = (self.new_shape / shape.max()).clip(0.0, 1.0)
        new_unpad: torch.Tensor = (shape * scale_ratio).round().to(torch.long)  # h,w
        delta_hw: torch.Tensor = (
            torch.fmod(self.new_shape - new_unpad, self.stride) / 2
        ).to(torch.long)
        new_unpad_l: List[torch.Tensor] = [new_unpad[0], new_unpad[1]]
        sample = F.resize(
            sample, new_unpad_l, interpolation=F.InterpolationMode.BILINEAR
        )
        padding: List[torch.Tensor] = [
            delta_hw[1],
            delta_hw[0],
            delta_hw[1],
            delta_hw[0],
        ]
        sample = F.pad(sample, padding, fill=sample.to(torch.float).mean().item())
        return sample


class YOLOv5Preprocessing(nn.Module, TritonExporter):
    """Pre-process yolov5 inputs.

    Run a raw input tensor image thru letterbox transformation
    and return a resized and normalized image along with resize ratio
    """

    def __init__(self, image_size, stride=DEFAULTS["stride"]):
        super(YOLOv5Preprocessing, self).__init__()
        self.transforms = nn.Sequential(Letterbox(new_shape=image_size, stride=stride))
        self.reshape = T.Lambda(lambda x: x.permute(2, 0, 1))

    def forward_one(
        self, sample: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.reshape(sample)
        img_size = get_tensor_size(sample)
        sample = self.transforms(sample)
        sample = (sample / 255.0).to(torch.float32)
        return sample, img_size, get_tensor_size(sample)

    def forward(
        self, sample: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, img_size, input_size = self.forward_one(sample)
        return out.unsqueeze(0), img_size, input_size

    def _load_model_torch(self, *args, **kwargs) -> torch.nn.Module:
        return self

    def generate_dummy_inputs(self, *args, **kwargs) -> List[torch.Tensor]:
        return (torch.randint(0, 255, size=(600, 680, 3), dtype=torch.uint8),)

    @property
    def output_names(self):
        return ["output", "image_size", "input_size"]

    @property
    def triton_batch_size(self):
        return 0

    @property
    def dynamic_axes(self):
        return {
            "image": {0: "h", 1: "w"},
            "output": {2: "h", 3: "w"},
            # "image_size": {0: "batch"},
            # "input_size": {0: "batch"},
        }


class YOLOv5Postprocessing(nn.Module, TritonExporter):
    """Post-process detection outputs."""

    def __init__(
        self,
    ):
        super(YOLOv5Postprocessing, self).__init__()
        self._ = nn.Sequential()
        self.conf_threshold: float = DEFAULTS["conf_threshold"]
        self.iou_threshold: float = DEFAULTS["iou_threshold"]
        self.max_detections: float = DEFAULTS["max_detections"]

    def _filter_detections(self, detections: torch.Tensor):
        """Parse detections, apply conf. threshold and run NMS."""
        boxes, scores, labels, label_probs = parse_yolov5_prediction(detections)
        boxes, scores, labels, label_probs = apply_conf_threshold(
            boxes=boxes,
            scores=scores,
            labels=labels,
            label_probs=label_probs,
            threshold=self.conf_threshold,
        )
        keep_inds: torch.Tensor = batched_nms(
            boxes, scores, labels, self.iou_threshold
        )[: self.max_detections]
        return boxes[keep_inds], scores[keep_inds].unsqueeze(1), label_probs[keep_inds]

    def _prepare_output(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        label_probs: torch.Tensor,
        input_size: torch.Tensor,
        img_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """rescale bboxes to original image size and one-hot encode labels"""
        rescaled_boxes = rescale_boxes(boxes, input_size=input_size, img_size=img_size)
        return rescaled_boxes, scores, label_probs

    def forward(
        self,
        prediction: torch.Tensor,
        image_size: torch.Tensor,
        input_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        detections = prediction.squeeze(0)
        # image_size = image_size[0]
        # input_size = input_size[0]

        boxes, scores, label_probs = self._filter_detections(detections)
        out = self._prepare_output(boxes, scores, label_probs, input_size, image_size)
        return out

    def _load_model_torch(self, *args, **kwargs) -> torch.nn.Module:
        return self

    @property
    def input_names(self):
        return ["prediction", "image_size", "input_size"]

    @property
    def output_names(self):
        return ["boxes__0", "scores__1", "class_probs__2"]

    @property
    def dynamic_axes(self):
        return {
            self.input_names[0]: {1: "n_detections", 2: "n_classes"},
            # self.input_names[1]: {0: "batch"},
            # self.input_names[2]: {0: "batch"},
            self.output_names[0]: {0: "n_detections"},
            self.output_names[1]: {0: "n_detections"},
            self.output_names[2]: {0: "n_detections", 2: "n_classes"},
        }

    @property
    def triton_batch_size(self):
        return 0

    def generate_dummy_inputs(self, **kwargs) -> List[torch.Tensor]:
        n = 18900
        prediction = torch.rand(1, n, 10)
        img_size = torch.tensor([480, 640])
        inp_size = torch.tensor([240, 320])
        return prediction, img_size, inp_size

    def _create_triton_config(self, torchscript: bool = False, **kwargs) -> dict:
        return dict(
            backend="pytorch" if torchscript else "onnxruntime",
            max_batch_size=self.triton_batch_size,
            input=[
                dict(
                    name=self.input_names[0],
                    data_type="TYPE_FP32",
                    dims=[1, -1, -1],
                ),
                dict(name=self.input_names[1], data_type="TYPE_INT64", dims=[2]),
                dict(name=self.input_names[2], data_type="TYPE_INT64", dims=[2]),
            ],
            output=[
                dict(
                    name=self.output_names[0],
                    data_type="TYPE_FP32",
                    dims=[-1, -1],
                ),
                dict(name=self.output_names[1], data_type="TYPE_FP32", dims=[-1, 1]),
                dict(
                    name=self.output_names[2],
                    data_type="TYPE_FP32",
                    dims=[-1, -1],
                ),
            ],
        )


class YOLOv5Model(TritonExporter):
    """Create a triton model and ensemble for a trained yolov5 model."""

    @property
    def triton_batch_size(self):
        return 0

    def export_onnx(self, model_path: str, write_path: str, **kwargs):
        """Copy existing ONNX file to write_path."""
        shutil.copy(model_path, write_path)

    def _create_triton_ensemble_config(
        self,
        preprocessing_name: str,
        model_name: str,
        postprocessing_name: str,
        **kwargs
    ):
        return dict(
            platform="ensemble",
            max_batch_size=self.triton_batch_size,
            input=[dict(name="IMAGE", data_type="TYPE_UINT8", dims=[-1, -1, 3])],
            output=[
                dict(name="BOXES", data_type="TYPE_FP32", dims=[-1, -1]),
                dict(name="SCORES", data_type="TYPE_FP32", dims=[-1, 1]),
                dict(name="CLASS_PROBS", data_type="TYPE_FP32", dims=[-1, -1]),
            ],
            ensemble_scheduling=dict(
                step=[
                    dict(
                        model_name=preprocessing_name,
                        model_version=-1,
                        input_map=dict(image="IMAGE"),
                        output_map=dict(
                            output="PREPROCESSED",
                            image_size="IMAGE_SIZE",
                            input_size="INPUT_SIZE",
                        ),
                    ),
                    dict(
                        model_name=model_name,
                        model_version=-1,
                        input_map=dict(images="PREPROCESSED"),
                        output_map=dict(output="DETECTIONS"),
                    ),
                    dict(
                        model_name=postprocessing_name,
                        model_version=-1,
                        input_map=dict(
                            prediction="DETECTIONS",
                            image_size="IMAGE_SIZE",
                            input_size="INPUT_SIZE",
                        ),
                        output_map=dict(
                            boxes__0="BOXES",
                            scores__1="SCORES",
                            class_probs__2="CLASS_PROBS",
                        ),
                    ),
                ]
            ),
        )
