"""Base constructs to use torchvision models.
"""
import fiftyone as fo
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms import functional as T
from tqdm import tqdm

from finegrained.models.torch_utils import get_device
from finegrained.utils.dataset import load_fiftyone_dataset


def _parse_bbox(box, image_size):
    h, w = image_size
    if h > w:
        box = box.T
    xmin, ymin, xmax, ymax = box.cpu() / torch.tensor([w, h, w, h])
    box_w = xmax - xmin
    box_h = ymax - ymin
    return [xmin.item(), ymin.item(), box_w.item(), box_h.item()]


def _parse_torchvision_detections(prediction, class_labels, image_size):
    detections = []
    for box, label, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        box = _parse_bbox(box, image_size)
        det = fo.Detection(
            label=class_labels[label.item()],
            bounding_box=box,
            condfidence=score.item(),
        )
        detections.append(det)
    return fo.Detections(detections=detections)


def _resize_image(img: torch.Tensor, target_size: int) -> torch.Tensor:
    orig_h, orig_w = img.shape[1:]
    max_dim = max(orig_h, orig_w)
    resize_h = int(orig_h * (target_size / max_dim))
    resize_w = int(orig_w * (target_size / max_dim))
    img = T.resize(img, size=(resize_h, resize_w))
    return img


def object_detection(
    dataset: str,
    label_field: str,
    conf: float = 0.25,
    image_size=None,
    device=None,
    **kwargs
):
    """Detect COCO objects with mask-rcnn-v2 from torchvision

    Args:
        dataset: fiftyone dataset name
        label_field: which field to write predictions to
        conf: box confidence threshold
        image_size: if specified, this will be a max image size
            (to save memory)
        **kwargs: dataset loading filters

    Returns:
        None
    """
    # prepare the model
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=conf)
    device = get_device()[0] if device is None else torch.device(device)
    model.eval().to(device)
    preprocess = weights.transforms()

    dataset = load_fiftyone_dataset(dataset, **kwargs)

    with torch.no_grad():
        for smp in tqdm(dataset.select_fields("filepath"), desc="detecting"):
            # prepare image input
            img = read_image(smp.filepath)
            if image_size:
                img = _resize_image(img, target_size=image_size)
            # predict
            batch = [preprocess(img).to(device)]
            prediction, *_ = model(batch)
            # parse detection and save results
            img_h, img_w = img.size(1), img.size(2)
            detections = _parse_torchvision_detections(
                prediction, weights.meta["categories"], (img_h, img_w)
            )
            smp[label_field] = detections
            smp.save()
