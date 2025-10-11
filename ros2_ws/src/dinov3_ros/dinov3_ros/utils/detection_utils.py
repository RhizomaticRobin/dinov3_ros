# deps: rclpy, vision_msgs, std_msgs
import torch

from vision_msgs.msg import (
    Detection2DArray, Detection2D, BoundingBox2D, 
    ObjectHypothesisWithPose, Pose2D, Point2D
)

from dinov3_toolkit.head_detection.utils import decode_outputs

def outputs_to_detection2darray(boxes: torch.Tensor,
                        scores: torch.Tensor,
                        labels: torch.Tensor,
                        header) -> Detection2DArray:
    """
    boxes:  Nx4 tensor [x1, y1, x2, y2] in pixels
    scores: N   tensor in [0,1]
    labels: N   tensor (int class ids)
    header: std_msgs.msg.Header from the source image
    """
    # Move to CPU + plain Python types
    if boxes.is_cuda:  boxes = boxes.detach().cpu()
    if scores.is_cuda: scores = scores.detach().cpu()
    if labels.is_cuda: labels = labels.detach().cpu()

    boxes  = boxes.float()
    scores = scores.float()
    labels = labels.long()

    arr = Detection2DArray()
    arr.header = header

    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        det = Detection2D()
        det.header = header  # keep per-detection header consistent

        # Convert to center-size (axis-aligned; theta=0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w  = max(0.0, x2 - x1)
        h  = max(0.0, y2 - y1)

        bb = BoundingBox2D()
        bb.center = Pose2D()
        bb.center.position = Point2D()
        bb.center.position.x = float(cx)
        bb.center.position.y = float(cy)
        bb.center.theta = 0.0
        bb.size_x = float(w)
        bb.size_y = float(h)
        det.bbox = bb

        ohp = ObjectHypothesisWithPose()
        ohp.hypothesis.class_id = str(c)
        ohp.hypothesis.score = float(s)
        # ohp.pose left default (unused for 2D)

        det.results.append(ohp)
        arr.detections.append(det)

    return arr

def decode_outputs_tensorrt(outputs, img_size, score_thresh, nms_thresh):
    n_levels = int(len(outputs)/3)
    model_outputs = {}
    model_outputs['cls'] = outputs[0:n_levels]
    model_outputs['reg'] = outputs[n_levels:2*n_levels]
    model_outputs['ctr'] = outputs[2*n_levels:3*n_levels]

    # Compute the stride (scaling factor) of the first output level
    first_stride = img_size / model_outputs['cls'][0].shape[2]

    # Collect strides for each feature level (used for decoding predictions)
    strides = [first_stride]
    for l in range(1, len(model_outputs['cls'])):
        strides.append(first_stride * 2**l)

    # Decode raw model outputs into final predictions
    # Applies thresholding and non-maximum suppression
    boxes, scores, labels = decode_outputs(
        model_outputs,
        (img_size, img_size),   # Original image size
        strides,
        score_thresh=score_thresh, 
        nms_thresh=nms_thresh
    )

    return boxes, scores, labels