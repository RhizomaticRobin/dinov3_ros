import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

import torch
import numpy as np

from pathlib import Path

from dinov3_toolkit.common import image_to_tensor

from dinov3_toolkit.head_detection.utils import generate_detection_overlay

from dinov3_toolkit.head_segmentation.utils import generate_segmentation_overlay, outputs_to_maps

from dinov3_toolkit.head_depth.utils import depth_to_colormap

from dinov3_toolkit.head_optical_flow.utils import flow_to_image

# Extra functions to convert data to msgs
from dinov3_ros.utils.detection_utils import outputs_to_detection2darray, decode_outputs_tensorrt

# TensorRT libs
import tensorrt as trt
from tensorrt_lib.tensorrt_python.tensorrt_engine import TensorRTEngine, Options

class Dinov3Node(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("dinov3_node")
        self.get_logger().info(f"[{self.get_name()}] Creating...")

        # General params
        self.declare_parameter("img_size", 800)
        self.declare_parameter("img_mean", [0.485, 0.456, 0.406])
        self.declare_parameter("img_std", [0.229, 0.224, 0.225])
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # DINO model params
        self.declare_parameter("dino_model.onnx_path", '')

        # Object detection model params
        self.declare_parameter("detection_model.onnx_path", '')
        self.declare_parameter("detection_model.classes_path", '')
        self.declare_parameter("detection_model.score_thresh", 0.2)
        self.declare_parameter("detection_model.nms_thresh", 0.2)

        # Segmentation model params
        self.declare_parameter("segmentation_model.onnx_path", '')
        self.declare_parameter("segmentation_model.classes_path", '')

        # Depth model params
        self.declare_parameter("depth_model.onnx_path", '')

        # Optical flow model params
        self.declare_parameter("optical_flow_model.onnx_path", '')

        # Params for TensorRT variables
        self.declare_parameter("tensorrt_params.precision", '')
        self.declare_parameter("tensorrt_params.device_index", 5)
        self.declare_parameter("tensorrt_params.dla_core", 6)

        # Modes
        self.declare_parameter("debug", False)
        self.declare_parameter("perform_detection", False)
        self.declare_parameter("perform_segmentation", False)
        self.declare_parameter("perform_depth", False)
        self.declare_parameter("perform_optical_flow", False)

        self.get_logger().info(f"[{self.get_name()}] Created...")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        try:
            # General params
            self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value
            self.img_mean = self.get_parameter("img_mean").get_parameter_value().double_array_value
            self.img_std = self.get_parameter("img_std").get_parameter_value().double_array_value
            self.image_reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

            # DINO model params
            self.dino_model = {
                'onnx_path': self.get_parameter('dino_model.onnx_path').get_parameter_value().string_value,
            }

            # Object detection params
            self.detection_model = {
                'onnx_path': self.get_parameter('detection_model.onnx_path').get_parameter_value().string_value,
                'classes_path': self.get_parameter('detection_model.classes_path').get_parameter_value().string_value,
                'score_thresh': self.get_parameter('detection_model.score_thresh').get_parameter_value().double_value,
                'nms_thresh': self.get_parameter('detection_model.nms_thresh').get_parameter_value().double_value,
            }

            # Semantic segmentation params
            self.segmentation_model = {
                'onnx_path': self.get_parameter('segmentation_model.onnx_path').get_parameter_value().string_value,
                'classes_path': self.get_parameter('segmentation_model.classes_path').get_parameter_value().string_value,
            }

            # Depth params
            self.depth_model = {
                'onnx_path': self.get_parameter('depth_model.onnx_path').get_parameter_value().string_value,
            }

            # Optical flow params
            self.optical_flow_model = {
                'onnx_path': self.get_parameter('optical_flow_model.onnx_path').get_parameter_value().string_value,
            }

            # TensorRT params
            self.tensorrt_params = {
                'precision': self.get_parameter('tensorrt_params.precision').get_parameter_value().string_value,
                'device_index': self.get_parameter('tensorrt_params.device_index').get_parameter_value().integer_value,
                'dla_core': self.get_parameter('tensorrt_params.dla_core').get_parameter_value().integer_value,
            }

            # Modes
            self.debug = self.get_parameter("debug").get_parameter_value().bool_value
            self.perform_detection = self.get_parameter("perform_detection").get_parameter_value().bool_value
            self.perform_segmentation = self.get_parameter("perform_segmentation").get_parameter_value().bool_value
            self.perform_depth = self.get_parameter("perform_depth").get_parameter_value().bool_value
            self.perform_optical_flow = self.get_parameter("perform_optical_flow").get_parameter_value().bool_value

            # Translate mean and std to 3D tensor
            self.img_mean = np.array(self.img_mean, dtype=np.float32)[:, None, None]
            self.img_std = np.array(self.img_std, dtype=np.float32)[:, None, None]

            # Image profile
            self.image_qos_profile = QoSProfile(
                reliability=self.image_reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )

            # Publishers
            self.pubs = []
            self.pubs_debug = []

            self.pub_detections = self.make_pub(Detection2DArray, "dinov3/detections", 10)
            self.pub_sem_seg = self.make_pub(Image, "dinov3/sem_seg", 10)
            self.pub_depth = self.make_pub(Image, "dinov3/depth", 10)
            self.pub_optical_flow = self.make_pub(Image, "dinov3/optical_flow", 10)

            if self.debug:
                self.pub_img_detections = self.make_pub(Image, "dinov3/debug/img_detections", 10, debug=True)
                self.pub_img_sem_seg = self.make_pub(Image, "dinov3/debug/img_sem_seg", 10, debug=True)
                self.pub_img_depth = self.make_pub(Image, "dinov3/debug/img_depth", 10, debug=True)
                self.pub_img_optical_flow = self.make_pub(Image, "dinov3/debug/img_optical_flow", 10, debug=True)

            # CV bridge
            self.cv_bridge = CvBridge()

            # Counter image
            self.img_counter = 0

            # Features from previous iteration
            self.feats_prev = None

     
        except Exception as e:
            self.get_logger().error(f"Configuration failed. Error: {e}")
            return TransitionCallbackReturn.FAILURE
            
        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")
        try:
            # Create subscription
            self.sub_image = self.create_subscription(Image, "topic_image", self.image_cb, self.image_qos_profile)

            # Activate publishers
            self.activate_pubs(state)

            # We create the options variable
            trt_options = Options()
            trt_options.out_file_path = str(Path(self.dino_model["onnx_path"]).parent)
            trt_options.precision = getattr(trt, self.tensorrt_params["precision"])
            trt_options.batch_size = 1
            trt_options.deviceIndex = self.tensorrt_params["device_index"]
            trt_options.dlaCore = self.tensorrt_params["dla_core"]

            # Create the engine, build and load the network
            self.engine_dino = TensorRTEngine(trt_options)
            self.engine_dino.build(self.dino_model["onnx_path"])
            self.engine_dino.loadNetwork()
        
            # Open detection model
            if self.perform_detection:
                with open(self.detection_model["classes_path"]) as f:
                    self.detection_class_names = [line.strip() for line in f]

                # Change option to use the depth path
                trt_options.out_file_path = str(Path(self.detection_model["onnx_path"]).parent)
                # Create the engine, build and load the network
                self.engine_detection = TensorRTEngine(trt_options)
                self.engine_detection.build(self.detection_model["onnx_path"])
                self.engine_detection.loadNetwork()

            # Open segmentation model
            if self.perform_segmentation:
                with open(self.segmentation_model["classes_path"]) as f:
                    self.segmentation_class_names = [line.strip() for line in f]

                # Change option to use the segmentation path
                trt_options.out_file_path = str(Path(self.segmentation_model["onnx_path"]).parent)
                # Create the engine, build and load the network
                self.engine_segmentation = TensorRTEngine(trt_options)
                self.engine_segmentation.build(self.segmentation_model["onnx_path"])
                self.engine_segmentation.loadNetwork()

            # Open depth model
            if self.perform_depth:
                # Change option to use the depth path
                trt_options.out_file_path = str(Path(self.depth_model["onnx_path"]).parent)
                # Create the engine, build and load the network
                self.engine_depth = TensorRTEngine(trt_options)
                self.engine_depth.build(self.depth_model["onnx_path"])
                self.engine_depth.loadNetwork()

            # Open optical flow model
            if self.perform_optical_flow:
                # Change option to use the optical flow path
                trt_options.out_file_path = str(Path(self.optical_flow_model["onnx_path"]).parent)
                # Create the engine, build and load the network
                self.engine_optical_flow = TensorRTEngine(trt_options)
                self.engine_optical_flow.build(self.optical_flow_model["onnx_path"])
                self.engine_optical_flow.loadNetwork()
        
        except Exception as e:
            self.get_logger().error(f"Activation failed. Error: {e}")
            return TransitionCallbackReturn.FAILURE
        
        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")
        self.destroy_subscription(self.sub_image)
        self.sub_image = None
        self.deactivate_pubs(state)
        torch.cuda.empty_cache()
        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        self.destroy_pubs()
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS
    

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        self.deactivate_pubs(state)
        self.destroy_pubs()
        torch.cuda.empty_cache()
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS
    
    def image_cb(self, msg: Image) -> None:

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return
        
        self.img_counter += 1

        img_resized = cv2.resize(cv_image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_tensor = image_to_tensor(img_resized, self.img_mean, self.img_std).unsqueeze(0).to("cuda")

        feats = self.engine_dino.do_inference([img_tensor], out_format='torch')[0]

        if self.perform_detection:
            outputs = self.engine_detection.do_inference([feats], out_format='torch')
            boxes, scores, labels = decode_outputs_tensorrt(outputs, self.img_size, 
                                                            self.detection_model['score_thresh'], self.detection_model['nms_thresh'])
            
            detections_msg = outputs_to_detection2darray(boxes, scores, labels, msg.header)
            self.pub_detections.publish(detections_msg)

            
            # If debug is activated, we draw the image with detections and publish it
            if self.debug:
                img_with_boxes = generate_detection_overlay(img_resized, boxes, scores, labels, class_names=self.detection_class_names)

                # Convert to ROS Image message
                img_detections_msg = self.cv_bridge.cv2_to_imgmsg(img_with_boxes, encoding='rgb8')

                # Publish
                self.pub_img_detections.publish(img_detections_msg)

        if self.perform_segmentation:
            semantic_logits = self.engine_segmentation.do_inference([feats], out_format='torch')[0]

            semantic_map = outputs_to_maps(semantic_logits, (self.img_size, self.img_size))

            sem = np.ascontiguousarray(semantic_map, dtype=np.uint16)  # HxW, class IDs
            sem_seg_msg = self.cv_bridge.cv2_to_imgmsg(sem, encoding='mono16')
            sem_seg_msg.header = msg.header

            self.pub_sem_seg.publish(sem_seg_msg)

            # If debug is activated, we obtain a colored instance map and publish it
            if self.debug:
                segmentation_img = generate_segmentation_overlay(
                    img_resized,
                    semantic_map,
                    class_names=self.segmentation_class_names,
                    alpha=0.6,
                    background_index=0,
                    seed=42,
                    draw_semantic_labels=True, 
                    semantic_label_fontsize=5,
                )

                # Convert to ROS Image message
                img_sem_seg_msg = self.cv_bridge.cv2_to_imgmsg(segmentation_img, encoding='rgb8')

                # Publish
                self.pub_img_sem_seg.publish(img_sem_seg_msg)

        if self.perform_depth:

            depth_np = self.engine_depth.do_inference([feats], out_format='numpy')[0].squeeze()

            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_np, encoding='32FC1')
            depth_msg.header = msg.header

            self.pub_depth.publish(depth_msg)

            # If debug is activated, we obtain a colored depth map and publish it
            if self.debug:
                img_depth = depth_to_colormap(depth_np)
                img_depth_msg = self.cv_bridge.cv2_to_imgmsg(img_depth, encoding='bgr8')
                img_depth_msg.header = msg.header
                self.pub_img_depth.publish(img_depth_msg)

        if self.perform_optical_flow and self.feats_prev is not None:
            optical_flow = self.engine_optical_flow.do_inference([self.feats_prev, feats], out_format = "numpy")[0]

            optical_flow_np = optical_flow.squeeze().transpose(1,2,0).astype(np.float32)
            optical_flow_msg = self.cv_bridge.cv2_to_imgmsg(optical_flow_np, encoding='32FC2')
            optical_flow_msg.header = msg.header

            self.pub_optical_flow.publish(optical_flow_msg)

            if self.debug:
                img_optical_flow = flow_to_image(optical_flow_np, clip_flow_min=3, convert_to_bgr=True)

                img_optical_flow_msg = self.cv_bridge.cv2_to_imgmsg(img_optical_flow, encoding='bgr8')
                img_optical_flow_msg.header = msg.header
                self.pub_img_optical_flow.publish(img_optical_flow_msg)

        self.feats_prev = feats.clone()

    # Helper functions to create publishers
    def make_pub(self, msg_type, topic, depth=10, debug=False):
        p = self.create_lifecycle_publisher(msg_type, topic, depth)
        (self.pubs_debug if debug else self.pubs).append(p)
        return p

    def activate_pubs(self, state: LifecycleState):
        for p in (*self.pubs, *self.pubs_debug):
            try: p.on_activate(state)
            except Exception as e: self.get_logger().warn(f"Activate publisher failed: {e}")

    def deactivate_pubs(self, state: LifecycleState):
        for p in (*self.pubs, *self.pubs_debug):
            try: p.on_deactivate(state)
            except Exception as e: self.get_logger().warn(f"Deactivate publisher failed: {e}")

    def destroy_pubs(self):
        for p in (*self.pubs, *self.pubs_debug):
            try: self.destroy_publisher(p)
            except Exception: pass
        self.pubs.clear()
        self.pubs_debug.clear()

def main():
    rclpy.init()
    node = Dinov3Node()
    node.trigger_configure()
    # Here we only trigger the activate function. In a more complex scenario the changes of state can be managed differently
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
