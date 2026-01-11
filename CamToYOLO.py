import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge
import cv2
import time

from ultralytics import YOLO

import tf2_ros
import tf2_geometry_msgs  # 중요


TARGET_CLASSES = ["cup"]

# ===============================
# base_link 바닥 보정값 (m)
# ===============================
BASE_HEIGHT = 0.15


class YoloDepthToBaseViz(Node):
    def __init__(self):
        super().__init__('yolo_depth_to_base_viz')

        self.bridge = CvBridge()

        # -----------------------------
        # YOLO 주기 제한
        # -----------------------------
        self.last_yolo_time = 0.0
        self.yolo_interval = 0.2  # 5 FPS

        # -----------------------------
        # Camera intrinsics
        # -----------------------------
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # -----------------------------
        # Depth image
        # -----------------------------
        self.depth_image = None

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.sub_color = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )

        self.sub_depth = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )

        # -----------------------------
        # YOLO
        # -----------------------------
        self.model = YOLO("yolov8n.pt")

        # -----------------------------
        # TF
        # -----------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("YOLO + Depth + TF + Visualization node started")

    # ======================================================
    # CameraInfo callback
    # ======================================================
    def info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

            self.get_logger().info(
                f"Camera intrinsics received "
                f"(fx={self.fx:.1f}, fy={self.fy:.1f})"
            )

    # ======================================================
    # Depth callback
    # ======================================================
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough'
        )

    # ======================================================
    # Color / YOLO callback
    # ======================================================
    def color_callback(self, msg):

        # YOLO 주기 제한
        now = time.time()
        if now - self.last_yolo_time < self.yolo_interval:
            return
        self.last_yolo_time = now

        if self.depth_image is None or self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        annotated = frame.copy()

        results = self.model(
            frame,
            imgsz=480,
            conf=0.5,
            verbose=False
        )

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            if v >= self.depth_image.shape[0] or u >= self.depth_image.shape[1]:
                continue

            depth_mm = int(self.depth_image[v, u])
            if depth_mm == 0:
                continue

            # ---------------------------------
            # Pixel + depth → Camera 3D
            # ---------------------------------
            Zc = depth_mm / 1000.0
            Xc = (u - self.cx) * Zc / self.fx
            Yc = (v - self.cy) * Zc / self.fy

            # ---------------------------------
            # TF → base_link
            # ---------------------------------
            point_cam = PointStamped()
            point_cam.header.frame_id = "camera_color_optical_frame"
            point_cam.header.stamp = self.get_clock().now().to_msg()
            point_cam.point.x = float(Xc)
            point_cam.point.y = float(Yc)
            point_cam.point.z = float(Zc)

            try:
                point_base = self.tf_buffer.transform(
                    point_cam,
                    "base_link",
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )

                Xb = point_base.point.x
                Yb = point_base.point.y
                Zb = point_base.point.z

            except Exception as e:
                self.get_logger().warn(f"TF failed: {e}")
                continue

            # ---------------------------------
            # Z 보정 (바닥 기준)
            # ---------------------------------
            Z_floor = Zb - BASE_HEIGHT

            # ---------------------------------
            # Visualization
            # ---------------------------------
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (u, v), 5, (0, 0, 255), -1)

            text1 = f"{cls_name} conf={conf:.2f}"
            text2 = f"Zc={Zc:.2f}m"
            text3 = f"X={Xb:.2f} Y={Yb:.2f} Z={Z_floor:.2f}"

            cv2.putText(annotated, text1, (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated, text2, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated, text3, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("YOLO + Depth + Base (Z corrected)", annotated)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = YoloDepthToBaseViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
