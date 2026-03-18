#!/usr/bin/env python3
"""LSDD-Net ROS推理节点。

以10Hz频率运行, 从MDOB/状态估计中读取数据,
通过Mamba step模式逐帧更新隐状态, 发布世界系/机体系低频力估计。

部署策略:
  - 传感器数据以100Hz到达, 节点缓存最近10帧
  - 每0.1s唤醒, 逐帧处理缓冲区中的10帧 (保持与训练一致的100Hz时间分辨率)
  - 取最后一帧输出发布
"""

import os
import sys
import threading
from collections import deque

import yaml
import torch
import numpy as np

# ROS (仅在有ROS环境时导入)
try:
    import rospy
    from geometry_msgs.msg import Vector3Stamped, PoseStamped, TwistStamped
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("警告: 未找到ROS环境, 仅可在离线模式下使用")

# 添加包路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsdd_net.model import LSDDNet
from lsdd_net.normalize import Normalizer


class SensorBuffer:
    """线程安全的传感器数据缓冲区。

    缓存最近N帧的传感器数据, 供推理线程批量读取。
    """

    def __init__(self, max_size: int = 20):
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=max_size)
        self.latest_pose = None
        self.latest_vel = None
        self.latest_mdob_w = None
        self.latest_mdob_b = None

    def update_pose(self, q_wxyz: np.ndarray):
        with self.lock:
            self.latest_pose = q_wxyz.copy()

    def update_velocity(self, vw: np.ndarray, vb: np.ndarray):
        with self.lock:
            self.latest_vel = (vw.copy(), vb.copy())

    def update_mdob(self, fw: np.ndarray, fb: np.ndarray):
        """收到新的MDOB数据, 组装完整帧并压入缓冲区。"""
        with self.lock:
            self.latest_mdob_w = fw.copy()
            self.latest_mdob_b = fb.copy()

            # 只有所有数据都就绪时才压入缓冲区
            if self.latest_pose is not None and self.latest_vel is not None:
                frame = {
                    "fw": self.latest_mdob_w,
                    "fb": self.latest_mdob_b,
                    "vw": self.latest_vel[0],
                    "vb": self.latest_vel[1],
                    "q": self.latest_pose,
                }
                self.buffer.append(frame)

    def get_recent_frames(self, n: int) -> list:
        """获取最近n帧数据 (线程安全)。"""
        with self.lock:
            frames = list(self.buffer)
            return frames[-n:] if len(frames) >= n else frames


class LSDDInferenceNode:
    """LSDD-Net ROS推理节点。"""

    def __init__(self):
        rospy.init_node("lsdd_inference", anonymous=False)

        # 加载配置
        config_path = rospy.get_param("~config", "")
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                self.cfg = yaml.safe_load(f)
        else:
            # 从ROS参数服务器读取
            self.cfg = {
                "model": {
                    "checkpoint": rospy.get_param("~model/checkpoint", ""),
                    "norm_stats": rospy.get_param("~model/norm_stats", ""),
                },
                "ros": {
                    "rate": rospy.get_param("~ros/rate", 10),
                    "mass": rospy.get_param("~ros/mass", 1.5),
                    "sensor_rate": rospy.get_param("~ros/sensor_rate", 100),
                },
            }

        rate = self.cfg["ros"]["rate"]
        sensor_rate = self.cfg["ros"]["sensor_rate"]
        self.frames_per_cycle = sensor_rate // rate  # 每次推理处理的帧数 (10)

        # 加载模型
        self._load_model()

        # 传感器缓冲区
        self.sensor_buf = SensorBuffer(max_size=self.frames_per_cycle * 2)

        # ROS话题
        topics = self.cfg.get("ros", {}).get("topics", {})

        # 订阅
        rospy.Subscriber(
            topics.get("pose", "/mavros/local_position/pose"),
            PoseStamped, self._pose_cb, queue_size=1,
        )
        rospy.Subscriber(
            topics.get("velocity", "/mavros/local_position/velocity_local"),
            TwistStamped, self._vel_cb, queue_size=1,
        )
        rospy.Subscriber(
            topics.get("mdob_world", "/lsdd/mdob_world"),
            Vector3Stamped, self._mdob_w_cb, queue_size=1,
        )
        rospy.Subscriber(
            topics.get("mdob_body", "/lsdd/mdob_body"),
            Vector3Stamped, self._mdob_b_cb, queue_size=1,
        )

        # 发布
        self.pub_w = rospy.Publisher(
            topics.get("pub_force_world", "/lsdd/force_world"),
            Vector3Stamped, queue_size=1,
        )
        self.pub_b = rospy.Publisher(
            topics.get("pub_force_body", "/lsdd/force_body"),
            Vector3Stamped, queue_size=1,
        )

        # 缓存最新的body frame MDOB (因为两个话题分开到达)
        self._latest_mdob_b = np.zeros(3, dtype=np.float32)

        # Mamba隐状态
        self.states = None

        # 定时器
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / rate), self._timer_cb,
        )

        rospy.loginfo(
            f"LSDD推理节点已启动: {rate}Hz推理, "
            f"每周期处理{self.frames_per_cycle}帧, "
            f"模型参数量={self.model.count_parameters()['total']}"
        )

    def _load_model(self):
        """加载模型和标准化参数。"""
        model_cfg = self.cfg["model"]
        ckpt_path = model_cfg["checkpoint"]
        norm_path = model_cfg["norm_stats"]

        # 标准化器
        self.normalizer = Normalizer()
        self.normalizer.load(norm_path)

        # 模型
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m_cfg = ckpt.get("model_config", ckpt.get("config", {}).get("model", {}))
        self.model = LSDDNet(**m_cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        rospy.loginfo(f"模型已加载: {ckpt_path}")

    def _pose_cb(self, msg: PoseStamped):
        """姿态回调: 缓存四元数 (w,x,y,z)。"""
        q = msg.pose.orientation
        q_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        self.sensor_buf.update_pose(q_wxyz)

    def _vel_cb(self, msg: TwistStamped):
        """速度回调: 缓存世界系和机体系速度。"""
        vw = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
        ], dtype=np.float32)

        # 机体系速度需要用姿态旋转
        # 这里用最新的姿态近似
        with self.sensor_buf.lock:
            q = self.sensor_buf.latest_pose
        if q is not None:
            from lsdd_net.rotation_utils import quat_to_rotmat, rotate_vector_inv
            R = quat_to_rotmat(torch.from_numpy(q).unsqueeze(0))
            vb = rotate_vector_inv(R, torch.from_numpy(vw).unsqueeze(0))[0].numpy()
        else:
            vb = vw.copy()

        self.sensor_buf.update_velocity(vw, vb)

    def _mdob_w_cb(self, msg: Vector3Stamped):
        """MDOB世界系力回调。"""
        fw = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float32)
        self.sensor_buf.update_mdob(fw, self._latest_mdob_b)

    def _mdob_b_cb(self, msg: Vector3Stamped):
        """MDOB机体系力回调。"""
        self._latest_mdob_b = np.array(
            [msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float32
        )

    @torch.no_grad()
    def _timer_cb(self, event):
        """10Hz推理定时器回调。"""
        # 获取最近N帧
        frames = self.sensor_buf.get_recent_frames(self.frames_per_cycle)
        if not frames:
            return

        # 逐帧通过Mamba step (保持100Hz时间分辨率)
        for frame in frames:
            fw = torch.from_numpy(frame["fw"]).unsqueeze(0)
            fb = torch.from_numpy(frame["fb"]).unsqueeze(0)
            vw = torch.from_numpy(frame["vw"]).unsqueeze(0)
            vb = torch.from_numpy(frame["vb"]).unsqueeze(0)
            q  = torch.from_numpy(frame["q"]).unsqueeze(0)

            # 标准化
            fw = self.normalizer.transform("fw", fw)
            fb = self.normalizer.transform("fb", fb)
            vw = self.normalizer.transform("vw", vw)
            vb = self.normalizer.transform("vb", vb)

            F_hat_w, F_hat_b, self.states = self.model.step(
                fw, fb, vw, vb, q, self.states,
            )

        # 反标准化最后一帧的输出
        F_hat_w = self.normalizer.inverse_transform("wind_gt", F_hat_w)
        F_hat_b = self.normalizer.inverse_transform("bf_gt", F_hat_b)

        # 发布
        now = rospy.Time.now()

        msg_w = Vector3Stamped()
        msg_w.header.stamp = now
        msg_w.header.frame_id = "world"
        msg_w.vector.x = float(F_hat_w[0, 0])
        msg_w.vector.y = float(F_hat_w[0, 1])
        msg_w.vector.z = float(F_hat_w[0, 2])
        self.pub_w.publish(msg_w)

        msg_b = Vector3Stamped()
        msg_b.header.stamp = now
        msg_b.header.frame_id = "body"
        msg_b.vector.x = float(F_hat_b[0, 0])
        msg_b.vector.y = float(F_hat_b[0, 1])
        msg_b.vector.z = float(F_hat_b[0, 2])
        self.pub_b.publish(msg_b)


def main():
    if not HAS_ROS:
        print("错误: 需要ROS环境 (rospy)")
        sys.exit(1)

    node = LSDDInferenceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
