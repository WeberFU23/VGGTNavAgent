"""校验重力对齐的安装俯仰角补偿。

原理：agent 在同一楼层移动时相机中心始终离地 1.5m（配置常量），
俯仰角补偿正确时，对齐坐标系下相机中心高度的散布最小；
补偿角度错误会把水平位移混入高度，轨迹水平范围越大散布越大。

用法（在 habitat 环境，eval 跑完、mapping server 仍持图时执行）::

    PYTHONPATH=/path/to/vggt_nav_agent python scripts/diagnostics/check_gravity.py \
        --port 5555

输出每个候选俯仰角下的高度散布，并给出推荐值。
"""

import argparse
import math

import numpy as np

from agents import navigator as nav
from mapping.client import MappingClient


def height_stats(poses, pitch_down_rad):
    """返回 (相机中心高度数组, 90%区间宽度, std)。"""
    cam_up = nav.mount_compensated_cam_up(pitch_down_rad)
    R = nav.gravity_alignment(poses, cam_up=cam_up)
    h = poses[:, :3, 3] @ R[2]
    spread = float(np.percentile(h, 95) - np.percentile(h, 5))
    return h, spread, float(h.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    client = MappingClient(host=args.host, port=args.port)
    poses, frame_ids = client.get_all_poses()
    if poses is None or len(poses) < 5:
        print("位姿不足（需要先在 server 里建过图）")
        return
    poses = np.asarray(poses, dtype=np.float64)
    print(f"共 {len(poses)} 个关键帧位姿")

    best_deg, best_spread = None, float("inf")
    print(f"{'pitch_down':>10} {'spread90':>10} {'std':>10}")
    for deg in range(-45, 46, 5):
        _, spread, std = height_stats(poses, math.radians(deg))
        marker = ""
        if spread < best_spread:
            best_deg, best_spread = deg, spread
        print(f"{deg:>10} {spread:>10.4f} {std:>10.4f}")

    print(f"\n推荐 mount pitch_down = {best_deg}°（散布 {best_spread:.4f} 地图单位）")
    if best_deg != 30:
        print("注意：与配置的安装角 30° 不一致，"
              "可能是 habitat 俯仰角符号约定相反，请更新 navigator.MOUNT_PITCH_DOWN_RAD")

    # 用轨迹 PCA 给安装角估计提供独立交叉检查。
    centers = poses[:, :3, 3]
    cov = np.cov((centers - centers.mean(0)).T)
    eigval, eigvec = np.linalg.eigh(cov)
    up_pca = eigvec[:, 0]
    cam_up = nav.mount_compensated_cam_up(math.radians(best_deg))
    R = nav.gravity_alignment(poses, cam_up=cam_up)
    if np.dot(up_pca, R[2]) < 0:
        up_pca = -up_pca
    h = centers @ up_pca
    spread_pca = float(np.percentile(h, 95) - np.percentile(h, 5))
    angle = math.degrees(math.acos(
        min(1.0, abs(float(np.dot(up_pca, R[2]))))))
    planarity = float(eigval[0] / max(eigval[1], 1e-12))
    extent = math.sqrt(float(eigval[2]))
    print(f"\n[PCA up] spread90={spread_pca:.4f} std={h.std():.4f} "
          f"与中位数法夹角={angle:.2f}°")
    print(f"[PCA up] 平面性 lambda_min/lambda_mid={planarity:.4f} "
          f"(<0.05 才可信赖), 轨迹范围={extent:.2f} 单位")


if __name__ == "__main__":
    main()
