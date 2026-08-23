"""OnlineSolver：VGGT-SLAM Solver 的显存优化子类。

上游 vggt_slam.solver.Solver 会把每个子图的帧张量常驻 GPU
（Submap.add_all_frames(images)，images 在 cuda 上）。在 8GB 级
GPU 上，子图累积 + Habitat 渲染 + SALAD/VGGT 常驻会很快 OOM。

本子类复制上游 run_predictions 逻辑，仅做显存相关改动：

1. 子图帧张量保存到 CPU（回环需要时再临时搬回 GPU）。
2. 语义记忆由独立 caption worker 处理，不在 SLAM 前向中计算图像向量。
"""

import time

import torch
from termcolor import colored

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.solver import Solver
from vggt_slam.submap import Submap


class OnlineSolver(Solver):
    def run_predictions(self, image_names, model, max_loops):
        if not torch.cuda.is_available():
            raise RuntimeError("OnlineSolver requires CUDA")
        device = "cuda"
        t1 = time.time()
        with self.vggt_timer:
            images = load_and_preprocess_images(image_names).to(device)
        print(
            f"Loaded and preprocessed {len(image_names)} images in "
            f"{time.time() - t1:.2f} seconds")
        print(f"Preprocessed images shape: {images.shape}")
        img_hw = images.shape[-2:]

        # Submap ids follow the last persistent non-loop frame.
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = (
                self.map.get_largest_key()
                + self.map.get_latest_submap().get_last_non_loop_frame_index()
                + 1)

        print(f"Creating new submap with id {new_pcd_num}")
        t1 = time.time()
        new_submap = Submap(new_pcd_num)
        # Persistent submap frames live on CPU to bound GPU memory growth.
        new_submap.add_all_frames(images.cpu())
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors(
            self.image_retrieval.get_all_submap_embeddings(new_submap))
        new_submap.set_img_names(image_names)

        self.current_working_submap = new_submap
        print(f"Created new submap in {time.time() - t1:.2f} seconds")

        with torch.no_grad():
            t1 = time.time()
            with self.vggt_timer:
                predictions = model(images)
            print(f"VGGT model inference took {time.time() - t1:.2f} seconds")

        # 主前向结束后即可释放输入帧的显存
        del images
        torch.cuda.empty_cache()

        # Loop closure uses only the selected frame pair on GPU.
        predictions_lc = None
        with self.loop_closure_timer:
            detected_loops = self.image_retrieval.find_loop_closures(
                self.map, new_submap, max_loop_closures=max_loops,
                max_similarity_thres=self.lc_thres)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            retrieved_frames = self.map.get_frames_from_loops(detected_loops)
            with torch.no_grad():
                query_frame = new_submap.get_frame_at_index(
                    detected_loops[0].query_submap_frame)
                lc_frames = torch.stack(
                    (query_frame, retrieved_frames[0]), axis=0).to(device)
                predictions_lc = model(lc_frames, compute_similarity=True)
                detected_submap = self.map.get_submap(
                    detected_loops[0].detected_submap_id)
                loop_closure_frame_names = [
                    new_submap.get_img_names_at_index(
                        detected_loops[0].query_submap_frame),
                    detected_submap.get_img_names_at_index(
                        detected_loops[0].detected_submap_frame),
                ]

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], img_hw)
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        predictions["detected_loops"] = detected_loops

        if predictions_lc is not None:
            image_match_ratio = predictions_lc["image_match_ratio"]
            if image_match_ratio < 0.95:
                print(colored(
                    "Loop closure image match ratio too low; skipping", "red"))
                predictions_lc = None
                predictions["detected_loops"] = []
            else:
                self.graph.increment_loop_closure()
                extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(
                    predictions_lc["pose_enc"], retrieved_frames[0].shape[-2:])
                predictions["extrinsic_lc"] = extrinsic_lc
                predictions["intrinsic_lc"] = intrinsic_lc
                predictions["depth_lc"] = predictions_lc["depth"]
                predictions["depth_conf_lc"] = predictions_lc["depth_conf"]

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor) and key != "target_tokens":
                predictions[key] = value.float().cpu().numpy().squeeze(0)

        if predictions_lc is not None:
            predictions["frames_lc"] = lc_frames[0:2, ...]
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names

        torch.cuda.empty_cache()
        return predictions
