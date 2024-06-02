# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import io
import glob
import torch
import pickle
import numpy as np
import mediapy as media
import imageio

from PIL import Image
from typing import Mapping, Tuple, Union

from models.spatracker.datasets.utils import CoTrackerData

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


class KubricEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        resize_to_256=True,
        queried_first=True,
    ):
        self.resize_to_256 = resize_to_256
        self.queried_first = queried_first
        self.seq_names = [
            fname
            for fname in os.listdir(self.dataRootFrames)
            if os.path.isdir(os.path.join(self.dataRootFrames, fname))
        ]
        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

    def __getitem__(self, index):
        seq_name = self.seq_names[index]
        npy_path = os.path.join(self.data_root, "512x512_anno", seq_name + ".npy")
        rgb_path = os.path.join(self.dataRootFrames, seq_name)
        depth_path = os.path.join(self.dataRootDepth, seq_name)

        img_paths = sorted(os.listdir(rgb_path))
        frames = []
        for i, img_path in enumerate(img_paths):
            frames.append(imageio.v2.imread(os.path.join(rgb_path, img_path)))
        frames = np.stack(frames) # T, H, W, 3
        dp_paths = sorted(os.listdir(depth_path))
        depths = []
        for i, dp_path in enumerate(dp_paths):
            depths.append(imageio.v2.imread(os.path.join(depth_path, dp_path)))
        depths = np.stack(depths)[..., None]   # T, H, W, 1

        annot_dict = np.load(npy_path, allow_pickle=True).item()
        target_points = annot_dict["target_points"].squeeze()
        target_occ = annot_dict["occluded"].squeeze()
        traj_3d = annot_dict["CamCoordPos"].squeeze()

        T, H, W, C = frames.shape
        N, T, D = target_points.shape

        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float() # N, T, D
        )  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        segs = torch.ones(T, 1, H, W).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[
            0
        ].permute(
            1, 0
        )  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N
        return CoTrackerData(
            video=rgbs,
            segmentation=segs,
            trajectory=trajs,
            visibility=visibles,
            seq_name=str(seq_name),
            query_points=query_points,
        )

    def __len__(self):
        return len(self.seq_names)
