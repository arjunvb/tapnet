# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation dataset creation functions."""

import csv
import functools
import io
import os
from os import path
import pickle
import random
from typing import Iterable, Mapping, Tuple, Union, Optional

from absl import logging

from .kubric.challenges.point_tracking import dataset
import mediapy as media
import numpy as np
from PIL import Image
import scipy.io as sio
import tensorflow as tf
import tensorflow_datasets as tfds

from .utils import transforms
from particlesfm.particlesfm_tracker.filter import TrajectoryFilter, Trajectories
from contrack_utils.consts import Datasets, GOOD_VIDEOS

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]

TRAIN_SIZE = (24, 256, 256, 3)


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, TRAIN_SIZE[1:3])


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)

    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.

    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.

    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.

    Returns:
        A dict with the following keys:

        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}

    """

    metrics = {}

    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.
    if query_mode == "first":
        for i in range(gt_occluded.shape[0]):
            index = np.where(gt_occluded[i] == 0)[0][0]
            evaluation_points[i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


def latex_table(mean_scalars: Mapping[str, float]) -> str:
    """Generate a latex table for displaying TAP-Vid and PCK metrics."""
    if "average_jaccard" in mean_scalars:
        latex_fields = [
            'average_jaccard',
            'average_pts_within_thresh',
            'occlusion_accuracy',
            'jaccard_1',
            'jaccard_2',
            'jaccard_4',
            'jaccard_8',
            'jaccard_16',
            'pts_within_1',
            'pts_within_2',
            'pts_within_4',
            'pts_within_8',
            'pts_within_16',
        ]
        header = (
            "AJ & $<\\delta^{x}_{avg}$ & OA & Jac. $\\delta^{0}$ & "
            + "Jac. $\\delta^{1}$ & Jac. $\\delta^{2}$ & "
            + "Jac. $\\delta^{3}$ & Jac. $\\delta^{4}$ & $<\\delta^{0}$ & "
            + "$<\\delta^{1}$ & $<\\delta^{2}$ & $<\\delta^{3}$ & "
            + "$<\\delta^{4}$"
        )
    else:
        latex_fields = ["PCK@0.1", "PCK@0.2", "PCK@0.3", "PCK@0.4", "PCK@0.5"]
        header = " & ".join(latex_fields)

    body = " & ".join(
        [f"{float(np.array(mean_scalars[x]*100)):.3}" for x in latex_fields]
    )
    return "\n".join([header, body])


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


def create_jhmdb_dataset(jhmdb_path: str) -> Iterable[DatasetElement]:
    """JHMDB dataset, including fields required for PCK evaluation."""
    gt_dir = jhmdb_path
    videos = []
    for file in tf.io.gfile.listdir(path.join(gt_dir, "splits")):
        # JHMDB file containing the first split, which is standard for this type of
        # evaluation.
        if not file.endswith("split1.txt"):
            continue

        video_folder = "_".join(file.split("_")[:-2])
        for video in tf.io.gfile.GFile(path.join(gt_dir, "splits", file), "r"):
            video, traintest = video.split()
            video, _ = video.split(".")

            traintest = int(traintest)
            video_path = path.join(video_folder, video)

            if traintest == 2:
                videos.append(video_path)

    if not videos:
        raise ValueError("No JHMDB videos found in directory " + str(jhmdb_path))

    # Shuffle so numbers converge faster.
    random.shuffle(videos)

    for video in videos:
        logging.info(video)
        joints = path.join(gt_dir, "joint_positions", video, "joint_positions.mat")

        if not tf.io.gfile.exists(joints):
            logging.info("skip %s", video)
            continue

        gt_pose = sio.loadmat(tf.io.gfile.GFile(joints, "rb"))["pos_img"]
        gt_pose = np.transpose(gt_pose, [1, 2, 0])
        frames = path.join(gt_dir, "Rename_Images", video, "*.png")
        framefil = tf.io.gfile.glob(frames)
        framefil.sort()

        def read_frame(f):
            im = Image.open(tf.io.gfile.GFile(f, "rb"))
            im = im.convert("RGB")
            im_data = np.array(im.getdata(), np.uint8)
            return im_data.reshape([im.size[1], im.size[0], 3])

        frames = [read_frame(x) for x in framefil]
        frames = np.stack(frames)
        height = frames.shape[1]
        width = frames.shape[2]
        invalid_x = np.logical_or(
            gt_pose[:, 0:1, 0] < 0,
            gt_pose[:, 0:1, 0] >= width,
        )
        invalid_y = np.logical_or(
            gt_pose[:, 0:1, 1] < 0,
            gt_pose[:, 0:1, 1] >= height,
        )
        invalid = np.logical_or(invalid_x, invalid_y)
        invalid = np.tile(invalid, [1, gt_pose.shape[1]])
        invalid = invalid[:, :, np.newaxis].astype(np.float32)
        gt_pose_orig = gt_pose

        gt_pose = transforms.convert_grid_coordinates(
            gt_pose,
            np.array([width, height]),
            np.array(TRAIN_SIZE[2:0:-1]),
        )
        # Set invalid poses to -1 (outside the frame)
        gt_pose = (1.0 - invalid) * gt_pose + invalid * (-1.0)

        frames = resize_video(frames, TRAIN_SIZE[1:3])
        frames = frames / (255.0 / 2.0) - 1.0
        queries = gt_pose[:, 0]
        queries = np.concatenate(
            [queries[..., 0:1] * 0, queries[..., ::-1]],
            axis=-1,
        )
        if gt_pose.shape[1] < frames.shape[0]:
            # Some videos have pose sequences that are shorter than the frame
            # sequence (usually because the person disappears).  In this case,
            # truncate the video.
            logging.warning("short video!!")
            frames = frames[:gt_pose.shape[1]]

        converted = {
            "video": frames[np.newaxis, ...],
            "query_points": queries[np.newaxis, ...],
            "target_points": gt_pose[np.newaxis, ...],
            "gt_pose": gt_pose[np.newaxis, ...],
            "gt_pose_orig": gt_pose_orig[np.newaxis, ...],
            "occluded": gt_pose[np.newaxis, ..., 0] * 0,
            "fname": video,
            "im_size": np.array([height, width]),
        }
        yield {"jhmdb": converted}


def create_kubric_eval_train_dataset(
    mode: str,
    max_dataset_size: int = 100,
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on Kubric training data."""
    res = dataset.create_point_tracking_dataset(
        split="train",
        train_size=TRAIN_SIZE[1:3],
        batch_dims=[1],
        shuffle_buffer_size=None,
        repeat=False,
        vflip="vflip" in mode,
        random_crop=False,
    )

    num_returned = 0

    for data in res[0]():
        if num_returned >= max_dataset_size:
            break
        num_returned += 1
        yield {"kubric": data}


def create_kubric_eval_dataset(mode: str) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on Kubric val data."""
    res = dataset.create_point_tracking_dataset(
        split="validation",
        batch_dims=[1],
        shuffle_buffer_size=None,
        repeat=False,
        vflip="vflip" in mode,
        random_crop=False,
    )
    np_ds = tfds.as_numpy(res)

    for data in np_ds:
        yield {"kubric": data}


def create_davis_dataset(
    davis_points_path: str, query_mode: str = "strided", full_resolution=False
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on DAVIS data."""
    pickle_path = davis_points_path

    with tf.io.gfile.GFile(pickle_path, "rb") as f:
        davis_points_dataset = pickle.load(f)

    if full_resolution:
        ds, _ = tfds.load(
            'davis/full_resolution', split='validation', with_info=True
        )
        to_iterate = tfds.as_numpy(ds)
    else:
        to_iterate = davis_points_dataset.keys()

    for tmp in to_iterate:
        if full_resolution:
            frames = tmp['video']['frames']
            video_name = tmp['metadata']['video_name'].decode()
        else:
            video_name = tmp
            frames = davis_points_dataset[video_name]['video']
            frames = resize_video(frames, TRAIN_SIZE[1:3])

        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        target_points = davis_points_dataset[video_name]["points"]
        target_occ = davis_points_dataset[video_name]["occluded"]
        target_points = transforms.convert_grid_coordinates(
            target_points,
            np.array([1.0, 1.0]),
            np.array([frames.shape[-2], frames.shape[-3]]),
        )

        if query_mode == "strided":
            converted = sample_queries_strided(target_occ, target_points, frames)
        elif query_mode == "first":
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            raise ValueError(f"Unknown query mode {query_mode}.")

        yield {"davis": converted}


def create_rgb_stacking_dataset(
    robotics_points_path: str, query_mode: str = "strided"
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on robotics data."""
    pickle_path = robotics_points_path

    with tf.io.gfile.GFile(pickle_path, "rb") as f:
        robotics_points_dataset = pickle.load(f)

    for example in robotics_points_dataset:
        frames = example["video"]
        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        target_points = example["points"]
        target_occ = example["occluded"]
        target_points *= np.array([TRAIN_SIZE[2], TRAIN_SIZE[1]])

        if query_mode == "strided":
            converted = sample_queries_strided(target_occ, target_points, frames)
        elif query_mode == "first":
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            raise ValueError(f"Unknown query mode {query_mode}.")

        yield {"robotics": converted}


def create_kinetics_dataset(
    kinetics_path: str, query_mode: str = "strided"
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on Kinetics point tracking."""

    all_paths = tf.io.gfile.glob(path.join(kinetics_path, "*_of_0010.pkl"))
    for pickle_path in all_paths:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                data = list(data.values())

        # idx = random.randint(0, len(data) - 1)
        for idx in range(len(data)):
            example = data[idx]

            frames = example["video"]

            if isinstance(frames[0], bytes):
                # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
                def decode(frame):
                    byteio = io.BytesIO(frame)
                    img = Image.open(byteio)
                    return np.array(img)

                frames = np.array([decode(frame) for frame in frames])

            if frames.shape[1] > TRAIN_SIZE[1] or frames.shape[2] > TRAIN_SIZE[2]:
                frames = resize_video(frames, TRAIN_SIZE[1:3])

            frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            target_points = example["points"]
            target_occ = example["occluded"]
            target_points *= np.array([TRAIN_SIZE[2], TRAIN_SIZE[1]])

            if query_mode == "strided":
                converted = sample_queries_strided(target_occ, target_points, frames)
            elif query_mode == "first":
                converted = sample_queries_first(target_occ, target_points, frames)
            else:
                raise ValueError(f"Unknown query mode {query_mode}.")

            yield {"kinetics": converted}


def create_davis_split_dataset(
    davis_points_path: str, query_mode: str = "strided"
) -> Iterable[DatasetElement]:
    pickle_path = davis_points_path

    with tf.io.gfile.GFile(pickle_path, "rb") as f:
        davis_points_dataset = pickle.load(f)

    # Need to split labels into distinct trajectories
    for video_name in davis_points_dataset:
        frames = davis_points_dataset[video_name]["video"]
        frames = media.resize_video(frames, TRAIN_SIZE[1:3])
        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        points = davis_points_dataset[video_name]["points"]
        occ = davis_points_dataset[video_name]["occluded"]
        points *= np.array([TRAIN_SIZE[2], TRAIN_SIZE[1]])

        target_points = []
        target_occ = []

        length = points.shape[1]
        valid = ~occ
        for k in range(points.shape[0]):
            # trajectory is valid for whole frame - keep it
            if np.all(valid[k, :]):
                target_points.append(points[k, ...])
                target_occ.append(occ[k, ...])
                continue
            
            # Split trajectory into valid segments
            indices = np.flatnonzero(valid[k, 1:] != valid[k, :-1]) + 1
            new_trajectories = np.split(points[k, ...], indices, axis=0)

            if np.all(new_trajectories[0] == 0.0):
                indices = indices[1::2]
                new_trajectories = new_trajectories[1::2]
            else:
                indices = indices[::2]
                new_trajectories = new_trajectories[::2]

            # Fix for case when trajectory is valid at the end
            if indices.shape[0] == len(new_trajectories) - 1:
                indices = np.concatenate([indices, [length]])

            # Build new trajectories out of valid segments
            for idx, traj in zip(indices, new_trajectories):
                traj_length = traj.shape[0]
                new_traj = np.full((length, 2), 0.0, dtype=np.float32)
                new_occ = np.full((length,), True, dtype=bool)
                new_traj[idx - traj_length:idx, :] = traj
                new_occ[idx - traj_length:idx] = False

                target_points.append(new_traj)
                target_occ.append(new_occ)

        # Concatenate trajectories
        target_points = np.stack(target_points, axis=0)
        target_occ = np.stack(target_occ, axis=0)

        if query_mode == "strided":
            converted = sample_queries_strided(target_occ, target_points, frames)
        elif query_mode == "first":
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            raise ValueError(f"Unknown query mode {query_mode}.")

        yield {"davis": converted}


def __create_sfm_dataset(
    ds: Mapping[str, np.typing.NDArray[np.float32]],
    sfm_path: str,
    query_mode: str = "strided",
    num_samples: int = 256,
    full_length: bool = False,
    video_length: Optional[int] = None,
):
    for video_name in ds:
        frames = ds[video_name]
        # Frames.shape[1:3] is shape [height, width], so we reverse it to be [x, y] format
        frames_shape = frames.shape[1:3][::-1]
        frames = resize_video(frames, TRAIN_SIZE[1:3])
        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0

        # Get ParticleSfM psuedolabels
        trajectories = Trajectories.load(sfm_path, video_name, frames_shape)

        # Reduce video to length video_length
        if video_length is not None:
            frames = frames[:video_length, ...]
            trajectories = trajectories.sliceFrames(video_length)

        # Filter for trajectories valid in all frames
        if full_length:
            trajectories = trajectories.filterFullVideo()

        sampled_trajectories = trajectories.sample(num_samples)
        final_resized_trajectories = sampled_trajectories.resize([TRAIN_SIZE[2], TRAIN_SIZE[1]])
        target_points, valid_mask = final_resized_trajectories.toData()
        target_occ = ~valid_mask

        if query_mode == "strided":
            converted = sample_queries_strided(target_occ, target_points, frames)
        elif query_mode == "first":
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            raise ValueError(f"Unknown query mode {query_mode}.")

        yield converted


def create_sfm_davis_dataset(
    davis_points_path: str,
    davis_sfm_path: str,
    query_mode: str = "strided",
    num_samples: int = 256,
    full_length: bool = False,
    video_length: Optional[int] = None,
) -> Iterable[DatasetElement]:
    pickle_path = davis_points_path

    with tf.io.gfile.GFile(pickle_path, "rb") as f:
        davis_points_dataset: dict = pickle.load(f)

    # Preprocess dataset
    ds = {
        video_name: data["video"] for video_name, data in davis_points_dataset.items()
        if video_name in GOOD_VIDEOS
    }

    sfm_ds = __create_sfm_dataset(
        ds,
        davis_sfm_path,
        query_mode=query_mode,
        num_samples=num_samples,
        full_length=full_length,
        video_length=video_length,
    )

    for converted in sfm_ds:
        yield {"davis": converted}


def create_sfm_lyft_dataset(
    lyft_path: str, 
    lyft_sfm_path: str,
    query_mode: str = "strided",
    num_samples: int = 256,
    full_length: bool = False
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on Lyft point tracking."""
    paths = tf.io.gfile.glob(os.path.join(lyft_path, "tracks/track_*.pkl"))

    ds = {}
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)

        track_idx = path.split("/")[-1].split(".")[0].split("_")[-1]
        track_name = f"track_{track_idx:0>5}"
        images = [media.read_image(track["rgb_path"]) for track in data["track_frames"]]
        if len(images) == 0:
            continue

        frames = np.stack(images)
        ds[track_name] = frames

    sfm_ds = __create_sfm_dataset(
        ds,
        lyft_sfm_path,
        query_mode=query_mode,
        num_samples=num_samples,
        full_length=full_length
    )

    for converted in sfm_ds:
        yield {"lyft": converted}


    
