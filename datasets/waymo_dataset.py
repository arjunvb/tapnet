import os
import functools
import pickle
from typing import Tuple, Mapping, Optional
import numpy as np
import numpy.typing as npt

import mediapy as media

import tensorflow as tf
import tensorflow_datasets as tfds

from particlesfm.particlesfm_tracker.filter import Trajectories


def resize_video(video: tf.Tensor, output_size: Tuple[int, int]) -> tf.Tensor:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return tf.image.resize(video, output_size)


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


def add_tracks(
    data,
    train_size=(256, 256),
    vflip=False,
    random_crop=True,
    tracks_to_sample=256,
    sampling_stride=4,
    max_seg_id=25,
    max_sampled_frac=0.1,
    crop_window=None,
):
    video = data["video"]
    _, height, width, _ = video.shape.as_list()
    target_points = tf.cast(data["target_points"], tf.float32)
    visibles = data["visibles"]
    num_points = tf.shape(target_points)[0]

    video = resize_video(video, train_size)
    video = video / (255.0 / 2.0) - 1.0

    ratio = tf.constant(train_size, dtype=tf.float32) / tf.constant(
        [width, height], dtype=tf.float32
    )
    target_points = target_points * ratio

    queries = []
    targets = []
    vis = []
    total = tf.constant(0)
    for i in range(1, 24, sampling_stride):
        mask = tf.reshape(tf.where(visibles[:, i] == 1), [-1])
        num_visible = tf.shape(mask)[0]
        total += num_visible
        query = tf.stack(
            [
                float(i) * tf.ones(num_points, dtype=tf.float32),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(tf.gather(query, mask, axis=0))
        targets.append(tf.gather(target_points, mask, axis=0))
        vis.append(tf.gather(visibles, mask, axis=0))

    query_points = tf.concat(queries, axis=0)
    target_points = tf.concat(targets, axis=0)
    visibles = tf.concat(vis, axis=0)
    
    if tf.shape(target_points)[0] < tracks_to_sample:
        tile_factor = (tracks_to_sample // total) + 1
        tf1 = tf.stack((tile_factor, tf.constant(1)), axis=0)
        tf2 = tf.concat((tf1, tf.constant([1])), axis=0)
        query_points = tf.tile(query_points, tf1)
        target_points = tf.tile(target_points, tf2)
        visibles = tf.tile(visibles, tf1)

    indices = tf.range(tf.shape(target_points)[0])
    samples = tf.random.shuffle(indices)[:tracks_to_sample]

    query_points = tf.gather(query_points, samples, axis=0)
    target_points = tf.gather(target_points, samples, axis=0)
    visibles = tf.gather(visibles, samples, axis=0)
    target_occluded = ~visibles
    query_points.set_shape([tracks_to_sample, 3])
    target_points.set_shape([tracks_to_sample, 24, 2])
    target_occluded.set_shape([tracks_to_sample, 24])

    if vflip:
        video = tf.reverse(video, axis=1)
        target_points = target_points * tf.constant([1, -1])
        query_points = query_points * tf.constant([1, -1, 1])

    return {
        "query_points": query_points,
        "target_points": target_points,
        "occluded": target_occluded,
        "video": video,  # / (255.0 / 2.0) - 1.0,
    }


def create_point_tracking_dataset(
    train_size=(256, 256),
    shuffle_buffer_size=256,
    split="train",
    batch_dims=tuple(),
    repeat=True,
    vflip=False,
    random_crop=True,
    tracks_to_sample=256,
    sampling_stride=4,
    max_seg_id=25,
    max_sampled_frac=0.1,
    num_parallel_point_extraction_calls=16,
    **kwargs,
):
    """Construct a dataset for point tracking using Davis.

    Args:
        train_size: Tuple of 2 ints. Cropped output will be at this resolution
        shuffle_buffer_size: Int. Size of the shuffle buffer
        split: Which split to construct from Kubric.  Can be 'train' or
        'validation'.
        batch_dims: Sequence of ints. Add multiple examples into a batch of this
        shape.
        repeat: Bool. whether to repeat the dataset.
        vflip: Bool. whether to vertically flip the dataset to test generalization.
        random_crop: Bool. whether to randomly crop videos
        tracks_to_sample: Int. Total number of tracks to sample per video.
        sampling_stride: Int. For efficiency, query points are sampled from a
        random grid of this stride.
        max_seg_id: Int. The maxium segment id in the video.  Note the size of
        the to graph is proportional to this number, so prefer small values.
        max_sampled_frac: Float. The maximum fraction of points to sample from each
        object, out of all points that lie on the sampling grid.
        num_parallel_point_extraction_calls: Int. The num_parallel_calls for the
        map function for point extraction.
        **kwargs: additional args to pass to tfds.load.

    Returns:
        The dataset generator.
    """
    ds: tf.data.Dataset = tfds.load(
        "waymo_point_track/1280x1920",
        split=split,
        data_dir="/microtel/nfs/datasets/tfds",
        shuffle_files=shuffle_buffer_size is not None,
        **kwargs,
    )

    if repeat:
        ds = ds.repeat()

    ds = ds.filter(lambda x: tf.shape(x["target_points"])[0] >= tracks_to_sample)
    ds = ds.filter(lambda x: tf.shape(x["visibles"])[0] >= tracks_to_sample)
    ds = ds.filter(
        lambda x: tf.reduce_any(tf.cast(x["visibles"], tf.bool)[:, ::sampling_stride])
    )

    ds = ds.map(
        functools.partial(
            add_tracks,
            train_size=train_size,
            vflip=vflip,
            random_crop=random_crop,
            tracks_to_sample=tracks_to_sample,
            sampling_stride=sampling_stride,
            max_seg_id=max_seg_id,
            max_sampled_frac=max_sampled_frac,
        ),
        num_parallel_calls=num_parallel_point_extraction_calls,
    )

    ds = ds.filter(lambda x: x is not None)

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    for bs in batch_dims[::-1]:
        ds = ds.batch(bs)

    return ds


if __name__ == "__main__":
    # tf.data.experimental.enable_debug_mode()

    ds = create_point_tracking_dataset(shuffle_buffer_size=None)
    for i, example in enumerate(ds):
        print("EXAMPLE:", i)
        print(example["video"].shape)
        print(example["target_points"].shape)
        print(example["query_points"].shape)
        print(example["occluded"].shape)
        # print(example["target_points"].shape)
