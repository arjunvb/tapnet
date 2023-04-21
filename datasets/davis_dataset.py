import random
import functools
import pickle
from typing import Tuple, Mapping
import numpy as np

import mediapy as media

import tensorflow as tf
import tensorflow_datasets as tfds


def get_pseudolabels(path: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
    npzfile = np.load(f"{path}/{name}/{name}.npz")
    return npzfile["xy"], npzfile["masks"]


def resize_video(video: tf.Tensor, output_size: Tuple[int, int]) -> tf.Tensor:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return tf.image.resize(video, output_size)


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
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
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
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
        "query_points": np.concatenate(queries, axis=0),
        "target_points": np.concatenate(tracks, axis=0),
        "occluded": np.concatenate(occs, axis=0),
    }


def add_tracks(
    video_name: tf.Tensor,
    video: tf.Tensor,
    segmentations: tf.Tensor,
    train_size: Tuple[int, int] = (256, 256),
    vflip=False,
    random_crop=True,
    tracks_to_sample=256,
    sampling_stride=4,
    max_seg_id=25,
    max_sampled_frac=0.1,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # Get ParticleSfM psuedolabels
    base_path = "/microtel/scratch/particlesfm_labels2/davis"
    name = video_name.numpy().decode()
    xy, masks = get_pseudolabels(base_path, name)

    # Resize the video and segmentations to the square train size
    video = resize_video(video, train_size)
    segmentations = resize_video(segmentations, train_size)

    # We want more points that lie on an object, so we do this
    # weighting scheme to bias towards tracks that start on an
    # object based off its segmentation
    first_segmentation = segmentations[0]
    object_points = set(tuple(pt)[0:2] for pt in np.argwhere(first_segmentation == 1))

    object_mask = np.zeros((xy.shape[0],))
    for track in range(xy.shape[0]):
        pt = tuple(xy[track, 0, :].astype(int))[::-1]
        if pt in object_points:
            object_mask[track] = 1

    weights = np.where(object_mask == 1, 1000, 1)

    # Get target points
    possible_points = np.arange(xy.shape[0])
    sampled_tracks = np.array(random.choices(possible_points, weights=weights, k=tracks_to_sample))
    target_points = xy[sampled_tracks, ...]
    occluded = ~(masks[sampled_tracks, :, 0].astype(bool))

    # Sample query points
    ret = sample_queries_strided(occluded, target_points, sampling_stride)

    return (
        video / (255.0 / 2.0) - 1.0,
        ret["query_points"],
        ret["target_points"],
        ret["occluded"],
    )


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
        "davis/480p",
        split=split,
        data_dir="/microtel/scratch/davis/",
        shuffle_files=shuffle_buffer_size is not None,
        **kwargs,
    )

    if repeat:
        ds = ds.repeat()

    # Workaround to load trajectories out of file storage
    ds = ds.map(
        lambda data: tf.py_function(
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
            [
                data["metadata"]["video_name"],
                data["video"]["frames"],
                data["video"]["segmentations"],
            ],
            [tf.float32, tf.float32, tf.float32, tf.bool],
        )
    )

    ds = ds.map(
        lambda video, query_points, target_points, occluded: {
            "query_points": query_points,
            "target_points": target_points,
            "occluded": occluded,
            "video": video,
        }
    )

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    for bs in batch_dims[::-1]:
        ds = ds.batch(bs)

    return ds


if __name__ == "__main__":
    ds = create_point_tracking_dataset(shuffle_buffer_size=None)
    for example in ds.take(1):
        print(example)

