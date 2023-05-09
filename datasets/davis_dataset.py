import functools
from typing import Tuple, Mapping
import numpy as np
import numpy.typing as npt

import mediapy as media

import tensorflow as tf
import tensorflow_datasets as tfds

from particlesfm.particlesfm_tracker.filter import TrajectoryFilter


def get_pseudolabels(
    path: str, name: str
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.bool_]]:
    npzfile = np.load(f"{path}/{name}/{name}.npz")
    trajectories: npt.NDArray[np.float_] = npzfile["xy"]
    valid_mask: npt.NDArray[np.int_] = npzfile["masks"]
    return (
        trajectories,
        valid_mask.squeeze().astype(bool),
    )


def resize_video(video: tf.Tensor, output_size: Tuple[int, int]) -> tf.Tensor:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return tf.image.resize(video, output_size)

def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
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
    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "query_points": query_points,
        "target_points": target_points,
        "occluded": target_occluded,
    }


def add_tracks(
    video_name: tf.Tensor,
    video: tf.Tensor,
    segmentations: tf.Tensor,
    train_size: Tuple[int, int] = (256, 256),
    vflip=False,
    random_crop=True,
    tracks_to_sample=256,
    video_length=24,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # Get ParticleSfM psuedolabels
    base_path = "/microtel/nfs/particlesfm_labels2/davis"
    name = video_name.numpy().decode()
    trajectories, valid_mask = get_pseudolabels(base_path, name)

    # Reduce video to length video_length
    video = video[:video_length, ...]
    trajectories = trajectories[:, :video_length, :]
    valid_mask = valid_mask[:, :video_length].squeeze()

    # Filter for trajectories valid in the first video_length frames
    valid_trajs = np.argwhere(np.any(valid_mask, axis=1)).squeeze()
    trajectories = trajectories[valid_trajs, ...]
    valid_mask = valid_mask[valid_trajs, ...]

    # Compute video size ratio
    video_size_arr = tf.shape(video)[1:3].numpy()
    train_size_arr = np.array(train_size)
    ratio = train_size_arr / video_size_arr[::-1]

    # Resize
    trajectories *= ratio
    video = resize_video(video, train_size)
    segmentations = resize_video(segmentations, train_size)

    trajectoryFilter = TrajectoryFilter(
        trajectory_length=5,
        object_masks=segmentations.numpy(),
        mask_threshold=0.5,
        video_shape=train_size_arr,
    )

    filt_traj, filt_valid_mask = trajectoryFilter.filterLength(trajectories, valid_mask)

    mask_trajectories, non_mask_trajectories = trajectoryFilter.splitTrajectoryTypes(filt_traj, filt_valid_mask)

    # Compute track samples
    # Selects at most 80% of keypoints to be on an object, the rest are random
    object_samples = int(tracks_to_sample * 0.8)
    num_mask_samples = min(mask_trajectories.shape[0], object_samples)
    num_non_mask_samples = tracks_to_sample - num_mask_samples
    mask_samples = np.random.choice(mask_trajectories, num_mask_samples, replace=False)
    non_mask_samples = np.random.choice(
        non_mask_trajectories, num_non_mask_samples, replace=False
    )
    samples = np.hstack((mask_samples, non_mask_samples)).astype(int)

    # Samples tracks into proper format
    target_points = filt_traj[samples]
    occluded = ~(filt_valid_mask[samples])

    # Sample query points - takes first occurence of the track
    ret = sample_queries_first(occluded, target_points)

    query_points = ret["query_points"]
    target_points = ret["target_points"]
    occluded = ret["occluded"]

    if vflip:
        video = video[:, ::-1, :, :]
        target_points = target_points * np.array([1, -1])
        query_points = query_points * np.array([1, -1, 1])

    return (
        video / (255.0 / 2.0) - 1.0,
        query_points,
        target_points,
        occluded,
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
    video_length=24,
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
        data_dir="/microtel/nfs/datasets/davis/",
        shuffle_files=shuffle_buffer_size is not None,
        **kwargs,
    )

    if repeat:
        ds = ds.repeat()

    def remove_short_videos(data):
        return tf.math.greater_equal(tf.shape(data["video"]["frames"])[0], video_length)

    ds = ds.filter(remove_short_videos)

    # Workaround to load trajectories out of file storage
    ds = ds.map(
        lambda data: tf.py_function(
            functools.partial(
                add_tracks,
                train_size=train_size,
                vflip=vflip,
                random_crop=random_crop,
                tracks_to_sample=tracks_to_sample,
                video_length=video_length
            ),
            [
                data["metadata"]["video_name"],
                data["video"]["frames"],
                data["video"]["segmentations"],
            ],
            [tf.float32, tf.float32, tf.float32, tf.bool],
        ),
        num_parallel_calls=num_parallel_point_extraction_calls,
    )

    ds = ds.map(
        lambda video, query_points, target_points, occluded: {
            "query_points": query_points,
            "target_points": target_points,
            "occluded": occluded,
            "video": video,
        },
        num_parallel_calls=num_parallel_point_extraction_calls,
    )

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    for bs in batch_dims[::-1]:
        ds = ds.batch(bs)

    return ds


if __name__ == "__main__":
    ds = create_point_tracking_dataset(shuffle_buffer_size=None)
    for example in ds:
        pass
        # print(example["target_points"].shape)
