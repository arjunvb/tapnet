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

"""Default config to train the TAPIR."""
from jaxline import base_config
from ml_collections import config_dict

from tapnet import tapir_model
from contrack_utils.consts import Datasets


# We define the experiment launch config in the same file as the experiment to
# keep things self-contained in a single file.
def get_config() -> config_dict.ConfigDict:
  """Return config object for training."""
  config = base_config.get_base_config()

  # Experiment config.
  config.training_steps = 102000

  # NOTE: duplicates not allowed.
  config.shared_module_names = ('tapir_model',)

  config.dataset_names = ('davis',)
  # Note: eval modes must always start with 'eval_'.
  config.eval_modes = (
      'eval_davis_points',
      'eval_sfm_davis_points',
  )
  config.checkpoint_dir = '/tmp/tapnet_training/'
  config.evaluate_every = 100

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              sweep_name='default_sweep',
              save_final_checkpoint_as_npy=True,
              # `enable_double_transpose` should always be false when using 1D.
              # For other D It is also completely untested and very unlikely
              # to work.
              optimizer=dict(
                  base_lr=1e-5,
                  max_norm=-1,  # < 0 to turn off.
                  weight_decay=1e-3,
                  schedule_type='cosine',
                  cosine_decay_kwargs=dict(
                      init_value=0.0,
                      warmup_steps=100,
                      end_value=0.0,
                  ),
                  optimizer='adam',
                  # Optimizer-specific kwargs.
                  adam_kwargs=dict(
                      b1=0.9,
                      b2=0.95,
                      eps=1e-8,
                  ),
              ),
              fast_variables=tuple(),
              shared_modules=dict(
                  shared_module_names=config.get_oneway_ref(
                      'shared_module_names',
                  ),
                  tapir_model_kwargs=dict(
                      bilinear_interp_with_depthwise_conv=True,
                      use_causal_conv=False,
                  ),
              ),
              datasets=dict(
                  dataset_names=config.get_oneway_ref('dataset_names'),
                  davis_kwargs=dict(
                        batch_dims=4,
                        shuffle_buffer_size=20,
                        train_size=tapir_model.TRAIN_SIZE[1:3],
                        data_dir="/microtel/nfs/datasets/davis/",
                        video_length=24,
                        trajectory_length=5,
                        mask_threshold=1.0,
                        tracks_to_sample=32,
                        split="train",
                        full_length=True,
                    )
              ),
              supervised_point_prediction_kwargs=dict(
                  input_key="davis",
                  prediction_algo='cost_volume_regressor',
                  model_key='tapir_model',
              ),
              checkpoint_dir=config.get_oneway_ref('checkpoint_dir'),
              evaluate_every=config.get_oneway_ref('evaluate_every'),
              eval_modes=config.get_oneway_ref('eval_modes'),
              # If true, run evaluate() on the experiment once before
              # you load a checkpoint.
              # This is useful for getting initial values of metrics
              # at random weights, or when debugging locally if you
              # do not have any train job running.
              davis_points_path=Datasets.TAPNET_DAVIS.value,
              jhmdb_path="",
              robotics_points_path=Datasets.TAPNET_ROBOTICS.value,
              davis_sfm_path=Datasets.SFM_DAVIS.value,
              training=dict(
                  # Note: to sweep n_training_steps, DO NOT sweep these
                  # fields directly. Instead sweep config.training_steps.
                  # Otherwise, decay/stopping logic
                  # is not guaranteed to be consistent.
                  n_training_steps=config.get_oneway_ref('training_steps'),
              ),
              eval_kwargs=dict(
                  video_length=None,
                  full_video=True,
                  num_samples=256,
                  trajectory_length=None,
              ),
              inference=dict(
                  input_video_path='',
                  output_video_path='',
                  resize_height=256,  # video height resized to before inference
                  resize_width=256,  # video width resized to before inference
                  num_points=20,  # number of random points to sample
              ),
          )
      )
  )

  # Set up where to store the resulting model.
  config.train_checkpoint_all_hosts = False
  config.save_checkpoint_interval = 10
  config.eval_initial_weights = True

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config
