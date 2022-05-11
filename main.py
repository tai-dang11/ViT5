print("Installing dependencies...")
%tensorflow_version 2.x
!pip install -q t5
!pip install transformers
!pip install datasets 
import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import t5

ON_CLOUD = True

import tensorflow_gcs_config
from google.colab import auth

auth.authenticate_user()

# if ON_CLOUD:
#     print("Setting up GCS access...")
#
#     # Set credentials for GCS reading/writing from Colab and TPU.
TPU_TOPOLOGY = "v2-8"
#     try:
#         tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU zdetection
#         TPU_ADDRESS = "10.15.69.19"
#         # TPU_ADDRESS = tpu.get_master()
#         print('Running on TPU:', TPU_ADDRESS)
#     except ValueError:
#         raise BaseException(
#             'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
#
#     tf.config.experimental_connect_to_host(TPU_ADDRESS)
#     tensorflow_gcs_config.configure_gcs_from_colab_auth()
#
# tf.disable_v2_behavior()

# Improve logging.
from contextlib import contextmanager
import logging as py_logging

if ON_CLOUD:
    tf.get_logger().propagate = False
    py_logging.root.setLevel('INFO')


@contextmanager
def tf_verbosity_level(level):
    og_level = tf.logging.get_verbosity()
    tf.logging.set_verbosity(level)
    yield
    tf.logging.set_verbosity(og_level)

# MODEL_NAME = "wiki_and_news_base"
length = 1024
MODEL_SIZE = "base"
# PRETRAINED_DIR = f'gs://t5_training/models/viT5_1024/{MODEL_SIZE}'
# PRETRAINED_DIR = "gs://t5_training/models/vie/viT5_large_1024/"
# MODEL_DIR = f"gs://t5_training/models/vie/WikiLingua_ViT5_{length}"
PRETRAINED_DIR = "gs://vien-translation/checkpoints/viT5_large_1024"
MODEL_DIR = f"gs://vien-translation/checkpoints/viT5_finetune/wikilingua_viT5_large"
# MODEL_DIR = f'gs://t5_training/models/vie/WikiLingua_viT5_large_1024'


# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (8, 128, 16),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

from t5 import models
tf.io.gfile.makedirs(MODEL_DIR)
# The models from paper are based on the Mesh Tensorflow Transformer.
model = models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 1024, "targets": 256},
    learning_rate_schedule=0.001,
    save_checkpoints_steps=500,
    keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
    iterations_per_loop=100,
)

tasks = [
         ['WikiLingua', "wikilingua"],
          # ['vietnews', "vietnews"],

         ]

!gsutil cp gs://best_vi_translation/raw/DATA_V2/V2_training_data_by_domains/* .
vocab = "gs://vien-translation/checkpoints/spm/cc100_envi_vocab.model"


import tensorflow.compat.v1 as tf
# checkpoints = [int(x.replace('.index', '').split('-')[-1]) for x in tf.io.gfile.glob(MODEL_DIR +'/*ckpt*.index')]
checkpoints = "gs://vien-translation/checkpoints/enviT5_finetune/mtet_envi_1000000enviT5"


from datasets import load_metric

predict_inputs_path = "en:" +. "gs://best_vi_translation/raw/DATA_V2/V2_training_data_by_domains/educational_content.en.txt"
predict_outputs_path ="vi:" + "gs://best_vi_translation/raw/DATA_V2/V2_training_data_by_domains/educational_content.vi"

with tf_verbosity_level('ERROR'):
      model.batch_size = 8  # Min size for small model on v2-8 with parallelism 1.
      model.predict(
          input_file=predict_inputs_path,
          output_file=predict_outputs_path,
          # Select the most probable output token at each step.
          vocabulary=t5.data.SentencePieceVocabulary(vocab),
          temperature=0,
          checkpoint_steps= -1,
      )

predictions = []
references = []
with open(f'../data/{eval}/{label_file}') as file:
  for line in file:
    references.append([f"{task[2:4]}: {line.strip()}"])
with open(prediction_files[-1]) as file:
  for line in file:
    predictions.append(line.strip())


print('DEBUG: few senctences of pred')
print(predictions[0:3])

print('DEBUG: few senctences of ref')
print(references[0:3])

metric = load_metric("sacrebleu", keep_in_memory=True)
result = metric.compute(predictions=predictions, references=references)
result = {"bleu": result["score"]}
print(result)