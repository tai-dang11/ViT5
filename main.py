import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import tensorflow.compat.v1 as tf
import t5
import tensorflow_gcs_config


parser = argparse.ArgumentParser(description='Finetunning ViT5')
parser.add_argument('-tpu', dest='tpu', type=str, help='tpu address', default='0.0.0.0')
parser.add_argument('-task', dest='task', type=str, help='En to Vi(envi) or Vi to En(vien) task', default='envi')
parser.add_argument('-eval', dest='eval', type=str, help='Eval test set', default='tst')
parser.add_argument('-filtered', dest='filtered', type=str, help='filtered or full', default='total')
parser.add_argument('-lr', dest='lr', type=float, help='learning rate', default=0.001)
parser.add_argument('-steps', dest='steps', type=int, help='tpu address', default=16266)
args = parser.parse_args()


TPU_TOPOLOGY = "v2-8"

tf.disable_v2_behavior()

# Improve logging.
from contextlib import contextmanager

@contextmanager
def tf_verbosity_level(level):
    og_level = tf.logging.get_verbosity()
    tf.logging.set_verbosity(level)
    yield
    tf.logging.set_verbosity(og_level)

task = 'wikilingua'
vocab = "gs://t5_training/models/spm/vie/wiki_vietnamese_vocab.model"

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
    # tpu=TPU_ADDRESS,
    tpu = "grpc://10.15.69.19",
    tpu_topology= "v2-8",
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

# !gsutil cp gs://best_vi_translation/raw/DATA_V2/V2_training_data_by_domains/* .
vocab = "gs://vien-translation/checkpoints/spm/cc100_envi_vocab.model"

import tensorflow.compat.v1 as tf
# checkpoints = [int(x.replace('.index', '').split('-')[-1]) for x in tf.io.gfile.glob(MODEL_DIR +'/*ckpt*.index')]
checkpoints = "gs://vien-translation/checkpoints/enviT5_finetune/mtet_envi_1000000enviT5"

from datasets import load_metric

eval = args.eval
if eval == 'tst':
    input_file = f'tst2013.{task[0:2]}.unfix'
    output_file = f'{task}_predict_output.txt'
    label_file = f"tst2013.{task[2:4]}.unfix"
elif eval == 'phomt':
    input_file = f'test.{task[0:2]}'
    output_file = f'{task}_predict_output.txt'
    label_file = f'test.{task[2:4]}'

with open('predict_input.txt', 'w') as out:
    for line in open(f'../data/{eval}/{input_file}'):
        out.write(f"{task[0:2]}: {line}")

predict_inputs_path = "gs://best_vi_translation/raw/DATA_V2/V2_training_data_by_domains/software.en.txt"
predict_outputs_path ="gs://best_vi_translation/raw/DATA_V2/V2_training_data_by_domains/software.vi.txt"

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

prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))

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