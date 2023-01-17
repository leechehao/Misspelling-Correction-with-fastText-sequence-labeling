# Misspelling-Correction-with-fastText-sequence-labeling
## Train fastText Model
Train a fastText model on a provided corpus file using gensim library, and save the trained model to a specified file path.

### Usage
To train a fastText model, refer to the following usage information:
```
usage: 1_train_fasttext.py [-h] --corpus_file CORPUS_FILE \
                                --model_path MODEL_PATH \
                                [--vector_size VECTOR_SIZE] \
                                [--min_count MIN_COUNT] \
                                [--window WINDOW] \
                                [--sg SG] \
                                [--max_vocab_size MAX_VOCAB_SIZE] \
                                [--min_n MIN_N] \
                                [--max_n MAX_N] \
                                [--epochs EPOCHS] \
                                [--seed SEED]

required arguments:
  --corpus_file CORPUS_FILE
                        Path to a corpus file in LineSentence format. (one line = one sentence.
                        Words must be already preprocessed and separated by whitespace.)
  --model_path MODEL_PATH
                        Store the model to this file path.
                        
optional arguments:
  -h, --help            show this help message and exit
  --vector_size VECTOR_SIZE
                        Dimensionality of the word vectors.
  --min_count MIN_COUNT
                        The model ignores all words with total frequency lower than this.
  --window WINDOW       The maximum distance between the current and predicted word within a sentence.
  --sg SG               Training algorithm: skip-gram if sg=1, otherwise CBOW.
  --max_vocab_size MAX_VOCAB_SIZE
                        Limits the RAM during vocabulary building; if there are more unique words than
                        this, then prune the infrequent ones. Set to None for no limit.
  --min_n MIN_N         Minimum length of char n-grams to be used for training word representations.
  --max_n MAX_N         Max length of char ngrams to be used for training word representations. Set max_n
                        to be lesser than min_n to avoid char ngrams being used.
  --epochs EPOCHS       Number of iterations (epochs) over the corpus.
  --seed SEED           Seed for the random number generator.
```

### Arguments
+ **corpus_file** *(str)* ─ Path to a corpus file in LineSentence format. (one line = one sentence. Words must be already preprocessed and separated by whitespace.)
+ **model_path** *(str)* ─ Store the model to this file path.
+ **vector_size** *(int, optional, defaults to **`100`**)* ─ Dimensionality of the word vectors.
+ **min_count** *(int, optional, defaults to **`5`**)* ─ The model ignores all words with total frequency lower than this.
+ **window** *(int, optional, defaults to **`5`**)* ─ The maximum distance between the current and predicted word within a sentence.
+ **sg** *(int, optional, defaults to **`0`**)* ─ Training algorithm: skip-gram if sg=1, otherwise CBOW.
+ **max_vocab_size** *(int, optional, defaults to **`None`**)* ─ Limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Set to None for no limit.
+ **min_n** *(int, optional, defaults to **`1`**)* ─ Minimum length of char n-grams to be used for training word representations.
+ **max_n** *(int, optional, defaults to **`6`**)* ─ Max length of char ngrams to be used for training word representations. Set max_n to be lesser than min_n to avoid char ngrams being used.
+ **epochs** *(int, optional, defaults to **`10`**)* ─ Number of iterations (epochs) over the corpus.
+ **seed** *(int, optional, defaults to **`2330`**)* ─ Seed for the random number generator.

## Prepare Label Data
Generate variations of words, which could be used for data augmentation or error analysis.

### Usage
To generate variations of words, refer to the following usage information:
```
usage: 3_prepare_label_data.py [-h] --input_file INPUT_FILE \
                                    --output_file OUTPUT_FILE \
                                     --vocab_file VOCAB_FILE \
                                     [--stop_words_file STOP_WORDS_FILE] \
                                     [--typo_probability TYPO_PROBABILITY] \
                                     [--max_num_typo MAX_NUM_TYPO] \
                                     [--data_multiple DATA_MULTIPLE] \
                                     [--min_num_characters MIN_NUM_CHARACTERS] \
                                     [--num_typo_weights NUM_TYPO_WEIGHTS] \
                                     [--max_length MAX_LENGTH] \
                                     [--seed SEED]

required arguments:
  --input_file INPUT_FILE
                        The path to the input file in LineSentence format. (one line = one sentence.)
  --output_file OUTPUT_FILE
                        The path to the output CSV file. This file will contain the sentences with typos
                        added.
  --vocab_file VOCAB_FILE
                        The path to the vocabulary file in LineWord format. (one line = one word.)

optional arguments:
  -h, --help            show this help message and exit
  --stop_words_file STOP_WORDS_FILE
                        The path to a file containing stop words in LineWord format. These words will not
                        be modified by the script.
  --typo_probability TYPO_PROBABILITY
                        The probability of adding typos to the sentence.
  --max_num_typo MAX_NUM_TYPO
                        The maximum number of typos to add to the sentence.
  --data_multiple DATA_MULTIPLE
                        The multiple of the original data size that will be generated.
  --min_num_characters MIN_NUM_CHARACTERS
                        The minimum number of characters required for a word to be considered as will
                        generate a misspelling.
  --num_typo_weights NUM_TYPO_WEIGHTS
                        The weights for sampling the number of typos to add.
  --max_length MAX_LENGTH
                        The maximum length of the generated sentences.
  --seed SEED           The random seed to use.
```

### Arguments
+ **input_file** *(str)* ─ The path to the input file in LineSentence format. (one line = one sentence.)
+ **output_file** *(str)* ─ The path to the output CSV file. This file will contain the sentences with typos added.
+ **vocab_file** *(str)* ─ The path to the vocabulary file in LineWord format. (one line = one word.)
+ **stop_words_file** *(str, optional, defaults to **`None`**)* ─ The path to a file containing stop words in LineWord format. These words will not be modified by the script.
+ **typo_probability** *(float, optional, defaults to **`0.5`**)* ─ The probability of adding typos to the sentence.
+ **max_num_typo** *(int, optional, defaults to **`3`**)* ─ The maximum number of typos to add to the sentence.
+ **data_multiple** *(int, optional, defaults to **`4`**)* ─ The multiple of the original data size that will be generated.
+ **min_num_characters** *(int, optional, defaults to **`4`**)* ─ The minimum number of characters required for a word to be considered as will generate a misspelling.
+ **num_typo_weights** *(Sequence[int], optional, defaults to **`(5, 4, 1)`**)* ─ The weights for sampling the number of typos to add.
+ **max_length** *(int, optional, defaults to **`None`**)* ─ The maximum length of the generated sentences.
+ **seed** *(int, optional, defaults to **`2330`**)* ─ The random seed to use.

## Train Model
Train a token classification model using the transformer library on a given dataset for misspelling correction.
```
usage: 4_train.py [-h] --output_dir OUTPUT_DIR \
                       --train_file TRAIN_FILE \
                       --valid_file VALID_FILE \
                       --vocab_file VOCAB_FILE \
                       --fasttext_model FASTTEXT_MODEL \
                       [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH] \
                       [--batch_size BATCH_SIZE] \
                       [--learning_rate LEARNING_RATE] \
                       [--accum_steps ACCUM_STEPS] \
                       [--epochs EPOCHS] \
                       [--warmup_ratio WARMUP_RATIO] \
                       [--max_norm MAX_NORM] \
                       [--num_workers NUM_WORKERS] \
                       [--seed SEED]

required arguments:
  --output_dir OUTPUT_DIR
                        The directory where the best performing model will be saved.
  --train_file TRAIN_FILE
                        The path to the training dataset file in CSV format.
  --valid_file VALID_FILE
                        The path to the validation dataset file in CSV format.
  --vocab_file VOCAB_FILE
                        The path to the vocabulary file in LineWord format. (one line = one word.)
  --fasttext_model FASTTEXT_MODEL
                        The path to the fastText model file

optional arguments:
  -h, --help            show this help message and exit
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        The name or path to a pre-trained transformer model to use for training.
  --batch_size BATCH_SIZE
                        The number of examples to include in each training batch.
  --learning_rate LEARNING_RATE
                        The learning rate to use for training.
  --accum_steps ACCUM_STEPS
                        The number of optimization steps to take before updating the model's parameters.
  --epochs EPOCHS       The number of training epochs to run.
  --warmup_ratio WARMUP_RATIO
                        The ratio of total number of steps for the warm up part of training.
  --max_norm MAX_NORM   The maximum gradient norm to use for clipping gradients.
  --num_workers NUM_WORKERS
                        How many subprocesses to use for data loading. 0 means that the data will be
                        loaded in the main process.
  --seed SEED           The random seed to use for training.
```

### Arguments
+ **output_dir** *(str)* ─ The directory where the best performing model will be saved.
+ **train_file** *(str)* ─ The path to the training dataset file in CSV format.
+ **valid_file** *(str)* ─ The path to the validation dataset file in CSV format.
+ **vocab_file** *(str)* ─ The path to the vocabulary file in LineWord format. (one line = one word.)
+ **fasttext_model** *(str)* ─ The path to the fastText model file
+ **pretrained_model_name_or_path** *(str, optional, defaults to **`distilbert-base-uncased`**)* ─ The name or path to a pre-trained transformer model to use for training.
+ **batch_size** *(int, optional, defaults to **`16`**)* ─ The number of examples to include in each training batch.
+ **learning_rate** *(float, optional, defaults to **`1e-4`**)* ─ The learning rate to use for training.
+ **accum_steps** *(int, optional, defaults to **`1`**)* ─ The number of optimization steps to take before updating the model's parameters.
+ **epochs** *(int, optional, defaults to **`10`**)* ─ The number of training epochs to run.
+ **warmup_ratio** *(float, optional, defaults to **`0.0`**)* ─ The ratio of total number of steps for the warm up part of training.
+ **max_norm** *(float, optional, defaults to **`1.0`**)* ─ The maximum gradient norm to use for clipping gradients.
+ **num_workers** *(int, optional, defaults to **`0`**)* ─ How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
+ **seed** *(int, optional, defaults to **`2330`**)* ─ The random seed to use for training.

## Evaluate Model
Evaluate a typo correction model on a test dataset.

### Usage
To evaluate the model, refer to the following usage information:
```
usage: 5_test.py [-h] --test_file TEST_FILE \
                      --fasttext_model FASTTEXT_MODEL \
                      --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH \
                      [--eval_batch_size EVAL_BATCH_SIZE] \
                      [--num_workers NUM_WORKERS]

required arguments:
  --test_file TEST_FILE
                        The path to a file containing the test data.
  --fasttext_model FASTTEXT_MODEL
                        The path to the fastText model file
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        The name or path of a pretrained model that the script should use.

optional arguments:
  -h, --help            show this help message and exit
  --eval_batch_size EVAL_BATCH_SIZE
                        The number of examples to include in each training batch.
  --num_workers NUM_WORKERS
                        How many subprocesses to use for data loading. 0 means that the data will be
                        loaded in the main process.
```

### Arguments
+ **test_file** *(str)* ─ The path to a file containing the test data.
+ **fasttext_model** *(str)* ─ The path to the fastText model file
+ **pretrained_model_name_or_path** *(str)* ─ The name or path of a pretrained model that the script should use.
+ **eval_batch_size** *(int, optional, defaults to **`100`**)* ─ The number of examples to include in each training batch.
+ **num_workers** *(int, optional, defaults to **`0`**)* ─ How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.

## Inference Pipeline
A pipeline for correcting misspellings in a sentence using a pre-trained token classification model and a fasttext model.

### Usage
To use the **`FastTextForMisspellingPipeline`** class, create an instance of the class and call it with a sentence as an argument.
The returned value will be the corrected sentence.

```python
pipeline = FastTextForMisspellingPipeline(
    pretrained_model_name_or_path="path/to/pretrained/model",
    fasttext_model_path="path/to/fasttext/model",
    vocab_file="path/to/vocab/file",
)
corrected_sentence = pipeline("This sntence has misspellings.")
```