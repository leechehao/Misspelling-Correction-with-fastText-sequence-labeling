"""
The main goal of this script is to train a token classification model using the transformer library
on a given dataset and save the best performing model to a specified directory. The script also uses
a pre-trained fasttext model, vocabulary file and pre-trained transformer model to preprocess the data,
train the model, and evaluate the performance of the model. The script also uses command line arguments
to customize the training process such as batch size, learning rate, number of optimization steps, number
of training epochs and others.
"""

import os
import ast
import time
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
from gensim.models.fasttext import FastText
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup, set_seed
from datasets import Dataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="The directory where the best performing model will be saved.")
    parser.add_argument("--train_file", type=str, required=True, help="The path to the training dataset file in CSV format.")
    parser.add_argument("--valid_file", type=str, required=True, help="The path to the validation dataset file in CSV format.")
    parser.add_argument("--vocab_file", type=str, required=True, help="The path to the vocabulary file in LineWord format. (one line = one word.)")
    parser.add_argument("--fasttext_model", type=str, required=True, help="The path to the fastText model file.")
    parser.add_argument("--log_file", default="train.log", type=str, help="The path to the log file.")
    parser.add_argument("--pretrained_model_name_or_path", default="distilbert-base-uncased", type=str, help="The name or path to a pre-trained transformer model to use for training.")
    parser.add_argument("--batch_size", default=16, type=int, help="The number of examples to include in each training batch.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate to use for training.")
    parser.add_argument("--accum_steps", default=1, type=int, help="The number of optimization steps to take before updating the model's parameters.")
    parser.add_argument("--epochs", default=10, type=int, help="The number of training epochs to run.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float, help="The ratio of total number of steps for the warm up part of training.")
    parser.add_argument("--max_norm", default=1.0, type=float, help="The maximum gradient norm to use for clipping gradients.")
    parser.add_argument("--num_workers", default=0, type=int, help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--seed", default=2330, type=int, help="The random seed to use for training.")
    args = parser.parse_args()

    WORDS = "words"
    TYPO_WORDS = "typo_words"
    LABELS = "labels"
    LENGTH = "length"

    # ===== 設定隨機種子 =====
    set_seed(args.seed)

    # ===== 載入詞彙表 =====
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        word2id = {line.strip(): i for i, line in enumerate(f.readlines())}

    # ===== log檔 =====
    log_file = open(args.log_file, "w", encoding="utf-8")

    # ===== 載入資料 =====
    df_train = pd.read_csv(args.train_file)
    df_valid = pd.read_csv(args.valid_file)

    # ===== 資料前處裡 =====
    tqdm.pandas(desc="Convert data type...")
    df_train[WORDS] = df_train[WORDS].progress_apply(ast.literal_eval)
    df_train[TYPO_WORDS] = df_train[TYPO_WORDS].progress_apply(ast.literal_eval)
    df_train[LABELS] = df_train[LABELS].progress_apply(ast.literal_eval)
    df_valid[WORDS] = df_valid[WORDS].progress_apply(ast.literal_eval)
    df_valid[TYPO_WORDS] = df_valid[TYPO_WORDS].progress_apply(ast.literal_eval)
    df_valid[LABELS] = df_valid[LABELS].progress_apply(ast.literal_eval)

    train_max_length = max(df_train[LENGTH])
    valid_max_length = max(df_valid[LENGTH])
    print(f"train_max_length: {train_max_length}")
    print(f"valid_max_length: {valid_max_length}")

    train_dataset = Dataset.from_pandas(df_train)
    valid_dataset = Dataset.from_pandas(df_valid)
    train_dataset.set_format(type="torch", columns=[TYPO_WORDS, LABELS])
    valid_dataset.set_format(type="torch", columns=[TYPO_WORDS, LABELS])

    def collate_fn(batch):
        labels = [example[LABELS] for example in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        inputs_embeds = []
        typo_words = [example[TYPO_WORDS] for example in batch]
        for sentence in typo_words:
            sentence_matrix = np.zeros((labels.shape[1], ft_model.vector_size))
            for i, word in enumerate(sentence):
                sentence_matrix[i] = ft_model.wv[word]
            inputs_embeds.append(sentence_matrix)

        inputs_embeds = np.array(inputs_embeds, dtype=np.float32)
        return torch.tensor(inputs_embeds), labels

    # ===== 動態padding =====
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    ft_model = FastText.load(args.fasttext_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_name_or_path, num_labels=len(word2id)).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    best_metrics = 0
    epochs = tqdm(range(args.epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        #################################
        ##########    Train    ##########
        #################################
        model.train()
        losses = AverageMeter()
        start_time = time.time()
        for step, (inputs_embeds, labels) in enumerate(tqdm(train_dataloader, desc="Training...", position=1)):
            batch_size = labels.size(0)

            outputs = model(inputs_embeds=inputs_embeds.to(device), labels=labels.to(device))
            loss = outputs.loss
            losses.update(loss.item(), batch_size)

            if args.accum_steps > 1:
                loss = loss / args.accum_steps
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

            if (step + 1) % args.accum_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if (step + 1) == 1 or (step + 1) % 2500 == 0 or (step + 1) == len(train_dataloader):
                epochs.write(
                    f"Epoch: [{epoch + 1}][{step + 1}/{len(train_dataloader)}] "
                    f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                    f"Grad: {grad_norm:.4f} "
                    f"LR: {scheduler.get_last_lr()[0]:.8f}",
                    file=log_file,
                )
                log_file.flush()
                os.fsync(log_file.fileno())
        duration = time.time() - start_time
        epochs.write(f"Training duration: {duration:.3f} sec", file=log_file)

        ####################################
        ##########    Evaluate    ##########
        ####################################
        losses = AverageMeter()
        model.eval()
        ground_true = []
        predict = []
        start_time = time.time()
        for step, (inputs_embeds, labels) in enumerate(tqdm(valid_dataloader, desc="Evaluating ...", position=2)):
            batch_size = labels.size(0)

            with torch.no_grad():
                outputs = model(inputs_embeds=inputs_embeds.to(device), labels=labels.to(device))
                loss = outputs.loss

            losses.update(loss.item(), batch_size)
            ground_true.extend([[idx for idx in seq_idx if idx != -100] for seq_idx in labels.cpu().numpy().tolist()])
            predict.extend(
                [
                    [pre_idx for (pre_idx, tar_idx) in zip(pre_seq, tar_seq) if tar_idx != -100]
                    for (pre_seq, tar_seq) in zip(torch.argmax(outputs.logits, dim=2).cpu().numpy().tolist(), labels)
                ]
            )
        metrics = sum([pre_seq == tar_seq for pre_seq, tar_seq in zip(predict, ground_true)]) / len(ground_true)

        duration = time.time() - start_time
        epochs.write(f"Evaluating Loss: {losses.avg:.4f}", file=log_file)
        epochs.write(f"Evaluation duration: {duration:.3f} sec", file=log_file)
        epochs.write(f"Accuracy: {metrics:.4f}", file=log_file)

        if metrics > best_metrics:
            model.save_pretrained(args.output_dir)
            best_metrics = metrics

    log_file.close()


if __name__ == "__main__":
    main()
