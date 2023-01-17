"""
This script is to evaluate a typo correction model by loading in test data, creating
a Dataset object, loading a fastText model, formatting the data, creating a DataLoader,
initializing a token classification model and then evaluating the test data using this
model, computing the loss and storing the predictions and ground truth labels. These
predictions and ground-truth labels are then used to compute various evaluation metrics
such as accuracy, precision, recall and F1-score, which can be used to measure the
performance of the model on the test data.
"""

import ast
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForTokenClassification
from gensim.models.fasttext import FastText
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
    parser.add_argument("--test_file", type=str, required=True, help="The path to a file containing the test data.")
    parser.add_argument("--fasttext_model", type=str, required=True, help="The path to the fastText model file")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="The name or path of a pretrained model that the script should use.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="The number of examples to include in each training batch.")
    parser.add_argument("--num_workers", default=0, type=int, help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    args = parser.parse_args()

    WORDS = "words"
    TYPO_WORDS = "typo_words"
    LABELS = "labels"

    # ===== 載入資料 =====
    df = pd.read_csv(args.test_file)

    tqdm.pandas(desc="Convert data type...")
    df[WORDS] = df[WORDS].progress_apply(ast.literal_eval)
    df[TYPO_WORDS] = df[TYPO_WORDS].progress_apply(ast.literal_eval)
    df[LABELS] = df[LABELS].progress_apply(ast.literal_eval)

    dataset = Dataset.from_pandas(df)
    dataset.set_format(type="torch", columns=[TYPO_WORDS, LABELS])

    ft_model = FastText.load(args.fasttext_model)

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

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_name_or_path).to(device)

    losses = AverageMeter()
    ground_true = []
    predict = []
    for inputs_embeds, labels in tqdm(dataloader, desc="Evaluating ..."):
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
    print(f"Accuracy: {metrics:.4f}")


if __name__ == "__main__":
    main()
