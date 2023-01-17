"""
Pipeline class for correcting misspellings in text using a pre-trained token classification model and a fasttext model.

The pipeline tokenizes the input text, embeds each token using a fasttext model, and feeds the
embeddings to a transformer model to predict the correct spelling of each word. The corrected text
is returned as a string.
"""

from typing import Optional

import numpy as np
from gensim.models.fasttext import FastText
import torch
from transformers import AutoModelForTokenClassification


class FastTextForMisspellingPipeline:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        fasttext_model_path: str,
        vocab_file: str,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        A pipeline class for correcting misspellings in a sentence using pre-trained token classification and fastText models.

        Args:
            pretrained_model_name_or_path (str): The path or name of the pre-trained token classification model.
            fasttext_model_path (str): The path of the pre-trained fastText model.
            vocab_file (str): The path of the vocabulary file.
            device (Optional[torch.device], optional): The device to run the models on. Defaults to CUDA if available, otherwise CPU.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path).to(self.device)
        self.fasttext_model = FastText.load(fasttext_model_path)
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.id2word = {i: line.strip() for i, line in enumerate(f.readlines())}
        self.model.eval()

    def __call__(self, sentence: str) -> str:
        """
        Corrects the misspellings in the input sentence using the pre-trained models.

        Args:
            sentence (str): The sentence to be corrected.

        Returns:
            str: The corrected sentence.
        """
        words = sentence.split(" ")
        inputs_embeds = np.zeros((len(words), self.fasttext_model.vector_size), dtype=np.float32)
        for i, word in enumerate(words):
            inputs_embeds[i] = self.fasttext_model.wv[word]
        inputs_embeds = torch.tensor(inputs_embeds).unsqueeze(dim=0).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs_embeds=inputs_embeds)

        predict_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        return " ".join([self.id2word[idx] if idx != 0 else word for idx, word in zip(predict_ids, words)])


if __name__ == "__main__":
    pipeline = FastTextForMisspellingPipeline(
        pretrained_model_name_or_path="models/distilbert-base-uncased-chest-ct-misspelling-ft",
        fasttext_model_path="models/chest-ct-fasttext.model",
        vocab_file="program_data/vocab.txt",
    )
    print(pipeline("2. Mid lug emphysma ."))
    print(pipeline("# Atrophy of lebt kidney ."))
