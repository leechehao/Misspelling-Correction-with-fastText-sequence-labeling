"""
Train a fastText model on a provided corpus file using gensim library, and save the trained model
to a specified file path.
"""

import argparse

from gensim.models import FastText


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str, required=True, help="Path to a corpus file in LineSentence format. (one line = one sentence. Words must be already preprocessed and separated by whitespace.)")
    parser.add_argument("--model_path", type=str, required=True, help="Store the model to this file path.")
    parser.add_argument("--vector_size", default=100, type=int, help="Dimensionality of the word vectors.")
    parser.add_argument("--min_count", default=5, type=int, help="The model ignores all words with total frequency lower than this.")
    parser.add_argument("--window", default=5, type=int, help="The maximum distance between the current and predicted word within a sentence.")
    parser.add_argument("--sg", default=0, type=int, choices=[0, 1], help="Training algorithm: skip-gram if sg=1, otherwise CBOW.")
    parser.add_argument("--max_vocab_size", default=None, type=int, help="Limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Set to None for no limit.")
    parser.add_argument("--min_n", default=1, type=int, help="Minimum length of char n-grams to be used for training word representations.")
    parser.add_argument("--max_n", default=6, type=int, help="Max length of char ngrams to be used for training word representations. Set max_n to be lesser than min_n to avoid char ngrams being used.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of iterations (epochs) over the corpus.")
    parser.add_argument("--seed", default=2330, type=int, help="Seed for the random number generator.")
    args = parser.parse_args()

    model = FastText(
        vector_size=args.vector_size,
        min_count=args.min_count,
        window=args.window,
        sg=args.sg,
        max_vocab_size=args.max_vocab_size,
        min_n=args.min_n,
        max_n=args.max_n,
        epochs=args.epochs,
        seed=args.seed,
    )
    model.build_vocab(corpus_file=args.corpus_file)
    model.train(
        corpus_file=args.corpus_file,
        total_words=model.corpus_total_words,
        epochs=model.epochs,
    )
    model.save(args.model_path)
    print(f"vocabulary size: {len(model.wv.key_to_index)}")
    # model.wv.similarity(word1, word2)
    # model.wv.most_similar(word1)


if __name__ == "__main__":
    main()
