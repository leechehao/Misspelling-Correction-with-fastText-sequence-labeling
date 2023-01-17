import pickle


with open("program_data/train/train.cor") as f:
    train_sentences = [line.strip() for line in f.readlines()]

with open("program_data/valid/valid.cor") as f:
    valid_sentences = [line.strip() for line in f.readlines()]

with open("program_data/test/test.cor") as f:
    test_sentences = [line.strip() for line in f.readlines()]

with open("program_data/vocab_UMLS.pickle", "rb") as f:
    vocab_umls = pickle.load(f)
print(f"UMLS vocabulary size: {len(vocab_umls)}")

vocab = set()
for sentences in [train_sentences, valid_sentences, test_sentences]:
    for sent in sentences:
        words = sent.lower().split(" ")
        for word in words:
            if word.encode().isalpha() and word in vocab_umls:
                vocab.add(word)
print(f"vocabulary size: {len(vocab)}")

with open("program_data/vocab.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(vocab))
