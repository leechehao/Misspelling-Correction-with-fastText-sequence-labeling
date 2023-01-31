python 4_train.py --output_dir models/distilbert-base-uncased-chest-ct-misspelling-ft \
                  --train_file program_data/train/ner.csv \
                  --valid_file program_data/valid/ner.csv \
                  --vocab_file program_data/vocab.txt \
                  --fasttext_model models/chest-ct-fasttext.model \
                  --epochs 30 \
                  --num_workers 12