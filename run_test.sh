python 5_test.py --test_file program_data/test/ner.csv \
                 --fasttext_model models/chest-ct-fasttext.model \
                 --pretrained_model_name_or_path models/distilbert-base-uncased-chest-ct-misspelling-ft \
                 --num_workers 12