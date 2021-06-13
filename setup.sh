conda create -n aditya python=3.8
conda activate aditya
pip install -r requirements.txt

# preprocessing
python preprocess.py --in_file ../wizard_of_wikipedia/test_random_split.json --out_file temp/test_seen.jsonl
python preprocess.py --in_file ../wizard_of_wikipedia/test_topic_split.json --out_file temp/test_unseen.jsonl

# prepare BERT/GPT2 files
python bert_config.py --out_file pretrain-models/bert_base_uncased
python gpt2_config.py --out_file pretrain-models/gpt2


