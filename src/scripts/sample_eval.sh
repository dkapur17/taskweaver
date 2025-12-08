dataset_name="stanfordnlp/snli"
dataset_path="stanfordnlp_snli"

python evaluate.py --model_type lora --model_path _models/lora/EleutherAI_pythia-70m/"$dataset_path" --datasets "$dataset_name" --split "test[:10]"
python evaluate.py --model_type hypernet --model_path _models/hypernet/EleutherAI_pythia-70m/mix_8_datasets/ --datasets "$dataset_name" --split "test"