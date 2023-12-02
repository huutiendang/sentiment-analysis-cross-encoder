import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import textattack
from baseline import read_jsonl_file


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained-model', required=True, help='trained model')
    parser.add_argument('--train-file', type=str, required=True, help='path of jsonl file used to training set')
    parser.add_argument('--checkpoint_interval', type=int, default=None)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = arguments()
    #load model to attack
    tokenizer = AutoTokenizer.from_pretrained(opt.trained_model)
    model = AutoModelForSequenceClassification.from_pretrained(opt.trained_model)
    # model.to(opt.device)
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

    # load data to attack
    train = opt.train_file
    train_data_jl = read_jsonl_file(train)
    data = []
    label2int = {"positive": 0, "negative": 1, "neutral": 2}
    for item in train_data_jl:
        for aspect in item["aspect_categories"]:
            data.append(((item['sentence'], aspect[1]), label2int[aspect[2]]))
    dataset = textattack.datasets.Dataset(data, input_columns=("premise", "hypothesis"))

    attack_args = textattack.AttackArgs(
        log_to_csv="log.csv",
        num_examples=-1,
        checkpoint_interval=opt.checkpoint_interval,
        disable_stdout=True
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
