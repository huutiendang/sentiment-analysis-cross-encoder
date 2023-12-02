import argparse
import json
import math

import pandas as pd
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
import csv
import numpy as np


def write_jsonl(path, data):
    with open(path, "w") as jsonl_file:
        for item in data:
            json_line = json.dumps(item)  # Convert dictionary to JSON string
            jsonl_file.write(json_line + '\n')  # Write JSON line to the file


def read_jsonl_file(file_path):
    """
    Reads a JSONL (JSON Lines) file and returns a list of parsed JSON objects.

    :param file_path: The path to the JSONL file.
    :return: A list of JSON objects.
    """
    with open(file_path, encoding="utf-8") as file:
        data_str = file.read()
        data_list = [json.loads(d) for d in data_str.split('\n') if d]
    return data_list


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', choices=['roberta-large', 'roberta-base'], required=True, help='model used')
    parser.add_argument('--train-file', type=str, required=True, help='path of jsonl file used to training dataset')
    parser.add_argument('--train-labeled', type=str, required=True, help='path of csv file used to training dataset')
    parser.add_argument('--train-augmented', type=str, required=True)
    parser.add_argument('--test-file', type=str, required=True, help='path of jsonl file to test dataset')
    parser.add_argument('--model-save-path', type=str, required=True, help='path to save checkpoints')
    parser.add_argument('--max-seq-len', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=15, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    opt = parser.parse_args()
    return opt


# train and save model
def train(model_name, train_labeled_file, train_augmented, train_file, batch_size, epoch, model_save_path):
    train_data_jl = read_jsonl_file(train_file)
    train_samples = []
    label2int = {"positive": 0, "negative": 1, "neutral": 2}
    for item in train_data_jl:
        for aspect in item["aspect_categories"]:
            label_id = label2int[aspect[2]]
            train_samples.append(InputExample(texts=[item['sentence'], aspect[1]], label=label_id))
    # if use_train_unlabel:
    train_labeled_file = pd.read_csv(train_labeled_file, sep=",")
    # print(train_labeled_file)
    train_augmented = pd.read_csv(train_augmented, sep=",")
    for _, row in train_labeled_file.iterrows():
        train_samples.append(InputExample(texts=[row['text'], row['aspect']], label=label2int[row['label']]))
    for _, row in train_augmented.iterrows():
        train_samples.append(InputExample(texts=[row['text'], row['aspect']], label=label2int[row['label']]))
    print(len(train_samples))
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    model = CrossEncoder(model_name, num_labels=len(label2int))
    warmup_steps = math.ceil(len(train_dataloader) * opt.epoch * 0.1)
    model.fit(train_dataloader=train_dataloader,
              epochs=epoch,
              warmup_steps=warmup_steps,
              output_path=model_save_path)
    # Save the trained model
    model.save(model_save_path)


def infer(model, test_file, save_file_name):
    test_data = read_jsonl_file(test_file)
    int2label = {0: "positive", 1: "negative", 2: "neutral"}
    sentence_combinations = []
    aspect_ids = []
    for item in test_data:
        for aspect in item["aspect_categories"]:
            sentence_combinations.append([item['sentence'], aspect[1]])
            aspect_ids.append(aspect[0])

    scores = model.predict(sentence_combinations)
    predited_labels = np.argmax(scores, axis=1)
    with open(save_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Label"])
        for item1, item2 in zip(aspect_ids, predited_labels):
            writer.writerow([item1, int2label[item2]])


if __name__ == '__main__':
    opt = arguments()
    # training and save
    train(
        model_name=opt.model_name,
        train_file=opt.train_file,
        batch_size=opt.batch_size,
        model_save_path=opt.model_save_path,
        train_labeled_file=opt.train_labeled,
        train_augmented=opt.train_augmented,
        epoch=opt.epoch,
    )
    # load model
    model = CrossEncoder(opt.model_save_path)
    save_test_path = f'data/ml/outputs/test_prediction_3_augmented_{opt.model_name}.csv'
    infer(model=model, test_file=opt.test_file, save_file_name=save_test_path)