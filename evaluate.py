import argparse
import os.path

import numpy as np
import csv

import tqdm
from transformers import AutoModelForSequenceClassification
from baseline import read_jsonl_file
from sentence_transformers import CrossEncoder
import torch


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, choices=['baseline', 'augmented', '3_augmented'])
    parser.add_argument('--model-name', required=True, choices=['roberta_base', 'roberta_large'])
    parser.add_argument('--trained-model', required=True, help='trained model')
    parser.add_argument('--test-file', type=str, required=True, help='path of jsonl file used to training set')
    parser.add_argument('--save-test-path', type=str, default='data/ml/outputs')
    opt = parser.parse_args()
    return opt


def prediction(model, test_file, save_test_path, type, model_name):
    test_data = read_jsonl_file(test_file)
    int2label = {0: "positive", 1: "negative", 2: "neutral"}
    test_sentence_combinations, test_aspect_ids = [], []

    for item in test_data:
        for aspect in item["aspect_categories"]:
            test_sentence_combinations.append([item['sentence'], aspect[1]])
            test_aspect_ids.append(aspect[0])

    test_scores = model.predict(test_sentence_combinations)
    test_predited_labels = np.argmax(test_scores, axis=1)
    save_file_name = "{}/test_prediction_{}_{}.csv".format(save_test_path, type, model_name)
    with open(save_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Label"])
        for item1, item2 in zip(tqdm.tqdm(test_aspect_ids), test_predited_labels):
            writer.writerow([item1, int2label[item2]])


if __name__ == '__main__':
    opt = arguments()
    # loaded_model = CrossEncoder(model_save_path)
    print("evaluation with model cross_encoder_{}_{}".format(opt.type, opt.model_name))
    model = CrossEncoder(opt.trained_model)
    # model.eval()
    prediction(
        model=model,
        test_file=opt.test_file,
        save_test_path=opt.save_test_path,
        type=opt.type,
        model_name=opt.model_name
    )
