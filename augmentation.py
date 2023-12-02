import argparse
import nlpaug.augmenter.word as naw
import pandas as pd
import tqdm

import nlpaug.augmenter.sentence as nas
from baseline import read_jsonl_file

bert_aug = naw.ContextualWordEmbsAug(
    model_path='roberta-base', action="substitute")
syn_aug = naw.SynonymAug(aug_src='wordnet')
t5_aug = nas.AbstSummAug(model_path='t5-base')


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--train-file', type=str, required=True, help='path of jsonl file used to training set')
    parser.add_argument('--train-labeled', type=str, required=True, help='path of csv file to labeled '
                                                                            'training set')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    #
    opt = arguments()
    train_data_jl = read_jsonl_file(opt.train_file)
    df_train_labeled = pd.read_csv(opt.train_labeled, sep=",")
    train_sentences, train_aspects, train_labels = [], [], []
    label2int = {"positive": 0, "negative": 1, "neutral": 2}
    for item in train_data_jl:
        for aspect in item["aspect_categories"]:
            train_sentences.append(item['sentence'])
            train_aspects.append(aspect[1])
            train_labels.append(aspect[2])
    # df_train = pd.DataFrame({"text": train_sentences, "aspect": train_aspects, "label": train_labels})

    text = train_sentences + list(df_train_labeled.text)
    label = train_labels + list(df_train_labeled.label)
    aspect = train_aspects + list(df_train_labeled.aspect)

    augmented_texts, augmented_labels, augmented_aspects = [], [], []
    for ag_text, aspect, label in zip(tqdm.tqdm(text), aspect, label):
        augmentations = [syn_aug.augment, bert_aug.augment, t5_aug.augment]
        augmented_texts.extend(aug(ag_text)[0] for aug in augmentations)
        augmented_labels.extend([label] * len(augmentations))
        augmented_aspects.extend([aspect] * len(augmentations))
    augmentation = pd.DataFrame({"text": augmented_texts, "aspect": augmented_aspects, "label": augmented_labels})
    augmentation.to_csv("data/ml/processed/augmentation_{}.csv".format(opt.model_name), index=False, sep=",")
