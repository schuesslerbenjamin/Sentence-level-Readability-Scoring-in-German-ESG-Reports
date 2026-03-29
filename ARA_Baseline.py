import json
import numpy as np
import random
import pandas as pd
import sklearn.linear_model
import torch
import sklearn
import argparse
import pickle
from collections import defaultdict
import nltk

import ARA_model
import indices
from ARA_dataset import ARA_Dataset


class ARA_Baseline(ARA_model.ARA_Model):
    def __init__(self, model=None):

        if model:
            self.model = model
        else:
            self.model = sklearn.linear_model.LinearRegression()

    def calculate_features(self, sentences) -> pd.DataFrame:

        features = []

        for sentence in sentences:
            sentence_length = indices.get_word_count(sentence)
            features.append([sentence_length])

        features_df = pd.DataFrame(features, columns=["sentence_length"])

        return features_df

    def train(self, train_data: ARA_Dataset):

        features = self.calculate_features(train_data.sentences)
        self.model.fit(features, train_data.target)

    def predict_readability(self, sentence: str, sent_tokenizing = False):
        """
        Predicts the readability score for a sentence.
        Setting sent_tokenizing to True tries to split the input text into individual sentences. If successful, this function returns the average score. Otherwise, this function treats the input like a single sentence.
        """

        if sent_tokenizing:
            sentences = nltk.sent_tokenize(sentence, language='german')
        else:
            sentences = [sentence]

        scores = []
        for sent in sentences:
            sentence_length = indices.get_word_count(sentence)

            features = pd.DataFrame([[sentence_length]], columns=["sentence_length"])

            readability = self.model.predict(features) 
            score = float(readability[0])
            scores.append(score)

        return np.mean(scores)
    
    def batch_evaluate(self, data: ARA_Dataset, file_path: str | None = None, csv_path: str | None = None):

        evaluations = defaultdict(list)

        for id, features in data.features.iterrows():

            sentence = data.sentences[id]

            label = data.target[id]

            prediction = self.predict_readability(sentence)

            evaluations["ids"].append(id)
            evaluations["sentences"].append(sentence)
            evaluations["labels"].append(label)
            evaluations["predictions"].append(prediction)

        if file_path:
            self.save_evaluations(evaluations, file_path)

        self.analyze_predictions(evaluations["predictions"], evaluations["labels"], csv_path, row_index="ARA_Baseline")
        
    def save_model(self, model_path: str):

        # Save the model
        with open(f"{model_path}_model.pkl", 'wb') as file:
            pickle.dump(self.model, file)
        
    @classmethod
    def load_model(cls, model_path:str):
        """
        Loads the model from a specified path.
        """

        # Load the model
        with open(f"{model_path}_model.pkl", 'rb') as file:
            model = pickle.load(file)

        return cls(model=model)

 
if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    
    # read settings
    with open("ARA_Baseline_settings.json", "r") as file:
        settings = json.load(file)


    # Check if their were parameters provided via command line
    parser = argparse.ArgumentParser(description="ARA Baseline")

    parser.add_argument("--mode", type=str, help="Mode of this ARA Model", default="train", choices=["train", "eval", "cli", "inference-timing"])
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--input_sentence", type=str, help="Input sentence to rate (only in cli mode)")


    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = settings["model_path"]
    
    if args.mode == "train":
        train_data = ARA_model.import_jsonl(settings["train_data"])
        dev_data = ARA_model.import_jsonl(settings["dev_data"])
        eval_data = ARA_model.import_jsonl(settings["eval_data"])

        train_data = ARA_Dataset(data = train_data, columns_to_keep=["text"], target_column="label")
        dev_data = ARA_Dataset(data = dev_data, columns_to_keep=["text"], target_column="label")
        eval_data = ARA_Dataset(data = eval_data, columns_to_keep=["text"], target_column="label")


        ARA_Baseline_Instance = ARA_Baseline()
        ARA_Baseline_Instance.train(train_data)

        ARA_Baseline_Instance.save_model(model_path)

        print("Train data")
        ARA_Baseline_Instance.batch_evaluate(train_data, file_path="predictions/ARA_Baseline_train.jsonl")

        print("\nDev data")
        ARA_Baseline_Instance.batch_evaluate(dev_data, file_path="predictions/ARA_Baseline_dev.jsonl")

        print("\nEval data")
        ARA_Baseline_Instance.batch_evaluate(eval_data, file_path="predictions/ARA_Baseline_eval.jsonl")

    elif args.mode == "eval":
        ARA_Baseline_Instance = ARA_Baseline.load_model(model_path)

        # print("Train data")
        # train_data = ARA_model.import_jsonl(settings["train_data"])
        # train_data = ARA_Dataset(data = train_data, columns_to_keep=["text"], target_column="label")
        # ARA_Baseline_Instance.batch_evaluate(train_data, file_path="predictions/ARA_Baseline_train.jsonl")

        # print("\nDev data")        
        # dev_data = ARA_model.import_jsonl(settings["dev_data"])
        # dev_data = ARA_Dataset(data = dev_data, columns_to_keep=["text"], target_column="label")
        # ARA_Baseline_Instance.batch_evaluate(dev_data, file_path="predictions/ARA_Baseline_dev.jsonl")

        # print("\nEval data")
        eval_data = ARA_model.import_jsonl(settings["eval_data"])
        eval_data = ARA_Dataset(data = eval_data, columns_to_keep=["text"], target_column="label")
        ARA_Baseline_Instance.batch_evaluate(eval_data, file_path="predictions/ARA_Baseline_eval.jsonl")

    elif args.mode == "cli":
        ARA_Baseline_Instance = ARA_Baseline.load_model(model_path)

        ARA_Baseline_Instance.cli_mode(args.input_sentence, "ARA_Sentence_Length_Baseline")



    elif args.mode == "inference-timing":
        ARA_Baseline_Instance = ARA_Baseline.load_model(model_path)

        test_sentences = ARA_model.import_jsonl(settings["eval_data"])
        test_dataset = ARA_Dataset(data=test_sentences, columns_to_keep=settings["features"], target_column=settings["target"])

        ARA_Baseline_Instance.measure_inference_time(test_dataset)