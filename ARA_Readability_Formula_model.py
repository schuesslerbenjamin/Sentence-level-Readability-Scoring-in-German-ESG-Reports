import json
import numpy as np
import random
import pandas as pd
import textstat
import sklearn.linear_model
import torch
import sklearn
import sklearn.metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats
import xgboost
import argparse
import nltk
import pickle
from collections import defaultdict

import ARA_model
import indices
from ARA_dataset import ARA_Dataset


class ARA_Readability_Formula_Dataset():
    def __init__(self, r_dataset: ARA_Dataset, feature_names: list = [], do_scale = False, scaler=None):

        self.feature_names = feature_names

        # default case when loading from a readability dataset
        if isinstance(r_dataset, ARA_Dataset):

            for i in r_dataset.sentences:
                assert isinstance(i, str)
        
            features = [self.calculate_features(sent, feature_names) for sent in r_dataset.sentences] # type: ignore
            features = pd.DataFrame(features, index=r_dataset.features.index)

            if do_scale:
                features, self.scaler = self.scale_features(features, scaler)
            else:
                self.scaler = scaler
            
            self.features = features
            self.labels = r_dataset.target
            self.sentences = r_dataset.sentences

        else:
            print(type(r_dataset))
            raise ValueError("r_dataset must be a ReadabilityDataset or a string.")
        
        assert isinstance(self.features, pd.DataFrame), "Features must be a pandas DataFrame"
        assert isinstance(self.labels, pd.Series), f"Labels must be a pandas Series"

    @staticmethod
    def calculate_features(sentence: str, feature_names: list, id = None) -> pd.Series:
        """
        Calculates the features for a given sentence.
        """

        features_to_calculate = feature_names.copy()
        
        try:
            features_to_calculate.remove("text")
        except:
            pass

        textstat.set_lang("de") # type: ignore

        features = defaultdict()
        if "flesch_reading_ease" in features_to_calculate:
            flesch_reading_ease = textstat.flesch_reading_ease(sentence) # type: ignore
            features["flesch_reading_ease"] = flesch_reading_ease
            features_to_calculate.remove("flesch_reading_ease")

        if "1_wiener_sachtextformel" in features_to_calculate:
            wiener_sachtextformel = textstat.wiener_sachtextformel(sentence, 1) # type: ignore
            features["1_wiener_sachtextformel"] = wiener_sachtextformel
            features_to_calculate.remove("1_wiener_sachtextformel")

        if "polysyllable_proportion" in features_to_calculate:
            polysyllabic_proportion = indices.proportion_of_polysyllabic_words(sentence)
            features["polysyllable_proportion"] = polysyllabic_proportion
            features_to_calculate.remove("polysyllable_proportion")

        if "hkps" in features_to_calculate:
            hkps = indices.hohenheimer_readability_index_politics(sentence, language="german")
            features["hkps"] = hkps
            features_to_calculate.remove("hkps")

        if "lix" in features_to_calculate:
            lix = indices.lix_score(sentence, language="german")
            features["lix"] = lix
            features_to_calculate.remove("lix")
        
        if features_to_calculate:
            raise ValueError(f"Features {features_to_calculate} are not supported.")
        
        features = pd.Series(features, name = id)
        

        return features
    
    def get_scaler(self):
        """
        Returns the scaler.
        """
        return self.scaler
    
    def scale_features(self, features: np.ndarray | pd.DataFrame, scaler: MinMaxScaler | None = None) -> tuple[pd.DataFrame, MinMaxScaler]:
        """
        Scales the given DataFrame using MinMaxScaler.
        """

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_features = scaler.fit_transform(features)
        else:
            scaled_features = scaler.transform(features)
        
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features)

        scaled_features = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

        return scaled_features, scaler


class ARA_Readability_Formula_Model(ARA_model.ARA_Model):
    def __init__(self, scaler=None, feature_names:list = [], model_type:str = "LR", model=None):

        # If a model is provided, use it
        if model:
            self.model = model
        else:
            if model_type == "LR":
                self.model = sklearn.linear_model.LinearRegression()
            elif model_type == "EN":
                self.model = sklearn.linear_model.ElasticNet()
            elif model_type == "RIDGE":
                self.model = sklearn.linear_model.Ridge()
            elif model_type == "LASSO":
                self.model = sklearn.linear_model.Lasso()
            elif model_type == "XGBOOST":
                self.model = xgboost.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth = 5, random_state=42)
            else:
                raise NotImplementedError
        
        self.model_type = model_type

        self.scaler = scaler
        self.feature_names = feature_names

    def train_model(self, train_data:ARA_Readability_Formula_Dataset):
        """
        Trains the regression model on the training data.
        """

        self.model.fit(train_data.features, train_data.labels)
        
        if self.model_type == "LR":
            print(f"Coefficients: {self.model.coef_}")
            print(f"Intercept: {self.model.intercept_}")
        
        return self.model
    
    def sentence_to_features(self, sentence: str) -> pd.Series:
        cols = self.feature_names
        try:
            cols.remove("text")  # Remove text column if it exists
        except:
            pass
    
        features = ARA_Readability_Formula_Dataset.calculate_features(sentence, self.feature_names)

        # Scale if there is a scaler
        if self.scaler:
            assert isinstance(self.scaler, MinMaxScaler)
            assert isinstance(features, pd.Series)

            id = features.name

             
            features = pd.DataFrame([features])
            features = self.scaler.transform(features) # scale

            features = pd.DataFrame(features, columns=cols, index=[id]) 

        else:
            features = pd.DataFrame([features], columns=cols)

        return features

    
    def predict_readability(self, input: str | pd.Series, sent_tokenizing = False) -> float:
        """
        Predicts the readability score for a sentence represented by either a string or a list of features.
        Setting sent_tokenizing to True tries to split the input text into individual sentences. If successful, this function returns the average score. Otherwise, this function treats the input like a single sentence.
        """

        cols = self.feature_names
        try:
            cols.remove("text")  # Remove text column if it exists
        except:
            pass
        
        # During inference, when we have a sentence as a string
        if isinstance(input, str):
            if sent_tokenizing:
                sentences = nltk.sent_tokenize(input, language='german')
            else:
                sentences = [input]
            
            scores = []
            for sent in sentences:

                features = self.sentence_to_features(sent)

                # Predict readability score
                readability = self.model.predict(features)

                readability = float(readability[0])

                scores.append(readability)
                
            return np.mean(scores)
        

        # When we have the features during the evaluation
        elif isinstance(input, pd.Series):
            features = pd.DataFrame([input], columns=cols)

             # Predict readability score
            readability = self.model.predict(features)

            readability = float(readability[0])

            return readability


        else:
            raise ValueError("Features must be a string or a pd.Series.")
            
    
    def batch_evaluate(self, test_data:ARA_Readability_Formula_Dataset, file_path:str | None = None):
        """
        Evaluates the model on the test dataset.
        """

        assert isinstance(test_data.features, pd.DataFrame), "Features must be a pandas DataFrame"
        assert isinstance(test_data.labels, pd.Series), f"Labels must be a pandas Series"

        evaluations = defaultdict(list)

        for id, features in test_data.features.iterrows():
            assert isinstance(id, str), f"Index {id} is not a string."

            evaluations["ids"].append(id)

            label = test_data.labels[id]
            evaluations["labels"].append(label)

            sentence = test_data.sentences[id]
            evaluations["sentences"].append(sentence)

            prediction = self.predict_readability(features)
            evaluations["predictions"].append(prediction)

        if file_path:
            self.save_evaluations(evaluations, file_path)

        mse = sklearn.metrics.mean_squared_error(evaluations["labels"], evaluations["predictions"])
        print(f"Mean Squared Error: {mse}")

        print(f"RMSE; {np.sqrt(mse)}")

        mae = sklearn.metrics.mean_absolute_error(evaluations["labels"], evaluations["predictions"])
        print(f"Mean Absolute Error: {mae}")

        kendall_tau, p_value = scipy.stats.kendalltau(evaluations["labels"], evaluations["predictions"], variant='b')
        print(f"Kendall Tau: {kendall_tau} (p-value: {p_value})")

        return mse

    def save_model(self, model_path:str):
        """
        Saves the model to a specified path.
        """

        # Save the model
        with open(f"{model_path}_model.pkl", 'wb') as file:
            pickle.dump(self.model, file)

        # Save the scaler
        with open(f"{model_path}_scaler.pkl", 'wb') as file:
            pickle.dump(self.scaler, file)

        settings = {
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            
        }

        with open(f'{model_path}_settings.json', 'w') as file:
            json.dump(settings, file)
    
    @classmethod
    def load_model(cls, model_path:str):
        """
        Loads the model from a specified path.
        """

        # Load the model
        with open(f"{model_path}_model.pkl", 'rb') as file:
            model = pickle.load(file)

        # Load the scaler
        with open(f"{model_path}_scaler.pkl", 'rb') as file:
            scaler = pickle.load(file)

        # Load settings
        with open(f'{model_path}_settings.json', 'r') as file:
            settings = json.load(file)

        return cls(scaler=scaler, feature_names=settings["feature_names"], model_type=settings["model_type"], model=model)


if __name__ == "__main__":

    nltk.download("punkt_tab", quiet = True)

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
    with open("ARA_Readability_Formula_model_settings.json", "r") as file:
        settings = json.load(file)


    # Check if their were parameters provided via command line
    parser = argparse.ArgumentParser(description="ARA RF Model")

    parser.add_argument("--mode", type=str, help="Mode of this ARA Model", default="train", choices=["train", "eval", "cli", "inference-timing"])
    parser.add_argument("--model_type", type=str, help="Model Type", choices=["LR", "EN", "RIDGE", "LASSO", "XGBOOST"])
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--input_sentence", type=str, help="Input sentence to rate (only in cli mode)")


    args = parser.parse_args()
    
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = settings["model_type"]

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = settings["model_path"]

    if args.mode == "train":
        train_data = ARA_model.import_jsonl(settings["train_data"])
        dev_data = ARA_model.import_jsonl(settings["dev_data"])
        eval_data = ARA_model.import_jsonl(settings["eval_data"])

        features = settings["features"]
        label = settings["target"]

        train_data = ARA_Dataset(data = train_data, columns_to_keep=features, target_column=label)
        dev_data = ARA_Dataset(data = dev_data, columns_to_keep=features, target_column=label)
        eval_data = ARA_Dataset(data = eval_data, columns_to_keep=features, target_column=label)

        train_data = ARA_Readability_Formula_Dataset(r_dataset=train_data, feature_names=features, do_scale=settings["do_scale"])
        dev_data = ARA_Readability_Formula_Dataset(r_dataset=dev_data, feature_names=features, do_scale=settings["do_scale"], scaler=train_data.get_scaler())
        eval_data = ARA_Readability_Formula_Dataset(r_dataset=eval_data, feature_names=features, do_scale=settings["do_scale"], scaler=train_data.get_scaler())

        ARA_RF_model = ARA_Readability_Formula_Model(
            scaler=train_data.get_scaler(),
            feature_names=settings["features"],
            model_type=model_type
        )
        ARA_RF_model.train_model(train_data)

        ARA_RF_model.save_model(model_path)

        print("Train data")
        ARA_RF_model.batch_evaluate(train_data, file_path="predictions/ARA_RF_model_train.jsonl")

        print("\nDev data")
        ARA_RF_model.batch_evaluate(dev_data, file_path="predictions/ARA_RF_model_dev.jsonl")

        print("\nEval data")
        ARA_RF_model.batch_evaluate(eval_data, file_path="predictions/ARA_RF_model_eval.jsonl")

    elif args.mode == "eval":

        ARA_RF_model = ARA_Readability_Formula_Model.load_model(model_path)

        features = settings["features"]
        label = settings["target"]

        # print("Train data")
        train_data = ARA_model.import_jsonl(settings["train_data"])
        train_data = ARA_Dataset(data = train_data, columns_to_keep=features, target_column=label)
        train_data = ARA_Readability_Formula_Dataset(r_dataset=train_data, feature_names=features, do_scale=settings["do_scale"])
        # ARA_RF_model.batch_evaluate(train_data, file_path="predictions/ARA_RF_model_train.jsonl")

        # print("\nDev data")
        # dev_data = ARA_model.import_jsonl(settings["dev_data"])
        # dev_data = ARA_Dataset(data = dev_data, columns_to_keep=features, target_column=label)
        # dev_data = ARA_Readability_Formula_Dataset(r_dataset=dev_data, feature_names=features, do_scale=settings["do_scale"], scaler=train_data.get_scaler())
        # ARA_RF_model.batch_evaluate(dev_data, file_path="predictions/ARA_RF_model_dev.jsonl")

        # print("\nEval data")
        eval_data = ARA_model.import_jsonl(settings["eval_data"])
        eval_data = ARA_Dataset(data = eval_data, columns_to_keep=features, target_column=label)
        eval_data = ARA_Readability_Formula_Dataset(r_dataset=eval_data, feature_names=features, do_scale=settings["do_scale"], scaler=train_data.get_scaler())
        ARA_RF_model.batch_evaluate(eval_data, file_path="predictions/ARA_RF_model_eval.jsonl")


    elif args.mode == "cli":
        # Load the model
        ARA_RF_model = ARA_Readability_Formula_Model.load_model(model_path)

        ARA_RF_model.cli_mode(args.input_sentence, "ARA_RF_model")

    elif args.mode == "inference-timing":
        # Load the model
        ARA_RF_model = ARA_Readability_Formula_Model.load_model(model_path)

        test_sentences = ARA_model.import_jsonl(settings["eval_data"])
        test_dataset = ARA_Dataset(data=test_sentences, columns_to_keep=settings["features"], target_column=settings["target"])

        ARA_RF_model.measure_inference_time(test_dataset)

