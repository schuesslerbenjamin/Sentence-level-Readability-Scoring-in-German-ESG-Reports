import torch
import transformers
import json
import sklearn
import sklearn.metrics
from tqdm import tqdm
import os
import numpy as np
import nltk
import random
from collections import defaultdict
import scipy.stats
import argparse

import ARA_model
from ARA_dataset import ARA_Dataset

torch._dynamo.config.cache_size_limit = 30  # otherwise some of the LLMs can crash
       
transformers.logging.disable_progress_bar() # removed to not clutter the output (e.g., in the cli mode)

class ARA_LLM_Model(ARA_model.ARA_Model):
    def __init__(self, model_id, system_message, user_message, shot_sample_file:str | None = None, x_shot:int = 0, feature_range=(0, 1), quantize:int | None = None):

        self.model_id = model_id
        self.system_message = system_message
        self.user_message = user_message
        self.feature_range = feature_range

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read the local.env file to get the HUGGINGFACE token
        self.access_token = None
        try:
            with open("local.env", "r") as env_file:
                for line in env_file:
                    if line.startswith("HUGGINGFACE_TOKEN"):
                        self.access_token = line.strip().split("=")[1]
                        break
        except:
            pass
        

        # Allow for quantizing. This is faster when an LLM is too large to fit in the VRAM of the GPU.
        if quantize is not None:
            if quantize == 4:
                quantize = transformers.BitsAndBytesConfig(load_in_4bit=True)
            elif quantize == 8:
                quantize = transformers.BitsAndBytesConfig(load_in_8bit=True)
            elif quantize == "None":
                quantize = None
            else:
                raise ValueError
            


        self.pipeline = transformers.pipeline(
            "text-generation",
            model = self.model_id,
            model_kwargs = {"torch_dtype": torch.bfloat16, "do_sample": False, "quantization_config": quantize},
            device_map = "auto",
            token = self.access_token
        )

        assert self.pipeline.tokenizer is not None
        
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
        ]

        if x_shot < 0:
            raise ValueError("x_shot must be a non-negative integer")
        if x_shot > 0 and shot_sample_file is None:
            raise ValueError("train_data must be provided if x_shot is greater than 0")

        self.x_shot = x_shot

        shot_samples = ARA_model.import_jsonl(shot_sample_file)
        shot_samples = ARA_Dataset(data=shot_samples, columns_to_keep=["text"], target_column="label")

        self.shot_samples = shot_samples
        self.shot_samples_file = shot_sample_file

    def add_x_shot(self):

        assert type(self.shot_samples) is ARA_Dataset

        x_shot_message = ""

        for i in range(self.x_shot):
            shot_index = random.choice(self.shot_samples.features.index.tolist())

            shot_sentence = self.shot_samples.sentences[shot_index]

            shot_label = self.shot_samples.target[shot_index]
            shot_label = ARA_model.scale_up(shot_label)  # Scale the label to the range of 1 to 4

            x_shot_message = x_shot_message + f"[Sentence] {shot_sentence} [Readability Score] {shot_label} \n "

        return x_shot_message

    def predict_readability_sentence(self, sentence, try_count=0) -> float:
        

        user_message = self.user_message


        # Add x-shot examples if specified
        if self.x_shot > 0:
            user_message = user_message + self.add_x_shot()

        user_message = user_message + "[Sentence] " + sentence

        # Construct the prompt for the LLM
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "[Readability Score] "}
        ]

        assert self.pipeline.tokenizer is not None, "Tokenizer is None"

        outputs = self.pipeline(
            messages,
            max_new_tokens=10,
            eos_token_id=self.terminators,
        )

        answer = outputs[0]["generated_text"][-1]["content"] # Extract the last generated text (includes "The readability of the sentence is:") # type: ignore
        score = answer.split(" ")[-1] # Only use the last "word" as the score (hopefully a number)

        # Handle wrong predictions
        retry_threshold = 5

        if "error" in answer:
            try_count += 1
            if try_count < retry_threshold:
                print("Error in prediction:\n\n", outputs)
                print(f"\n\nRetrying prediction... Attempt {try_count}")
                return self.predict_readability(sentence, try_count)
            else:
                print("Error in prediction:", answer)
                print("Max retries reached")
                return self.feature_range[0]  # Return the minimum / worst value of the feature range

        try:
            score = float(score)
        except ValueError:
            print("Failed to convert answer to float")
            try_count += 1
            if try_count < retry_threshold:
                print("Error in prediction:\n\n", outputs)
                print(f"\n\nRetrying prediction... Attempt {try_count}")
                return self.predict_readability(sentence, try_count)
            else:
                print("Error in prediction:\n\n", outputs)
                print(f"\n\nRetrying prediction... Attempt {try_count}")
                return self.feature_range[0]  # Return the minimum / worst value of the feature range
            
        # Scale the score to the feature range
        if self.feature_range != (1, 4):
            score = self.scale_prediction(score)
        
        return score
    
    def predict_readability(self, sentence, sent_tokenizing=False):
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
            score = self.predict_readability_sentence(sent)
            scores.append(score)
        
        return np.mean(scores)
        
    
    def scale_prediction(self, score:float) -> float:
        
        # scale down to 0 - 1
        score = (score - 1) / 3.0

        # scale up to the feature range
        score = score * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        return score
    
    def batch_evaluate(self, test_data:ARA_Dataset, file_path:str | None = None) -> float:
        """
        Evaluates the model on the test dataset.
        """

        progress_bar = tqdm(total=len(test_data), desc=f"Batch evaluation", unit="sentence")

        evaluations = defaultdict(list)
        for id, row in test_data.features.iterrows():

            assert isinstance(id, str)

            evaluations["ids"].append(id)
            
            sentence = test_data.sentences[id]
            evaluations["sentences"].append(sentence)

            prediction = self.predict_readability(sentence)
            evaluations["predictions"].append(prediction)
            
            evaluations["labels"].append(test_data.target[id])

            progress_bar.update(1)
        
        progress_bar.close()
        
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
        Saves the model to the specified path.
        """
        
        
        # Save the settings
        settings = {
            "model_id": self.model_id,
            "system_message": self.system_message,
            "user_message": self.user_message,
            "x_shot": self.x_shot,
            "feature_range": self.feature_range,
            "shot_sample_file": self.shot_samples_file
        }

        with open(f"{model_path}_settings.json", "w") as file:
            json.dump(settings, file)

    
    @classmethod
    def load_model(cls, model_path:str):
        """
        Loads the model from the specified path.
        """
        
        raise NotImplementedError("Instantiate the model instead of loading it here.")


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
    with open("ARA_LLM_model_settings.json", "r") as file:
        settings = json.load(file)

    parser = argparse.ArgumentParser(description="ARA LLM Model")

    parser.add_argument("--mode", type=str, help="Mode of this ARA Model", default="train", choices=["train", "eval", "cli", "inference-timing"])
    parser.add_argument("--model_type", type=str, help="Model Type", choices=["LR", "EN", "RIDGE", "LASSO", "XGBOOST"])
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--input_sentence", type=str, help="Input sentence to rate (only in cli mode)", default="")

    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = settings["ARA_model_path"]

    if args.mode == "train":
        train_data = ARA_model.import_jsonl(settings["train_data"])
        dev_data = ARA_model.import_jsonl(settings["dev_data"])
        eval_data = ARA_model.import_jsonl(settings["eval_data"])


        train_data = ARA_Dataset(data=train_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        dev_data = ARA_Dataset(data=dev_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        eval_data = ARA_Dataset(data=eval_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])

        model_id = settings["model_id"]

        # Initialize the model
        ARA_LLM_Model_instance = ARA_LLM_Model(
            model_id=model_id,
            system_message=settings["system_message"],
            user_message=settings["user_message"],
            shot_sample_file=settings["train_data"],
            x_shot=settings["x_shot"],
            quantize = settings["quantize"]
        )  

        print("Train")
        print(ARA_LLM_Model_instance.batch_evaluate(train_data, file_path="predictions/ARA_LLM_model_train.jsonl"))

        print("Dev")
        print(ARA_LLM_Model_instance.batch_evaluate(dev_data, file_path="predictions/ARA_LLM_model_dev.jsonl"))

        print("Eval")
        print(ARA_LLM_Model_instance.batch_evaluate(eval_data, file_path="predictions/ARA_LLM_model_eval.jsonl"))


    elif args.mode == "eval":

        model_id = settings["model_id"]

        # Initialize the model
        ARA_LLM_Model_instance = ARA_LLM_Model(
            model_id=model_id,
            system_message=settings["system_message"],
            user_message=settings["user_message"],
            shot_sample_file=settings["train_data"],
            x_shot=settings["x_shot"],
            quantize = settings["quantize"]
        )  

        # print("Train")
        # train_data = ARA_model.import_jsonl(settings["train_data"])
        # train_data = ARA_Dataset(data=train_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        # print(ARA_LLM_Model_instance.batch_evaluate(train_data, file_path="predictions/ARA_LLM_model_train.jsonl"))

        # print("Dev")
        # dev_data = ARA_model.import_jsonl(settings["dev_data"])
        # dev_data = ARA_Dataset(data=dev_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        # print(ARA_LLM_Model_instance.batch_evaluate(dev_data, file_path="predictions/ARA_LLM_model_dev.jsonl"))

        # print("Eval")
        eval_data = ARA_model.import_jsonl(settings["eval_data"])
        eval_data = ARA_Dataset(data=eval_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        print(ARA_LLM_Model_instance.batch_evaluate(eval_data, file_path="predictions/ARA_LLM_model_eval.jsonl"))
        
    elif args.mode == "cli":
        model_id = settings["model_id"]

        # Initialize the model
        ARA_LLM_Model_instance = ARA_LLM_Model(
            model_id=model_id,
            system_message=settings["system_message"],
            user_message=settings["user_message"],
            shot_sample_file=settings["train_data"],
            x_shot=settings["x_shot"],
            quantize = settings["quantize"]
        )  
        ARA_LLM_Model_instance.cli_mode(args.input_sentence, "ARA_LLM_model")
    
    elif args.mode == "inference-timing":
        
        model_id = settings["model_id"]

        # Initialize the model
        ARA_LLM_Model_instance = ARA_LLM_Model(
            model_id=model_id,
            system_message=settings["system_message"],
            user_message=settings["user_message"],
            shot_sample_file=settings["train_data"],
            x_shot=settings["x_shot"],
            quantize = settings["quantize"]
        )  

        test_sentences = ARA_model.import_jsonl(settings["eval_data"])
        test_dataset = ARA_Dataset(data=test_sentences, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])

        ARA_LLM_Model_instance.measure_inference_time(test_dataset)
