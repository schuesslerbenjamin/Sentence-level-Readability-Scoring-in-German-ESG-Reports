import json
import numpy as np
import random
import torch
from collections import defaultdict
import os
import transformers
import sklearn
import sklearn.metrics
from tqdm import tqdm
import argparse
import scipy.stats
import nltk

import ARA_model
from ARA_dataset import ARA_Dataset


LOSS_FUNCTION = "MSE" # MSE, RMSE, MAE


class CustomTrainer(transformers.Trainer):
    """
    This class allows us to choose and implement different loss functions.
    Other than that the code is similar to the transformers library.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = 1):

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()

        # Ensure labels and logits are the same shape
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        labels = labels.float() # for regression

        if LOSS_FUNCTION == "MSE":
            loss = torch.nn.functional.mse_loss(logits, labels)
        elif LOSS_FUNCTION == "RMSE":
            loss = torch.sqrt(torch.nn.functional.mse_loss(logits, labels))
        elif LOSS_FUNCTION == "MAE":
            loss = torch.nn.functional.l1_loss(logits, labels)
        else:
            raise ValueError
        
        return (loss, outputs) if return_outputs else loss


class ARA_Transformer_Model(ARA_model.ARA_Model):
    def __init__(self, model_checkpoint: str, model = None, eval=False):

        if model is None:
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1) # num labels = 1 for regression
        else:
            self.model = model
    
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True) 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if eval:
            self.model.eval()

    def load_model(self, model_path: str):
        """
        Load the model from the specified path.
        """
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True)

        self.model.eval()
        self.model.to(device)
    
    def save_model(self, model_path: str):
        """
        Save the model to the specified path.
        """
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    
    def disable_gradient(self, layers=None, debug = False):
        """
        Disable gradient calculation for the specified layers. "all" disables gradients for all layers. None disables gradients for no layers.
        """

        if layers is None:
            return self.model
        
        elif layers == "all":
            for param in self.model.parameters():
                param.requires_grad = False

        elif isinstance(layers, list):
            for name, param in self.model.named_parameters():
                for layer in layers:
                    if name.startswith(layer):
                        param.requires_grad = False
                        break
        else:
            raise ValueError("Invalid layers argument. Must be 'all', None, or a list of layer names.")
        
        if debug:
            print("Layers with gradient calculation disabled:")
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    print(name, param.requires_grad)
            
            print("\nLayers with gradient calculation enabled:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.requires_grad)
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        predictions = predictions[:, 0]

        mse = sklearn.metrics.mean_squared_error(labels, predictions)
        return {
            'MSE': mse,
	}

    def train_model(self, model_name, train_data_list, test_data_list, batch_size, num_epochs, learning_rate, weight_decay, checkpoint_dir, layers_to_disable_gradient=None, debug = False):
        if debug:
            print("Training model with the following parameters:")
            print(f"Model name: {model_name}")
            print(f"Batch size: {batch_size}")
            print(f"Number of epochs: {num_epochs}")
            print(f"Learning rate: {learning_rate}")
            print(f"Weight decay: {weight_decay}")
            print(f"Layers to disable gradient: {layers_to_disable_gradient}")
            print("\n\n")


        self.model.to(device)
        self.disable_gradient(layers=layers_to_disable_gradient)


        args = transformers.TrainingArguments(
            output_dir=f"{checkpoint_dir}/{model_name}/{batch_size}-{num_epochs}-{learning_rate}-{weight_decay}",
            eval_strategy = "no",
            save_strategy = "no",
            learning_rate=learning_rate,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            num_train_epochs = num_epochs,
            weight_decay = weight_decay,
            load_best_model_at_end=True,
            #metric_for_best_model=mse,
            push_to_hub=False
        )

        trainer = CustomTrainer(
            self.model,
            args,
            train_dataset = train_data_list,
            eval_dataset = test_data_list,
            processing_class = self.tokenizer,
            compute_metrics = self.compute_metrics,
        )

        trainer.train()

        results = trainer.evaluate()

        return results, trainer
    
    def predict_readability(self, sentence: str, sent_tokenizing = False) -> float:
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
            inputs = self.tokenizer(sent, return_tensors="pt", truncation=True, padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get the predicted readability score
            predicted_readability = logits[0][0].item()

            scores.append(predicted_readability)
        
        readability_score = np.mean(scores)

        return readability_score
    
    def batch_evaluate(self, test_data:ARA_Dataset, file_path:str | None = None):
        """
        Evaluates the model on the test dataset.
        """
        progress_bar = tqdm(total=len(test_data), desc=f"Batch evaluation", unit="sentence")

        evaluations = defaultdict(list)
        for id, row in test_data.features.iterrows():

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


def import_dataset(filepath:str, tokenizer, columns_to_keep = ['text', 'label', 'input_ids', 'attention_mask']):

        imported_data = ARA_model.import_jsonl(filepath)

        new_data = imported_data.copy()

        for key, value in new_data.items():

            value["label"] = float(value["label"]) # Convert label to float

            # Tokenize the text
            encodings = tokenizer(value['text'], truncation=True, padding=True) # truncate if too long, pad with 0 if too short 
            value['input_ids'] = encodings['input_ids']
            value['attention_mask'] = encodings['attention_mask']

            # Remove all columns that are not in columns_to_keep
            for column in list(value.keys()):
                if column not in columns_to_keep:
                    del value[column] 
        return new_data


def tune_hyperparameters(model_checkpoint: str, train_data: list, test_data: list, batch_sizes: list, num_epochs: list, learning_rates: list, weight_decays:list, checkpoint_dir:str, layers_to_disable_gradient: list | None = None):
    
    model_checkpoint = model_checkpoint
    model_name = model_checkpoint.split("/")[-1]

    total_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


    for batch_size in batch_sizes:
        for num_epoch in num_epochs:
            for learning_rate in learning_rates:
                for weight_decay in weight_decays:


                    readability_transformer_model = ARA_Transformer_Model(model_checkpoint=model_checkpoint)
                    readability_transformer_model.model.to(device)

                    # Tune hyperparameters

                    results, trainer = readability_transformer_model.train_model(
                        model_name = model_name,
                        train_data_list = train_data,
                        test_data_list = test_data,
                        batch_size = batch_size,
                        num_epochs = num_epoch,
                        learning_rate = learning_rate,
                        weight_decay = weight_decay,
                        checkpoint_dir = checkpoint_dir,
                        layers_to_disable_gradient=layers_to_disable_gradient,
                        debug = True
                    )

                    print(results)

                    # Save the results
                    total_results[batch_size][num_epoch][learning_rate][weight_decay] = results
                    total_results[batch_size][num_epoch][learning_rate][weight_decay]['trainer'] = trainer
                    total_results[batch_size][num_epoch][learning_rate][weight_decay]['model'] = readability_transformer_model.model
                    total_results[batch_size][num_epoch][learning_rate][weight_decay]['tokenizer'] = readability_transformer_model.tokenizer
                    total_results[batch_size][num_epoch][learning_rate][weight_decay]['readability_transformer_model'] = readability_transformer_model
                   

    
    return total_results

def find_best_model(results, order_by = 'eval_MSE'):
    best_model = None
    best_score = float('inf')

    for batch_size, batch_results in results.items():
        for num_epoch, epoch_results in batch_results.items():
            for learning_rate, lr_results in epoch_results.items():
                for weight_decay, wd_results in lr_results.items():
                    score = wd_results[order_by]
                    if score < best_score:
                        best_score = score
                        best_model = wd_results['model']
                        best_readability_transformer_model = wd_results['readability_transformer_model']

    return best_model, best_score, best_readability_transformer_model
    


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
    with open("ARA_Transformer_model_settings.json", "r") as settings_file:
        settings = json.load(settings_file)


    # Check if their were parameters provided via command line
    parser = argparse.ArgumentParser(description="ARA Transformer Model")

    parser.add_argument("--mode", type=str, help="Mode of this ARA Model", default="train", choices=["train", "eval", "cli", "inference-timing"])
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--input_sentence", type=str, help="Input sentence to rate (only in cli mode)")

    args = parser.parse_args()

    if args.model_path:
        ARA_model_path = args.model_path
    else:
        ARA_model_path = settings["ARA_model_path"]


    if args.mode == "train":

        ARA_Transformer_model_instance = ARA_Transformer_Model(model_checkpoint=settings["model_checkpoint"])
        ARA_Transformer_model_instance.model.to(device)

        # Import the dataset
        train_data = import_dataset(settings["train_data"], ARA_Transformer_model_instance.tokenizer)
        dev_data = import_dataset(settings["dev_data"], ARA_Transformer_model_instance.tokenizer)


        # Convert to list
        # [{text: ..., label: ..., input_ids: ..., attention_mask: ...}, ...]
        train_data_list = [train_data[key] for key in list(train_data.keys())]
        dev_data_list = [dev_data[key] for key in list(dev_data.keys())]

        # For use on the FAU HPC Server so that we can save the checkpoint to the WORK directory
        try:
            checkpoint_dir = os.getenv("WORK")

            assert isinstance(checkpoint_dir, str), "Checkpoint dir must be string"
            checkpoint_dir = checkpoint_dir + "/checkpoints/ARA_Transformer_model"

        except:
            checkpoint_dir = settings["checkpoint_dir"]


        results = tune_hyperparameters(
            model_checkpoint = settings["model_checkpoint"],
            train_data = train_data_list,
            test_data = dev_data_list,
            batch_sizes = settings["batch_sizes"],
            num_epochs = settings["num_epochs"],
            learning_rates = settings["learning_rates"],
            weight_decays = settings["weight_decays"],
            checkpoint_dir = checkpoint_dir,
            layers_to_disable_gradient = settings["layers_to_disable_gradient"]
        )

        # Find the best model
        best_model, best_score, best_ARA_Transformer_model = find_best_model(results)

        assert type(best_ARA_Transformer_model) == ARA_Transformer_Model

        print(f"Best score: {best_score}")
        
        # Save the model
        best_ARA_Transformer_model.save_model(ARA_model_path)

        
        print("Train data")
        train_data = ARA_model.import_jsonl(settings["train_data"])
        train_data = ARA_Dataset(data=train_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        best_ARA_Transformer_model.batch_evaluate(train_data, file_path="predictions/ARA_Transformer_model_train.jsonl")

        print("Dev data")
        dev_data = ARA_model.import_jsonl(settings["dev_data"])
        dev_data = ARA_Dataset(data=dev_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        best_ARA_Transformer_model.batch_evaluate(dev_data, file_path="predictions/ARA_Transformer_model_dev.jsonl")

        print("Eval data")
        eval_data = ARA_model.import_jsonl(settings["eval_data"])
        eval_data = ARA_Dataset(data=eval_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        best_ARA_Transformer_model.batch_evaluate(eval_data, file_path="predictions/ARA_Transformer_model_eval.jsonl")


    elif args.mode == "eval":

        ARA_Transformer_Model_instance = ARA_Transformer_Model(model_checkpoint=ARA_model_path, eval=True)

        # print("Train data")
        # train_data = ARA_model.import_jsonl(settings["train_data"])
        # train_data = ARA_Dataset(data=train_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        # ARA_Transformer_Model_instance.batch_evaluate(train_data, file_path="predictions/ARA_Transformer_model_train.jsonl")

        # print("Dev data")
        # dev_data = ARA_model.import_jsonl(settings["dev_data"])
        # dev_data = ARA_Dataset(data=dev_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        # ARA_Transformer_Model_instance.batch_evaluate(dev_data, file_path="predictions/ARA_Transformer_model_dev.jsonl")

        # print("Eval data")
        eval_data = ARA_model.import_jsonl(settings["eval_data"])
        eval_data = ARA_Dataset(data=eval_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        ARA_Transformer_Model_instance.batch_evaluate(eval_data, file_path="predictions/ARA_Transformer_model_eval.jsonl")


    elif args.mode == "cli":
        # Load the model
        ARA_Transformer_Model_instance = ARA_Transformer_Model(model_checkpoint=ARA_model_path, eval=True)

        ARA_Transformer_Model_instance.cli_mode(args.input_sentence, "ARA_Transformer_model")

    elif args.mode == "inference-timing":
        ARA_Transformer_Model_instance = ARA_Transformer_Model(model_checkpoint=ARA_model_path, eval=True)

        test_sentences = ARA_model.import_jsonl(settings["eval_data"])
        test_dataset = ARA_Dataset(data=test_sentences, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])

        ARA_Transformer_Model_instance.measure_inference_time(test_dataset)