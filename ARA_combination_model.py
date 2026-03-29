import json
import numpy as np
import random
import pandas as pd
import argparse
from collections import defaultdict
import sklearn.metrics
import scipy.stats



    

def load_submodel_predictions(predictions_paths: dict) -> pd.DataFrame:
    """
    Loads the predictions from the submodels and returns them as a DataFrame.
    The input is a dictionary where the keys are the submodel names and the values are the file paths to the predictions of the respective submodel.
    The output is a DataFrame where each column corresponds to a submodel and contains the predictions of that submodel.
    """
    submodel_predictions = defaultdict(dict)

    for model_name, file_path in predictions_paths.items():
        with open(file_path, 'r') as file:
            for line in file:
                item = json.loads(line)

                id = item["id"]
                prediction = item["prediction"]

                submodel_predictions[id]["label"] = item["label"]
                submodel_predictions[id][model_name] = prediction

    submodel_predictions_df = pd.DataFrame.from_dict(submodel_predictions, orient='index')
    return submodel_predictions_df

def batch_predict_readability(submodel_predictions: pd.DataFrame, submodels:list):

    columname = " + ".join([model for model in submodels])

    submodel_predictions[columname] = submodel_predictions[submodels].mean(axis=1)

    return submodel_predictions

def evaluation(labels, predictions):
    
    mse = sklearn.metrics.mean_squared_error(labels, predictions)
    mae = sklearn.metrics.mean_absolute_error(labels, predictions)
    kendall_tau, p_value = scipy.stats.kendalltau(labels, predictions, variant='b')

    results = {
        "mse": mse,
        "mae": mae,
        "kendall_tau": kendall_tau,
        "p_value": p_value
    }

    return results

def batch_evaluation(readability_combinations:pd.DataFrame):

    evaluation_results = pd.DataFrame()

    evaluation_results["Syntax + Transformer"] = evaluation(readability_combinations["label"], readability_combinations["Syntax + Transformer"])
    evaluation_results["Syntax + LLM"] = evaluation(readability_combinations["label"], readability_combinations["Syntax + LLM"])
    evaluation_results["Transformer + LLM"] = evaluation(readability_combinations["label"], readability_combinations["Transformer + LLM"])
    evaluation_results["Syntax + Transformer + LLM"] = evaluation(readability_combinations["label"], readability_combinations["Syntax + Transformer + LLM"])

    print(evaluation_results.round(4).T)



if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    

    # Check if their were parameters provided via command line
    parser = argparse.ArgumentParser(description="ARA Combination Model")

    parser.add_argument("--mode", type=str, help="Mode of this Combination Model", default="train", choices=["eval", "cli"])
    parser.add_argument("--submodels", type=list, help="Submodels to combine the scores from", default=None)
    parser.add_argument("--cli_scores_file", type=str, help="Filepath to the file that holds the scores from the other models (only in cli mode)")

    args = parser.parse_args()

    if args.submodels:
        submodels = args.submodels
    else:
        submodels = None



    if args.mode == "eval":

        eval_predictions_paths = {
                    "Syntax": "predictions/ARA_Syntax_model_eval.jsonl",
                    "Transformer": "predictions/ARA_Transformer_model_eval.jsonl",
                    "LLM": "predictions/ARA_LLM_model_eval.jsonl"}
        
        submodel_predictions = load_submodel_predictions(eval_predictions_paths)

        readability_combinations = batch_predict_readability(submodel_predictions, submodels=["Syntax", "Transformer"])
        readability_combinations = batch_predict_readability(submodel_predictions, submodels=["Syntax", "LLM"])
        readability_combinations = batch_predict_readability(submodel_predictions, submodels=["Transformer", "LLM"])
        readability_combinations = batch_predict_readability(submodel_predictions, submodels=["Syntax", "Transformer", "LLM"])


        batch_evaluation(readability_combinations)


    elif args.mode == "cli":
        with open(args.cli_scores_file, 'r') as file:
            cli_scores = json.load(file)

        print(f"Syntax + XLM-base: {np.mean([cli_scores['ARA_Syntax_model'], cli_scores['ARA_Transformer_model']]):.2f}")
        print(f"Syntax + Qwen: {np.mean([cli_scores['ARA_Syntax_model'], cli_scores['ARA_LLM_model']]):.2f}")
        print(f"XLM-base + Qwen: {np.mean([cli_scores['ARA_Transformer_model'], cli_scores['ARA_LLM_model']]):.2f}")
        print(f"Syntax + XLM-base + Qwen: {np.mean([cli_scores['ARA_Syntax_model'], cli_scores['ARA_Transformer_model'], cli_scores['ARA_LLM_model']]):.2f}")