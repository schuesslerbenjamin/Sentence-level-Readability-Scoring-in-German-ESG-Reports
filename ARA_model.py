import json
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import sklearn.metrics
import scipy.stats
import time


class ARA_Model:
    def __init__(self, model_name: str):
        raise NotImplementedError

    def load_model(self):
        # Placeholder for loading the actual model     
        raise NotImplementedError
    
    def save_model(self):
        # Placeholder for saving the actual model
        raise NotImplementedError
    
    def predict_readability(self, sentence: str, sent_tokenizing = False) -> float:
        # Placeholder for predicting readability
        raise NotImplementedError
    
    def save_evaluations(self, evaluations: defaultdict[str, list], file_path: str):
        """
          Save the evaluations to a jsonl
        """
        with open(file_path, 'w') as file:
            for i in range(len(evaluations["predictions"])):
                new_item = {
                    "id": evaluations["ids"][i],
                    "sentence": evaluations["sentences"][i],
                    "prediction": evaluations["predictions"][i],
                    "label": evaluations["labels"][i]
                }
                json_line = json.dumps(new_item)
                file.write(json_line + '\n')

    def analyze_predictions(self, predictions, labels, file_path: str, row_index: str | None = None, round_to:int = 4):
        """
        Save analysis results as a csv file
        """
        
        mse = sklearn.metrics.mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = sklearn.metrics.mean_absolute_error(labels, predictions)
        kendall_tau, p_value = scipy.stats.kendalltau(labels, predictions, variant='b')

        analysis = pd.DataFrame({
            "mse": [mse],
            "mae": [mae],
            "rmse": [rmse],
            "kendall_tau": [kendall_tau],
            "p_value": [p_value]
        }, index=[row_index])

        analysis = analysis.astype(float)
        print(analysis.round(4))

        if file_path:
            if os.path.exists(file_path):
                header = False
            else:
                header = True

            analysis.to_csv(file_path, index=True, header=header, mode = "a", float_format=f'%.{round_to}f')

    
    def cli_mode(self, input_sentence: str, model:str):
        """
        Runs the predict_readability function to rate the input_sentence.
        """

        if not input_sentence:
            try: 
                with open("cli_sentence.json", 'r') as file:
                    input_sentence = json.load(file)
            except:
                print("Please provide an input sentence to rate using --input_sentence")
                quit()
        
        rating = self.predict_readability(input_sentence, sent_tokenizing = True)
        print(f"{rating:.2f}")

        self.add_cli_score(model, rating)


    
    def add_cli_score(self, model:str, score):
        """
        Persist the ARA score.
        """

        try:
            with open("cli_scores.json", 'r') as file:
                cli_scores = json.load(file)
        except FileNotFoundError:
            cli_scores = {}
            
        cli_scores[model] = score

        with open("cli_scores.json", 'w') as file:
            json.dump(cli_scores, file)

    def measure_inference_time(self, test_data) -> float:
        """
        Measures the average inference time per sentence on the test dataset.
        """

        total_time = 0.0
        for id, row in test_data.features.iterrows():

            assert isinstance(id, str)

            sentence = test_data.sentences[id]

            start_time = time.time()
            readability_score = self.predict_readability(sentence)
            end_time = time.time()

            total_time += (end_time - start_time)
            print(f"Sentence ID: {id} | Inference time: {end_time - start_time:.4f} seconds | Readability Score: {readability_score}")
        
        average_time = total_time / len(test_data)
        print(f"Average inference time per sentence: {average_time:.4f} seconds")
        return average_time

        
        


def import_jsonl(filepath: str, row_key: str = "id"):
        """
        Imports jsonl files 
        Returns a dictionary with {key} as key and the rest of the data as value    
        """
        
        with open(filepath, 'r') as file:
            data = {row[row_key]: {key: value for key, value in row.items() if key != row_key} for row in (json.loads(line) for line in file)}

        return data

def scale_down(label: int) -> float:
    """
    Scale the label down to the range of 0 to 1.
    """

    return (label - 1) / 3.0

def scale_up(label: float) -> int:
    """
    Scale the label up to the range of 1 to 4.
    """
    
    return int(label * 3.0 + 1)