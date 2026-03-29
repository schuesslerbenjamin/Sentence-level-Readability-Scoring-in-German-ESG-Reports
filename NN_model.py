import json
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
import scipy.stats

import ARA_model
from NN_Dataset import NNDataset


class NNModel(ARA_model.ARA_Model):
    def __init__(self, criterion = None, batch_size:int = 10, seed:int = 42, model = None, optimizer = None):

        """
        Only initializes a this class, to create and train a NN use train_model afterwards
        """

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")

        self.seed = seed

        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer

    def train_model(self, train_data: NNDataset, dev_data: NNDataset, batch_size:int = 15, num_epochs:int = 40, learning_rate:float = 0.01, early_stopping_patience:int = 20):
        """
        creates and trains a NN
        """

        self.batch_size = batch_size

        input_size = train_data.get_num_features()

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_data, batch_size=self.batch_size, shuffle=False)

        model = self.create_model(input_size)
        model = model.to(self.device)

        optimizer = self.create_optimizer(model, learning_rate)

        model, optimizer, validation_losses = self.fit_model(model, optimizer, train_loader, dev_loader, self.criterion, num_epochs, early_stopping_patience)

        self.model = model

        return model, optimizer, validation_losses
    
    def create_model(self, input_size):
        """
        Creates the model with the specified parameters.
        """

        model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer to prevent overfitting

            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer to prevent overfitting

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer to prevent overfitting

            nn.Linear(32, 1)
        )

        return model.to(self.device)
    
    def create_optimizer(self, model, learning_rate):
        """
        Creates the optimizer with the specified parameters.
        """

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        return optimizer
    
    def fit_model(self, model, optimizer, train_loader, val_loader, criterion, num_epoch, early_stopping_patience):
        """
        Fits the model with the specified parameters.
        """

        best_val_loss = float('inf')
        patience_counter = 0

        validation_losses = []

        for epoch in range(num_epoch):
            model.train()
            
            for features, target, sentence in train_loader:
                features = features.to(self.device)
                target = target.to(self.device)
                target = torch.unsqueeze(target, 1) # adds dimension to target -> solves Warning

                optimizer.zero_grad()

                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, target)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0.0

            predictions = []
            targets = []

            with torch.no_grad():
                for features, target, sentence in val_loader:
                    features = features.to(self.device)
                    target = target.to(self.device)

                    outputs = model(features)
                    
                    predictions.append(outputs.cpu().numpy())
                    targets.append(target.cpu().numpy())

            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)

            # Calculate the validation loss
            predictions_tensor = torch.tensor(predictions)
            targets_tensor = torch.tensor(targets)
            targets_tensor = torch.unsqueeze(targets_tensor, 1) # adds dimension to target -> solves Warning
            val_loss = criterion(predictions_tensor, targets_tensor).item()

            validation_losses.append(val_loss)

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                break
    
        return model, optimizer, validation_losses
    
    @classmethod
    def from_saved(cls, model_path):

        """
        Loads the model from a specified path.
        """

        with open(model_path, 'r') as file:
            input = json.load(file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        NN_model_path = input['NN_model_path']
        model = torch.load(NN_model_path, map_location=device, weights_only=False)

        # Load the optimizer
        NN_optimizer_path = input['NN_optimizer_path']
        optimizer = torch.load(NN_optimizer_path, map_location=device, weights_only=False)

        # Load the criterion
        NN_criterion_path = input['NN_criterion_path']
        with open(NN_criterion_path, 'rb') as file:
            criterion = pickle.load(file)

        # Load the settings
        batch_size = input['batch_size']
        seed = input.get('seed', 42)

        return cls(criterion=criterion, batch_size=batch_size, seed=seed, model=model, optimizer=optimizer)


    def save_model(self, model_path: str):
        """
        Saves the model to a specified path.
        """

        NN_model_path = f"{model_path}_NN_model.pt"
        NN_criterion_path = f"{model_path}_NN_criterion.pt"
        NN_optimizer_path = f"{model_path}_NN_optimizer.pt"

        output = {
            'batch_size': self.batch_size,
            'NN_model_path': NN_model_path,
            'NN_optimizer_path': NN_optimizer_path,
            'NN_criterion_path': NN_criterion_path
            }
        
        # Save the whole model. When tuning the model structure, the only saving the learned parameters is not enough to recreate.
        torch.save(self.model, NN_model_path)

        # Save the optimizer
        torch.save(self.optimizer, NN_optimizer_path)

        # Save the criterion
        with open(NN_criterion_path, 'wb') as file:
            pickle.dump(self.criterion, file)

        # Save the settings file
        with open(model_path, 'w') as file:
            json.dump(output, file)

    def eval_model(self, eval_data:NNDataset, file_path:str | None = None, verbose:bool = False, model = None, criterion = None, analysis_csv_path = None):
        """
        Evaluates the model on the eval dataset.
        """

        eval_loader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False)        

        if model is None:
            model = self.model

        assert model is not None

        model.to(self.device)
        model.eval()

        if criterion is None:
            criterion = self.criterion
 
        self.criterion.to(self.device)

        predictions = []
        targets = []

        with torch.no_grad():
            for features, target, sentence in eval_loader:
                features = features.to(self.device)
                target = target.to(self.device)

                outputs = model(features)
                predictions.append(outputs.cpu().numpy())
                
                try:
                    targets.extend(target.cpu().numpy())
                except:
                    targets.append(target.cpu().numpy())

        predictions = np.concatenate(predictions)
        targets = np.array(targets)

        # Calculate the loss
        predictions_tensor = torch.tensor(predictions)
        targets_tensor = torch.tensor(targets)
        targets_tensor = torch.unsqueeze(targets_tensor, 1) # adds dimension to target -> solves Warning
        loss = self.criterion(predictions_tensor, targets_tensor).item()

        if file_path and isinstance(eval_data, NNDataset):
            # Save the results
            evaluations = defaultdict()

            evaluations["ids"] = eval_data.indices

            evaluations["sentences"] = eval_data.sentences
            evaluations["labels"] = [float(sentence) for sentence in targets]

            predictions = predictions.flatten()
            evaluations["predictions"] = [float(sentence) for sentence in predictions]
            
            self.save_evaluations(evaluations, file_path)

        # Calculate the mean absolute error
        mae = sklearn.metrics.mean_absolute_error(targets, predictions)

        # Calculate the mean squared error
        mse = sklearn.metrics.mean_squared_error(targets, predictions)

        # Calculate the root mean squared error
        rmse = sklearn.metrics.root_mean_squared_error(targets, predictions)

        # Kendall tau b
        kendall_tau, p_value = scipy.stats.kendalltau(targets, predictions, variant='b')

        if verbose:
            print(f"MAE: {mae:.6f} MSE: {mse:.6f} RMSE: {rmse:.6f} Kendall Tau: {kendall_tau:.6f} (p-value: {p_value:.6f})")

        if analysis_csv_path:
            self.analyze_predictions(predictions, targets, file_path=analysis_csv_path, row_index="NN")
        return loss



def hyperparameter_tuning(train_data: NNDataset, dev_data: NNDataset, batch_sizes:list, num_epochs:list, learning_rates:list, early_stopping_patiences:list, criterion, seed = 42):
    """
    Creates a NNModel for every combination of hyperparameters, does cross validation on the training set, and evaluates the model on the dev set.
    Return the besr hyperparameters and the loss of the best model
    """
    print("starting hyperparameter tuning...")

    input_size = train_data.get_num_features()

    num_tuning = len(batch_sizes) * len(num_epochs) * len(learning_rates) * len(early_stopping_patiences)
    i = 1

    hyperparameter_dev_loss = defaultdict(float)

    for batch_size in batch_sizes:
        for num_epoch in num_epochs:
            for learning_rate in learning_rates:
                for early_stopping_patience in early_stopping_patiences:
                    print(f"Training model {i}/{num_tuning} with batch_size={batch_size}, num_epochs={num_epoch}, lr={learning_rate}, early_stopping={early_stopping_patience}")

                    i += 1

                    # Repeated cross validation (repeated kfold) for training and validation sets
                    rkf  = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)
                    results = defaultdict(list) # {fold_index: [model, best_val_loss, dev_loss]}}

                    for fold_index, (train_index, val_index) in enumerate(rkf.split(train_data.features)):

                        train_set = NNDataset(train_data.get_features_by_idx(train_index), train_data.get_targets_by_idx(train_index), train_data.get_sentences_by_idx(train_index), train_data.indices[train_index])

                        val_set = NNDataset(train_data.get_features_by_idx(val_index), train_data.get_targets_by_idx(val_index), train_data.get_sentences_by_idx(val_index), train_data.indices[val_index])

                        # create NNmodel
                        NNmodel = NNModel(criterion=criterion, batch_size=batch_size, seed=seed)

                        model, optimizer, _ = NNmodel.train_model(train_set, val_set, batch_size, num_epoch, learning_rate, early_stopping_patience)

                        results[fold_index].append((model, optimizer))



                    # Evaluate the model on the dev set
                    for fold_index, toup in results.items():
                        model, optimizer = toup[0]
                        
                        dev_loss = NNmodel.eval_model(dev_data, model = model, criterion = criterion)

                        toup = (model, optimizer, dev_loss)
                        results[fold_index] = toup

                    # Save the mean dev loss for this hyperparameter combination
                    mean_dev_loss = np.mean([toup[2] for toup in results.values()])                      
                    hyperparameter_dev_loss[(batch_size, num_epoch, learning_rate, early_stopping_patience)] = mean_dev_loss


    # Find the best hyperparameter combination
    best_model = None
    best_dev_loss = float('inf')
    best_hyperparameters = None
    for hyperparameters, mean_dev_loss in hyperparameter_dev_loss.items():
        
        if mean_dev_loss < best_dev_loss:
            best_dev_loss = mean_dev_loss
            best_hyperparameters = hyperparameters

    assert best_hyperparameters is not None

    print()
    print(f"Hyperparameters of best model: {best_hyperparameters}")

    return best_hyperparameters, best_dev_loss