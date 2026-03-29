import spacy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from tqdm import tqdm
import pickle
import sklearn.metrics
from sklearn.model_selection import RepeatedKFold
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import argparse
import random
import os
import nltk
import textdescriptives # required for the spacy pipeline (dependency_distance), even if pylint doesn't recognize it

import ARA_model
from ARA_dataset import ARA_Dataset
from NN_Dataset import NNDataset


class CompressedInputNN(nn.Module):
    """
    THis class is basically the normal torch NN, but it compresses the first part of the input (the ngrams) first, before it pushes it to the normal second layer (and then as usual).
    The split-point is defined by the compressed_input_size
    """
    def __init__(self, compressed_input_size, non_compressed_input_size):
        
        self.compressed_input_size = compressed_input_size
        self.non_compressed_input_size = non_compressed_input_size
        
        super(CompressedInputNN, self).__init__()
            
        # First Layer: Compressed part (n-grams) on the left and non-compressed part (other features) on the right

        compressed_output_size = 500
        non_compressed_output_size = 25

        # First part to be compressed
        self.first_layer_compressed_part = nn.Sequential(
            nn.Linear(compressed_input_size, compressed_output_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout layer to prevent overfitting
        )

        # Second part, not compressed
        self.first_layer_non_compressed_part = nn.Sequential(
            nn.Linear(non_compressed_input_size, non_compressed_output_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout layer to prevent overfitting
        )

        # Second Layer
        self.second_layer = nn.Sequential(
            nn.Linear(compressed_output_size + non_compressed_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout layer to prevent overfitting
        )

        # Third Layer
        self.third_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout layer to prevent overfitting
        )

        # Output Layer
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        """
        x is the full input vector, both compressed and non compressed
        """

        # if x is one dim
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # First Layer
        # Split the input into compressed and non-compressed parts
        compressed_part = x[:, :self.compressed_input_size]
        non_compressed_part = x[:, self.compressed_input_size:]

        # Process the compressed part
        compressed_output = self.first_layer_compressed_part(compressed_part)

        # Process the non-compressed part
        non_compressed_output = self.first_layer_non_compressed_part(non_compressed_part)

        # Concatenate the outputs of both parts
        combined_output = torch.cat((compressed_output, non_compressed_output), dim=1)


        # Second layer
        output = self.second_layer(combined_output)

        # third layer
        output = self.third_layer(output)

        # output Layer
        output = self.output_layer(output)

        return output
    


class ARA_Syntax_Model(ARA_model.ARA_Model):
    def __init__(self, spacy_model: str, banned_pos_tags: list, train_data: ARA_Dataset | None = None, dev_data: ARA_Dataset | None = None, n_list:list = [2, 3], batch_sizes:list = [20], num_epochs:list = [40], learning_rates:list = [0.01], early_stopping_patiences:list = [15], seed:int = 42, criterion = None, batch_size: int | None = None, optimizer = None, model = None, ngrams_baseline = None, sentence_roots = None):

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")

        self.seed = seed
        
        self.spacy_model = spacy_model
        self.spacy_nlp = spacy.load(spacy_model)
        self.spacy_nlp.add_pipe("textdescriptives/dependency_distance")

        self.banned_pos_tags = banned_pos_tags

        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
                
        self.n_list = n_list

        if train_data is not None and dev_data is not None:

            self.train_data = train_data
            self.dev_data = dev_data

            self.ngrams_baseline, self.sentence_roots = self.train_baseline(train_data.get_features())

            train_features = self.batch_feature_extraction(train_data)
            NN_train_data = NNDataset(train_features, train_data.target, train_data.sentences)

            dev_features = self.batch_feature_extraction(dev_data)
            NN_dev_data = NNDataset(dev_features, dev_data.target, dev_data.sentences)


            best_model, best_optimizer, best_hyperparameters = self.hyperparameter_tuning(
                train_data=NN_train_data,
                dev_data=NN_dev_data,
                batch_sizes=batch_sizes,
                num_epochs=num_epochs,
                learning_rates=learning_rates,
                early_stopping_patiences=early_stopping_patiences,
                criterion = nn.MSELoss()
            )

            self.batch_size = best_hyperparameters[0]
            self.model = best_model.to(self.device)
            self.optimizer = best_optimizer

        else:
            assert batch_size is not None
            self.batch_size = batch_size

            assert model is not None
            self.model = model

            assert optimizer is not None
            self.optimizer = optimizer

            assert ngrams_baseline is not None
            self.ngrams_baseline = ngrams_baseline

            assert sentence_roots is not None
            self.sentence_roots = sentence_roots



    def hyperparameter_tuning(self, train_data: NNDataset, dev_data: NNDataset, batch_sizes:list, num_epochs:list, learning_rates:list, early_stopping_patiences:list, criterion):

        input_size = train_data.get_num_features()

        hyperparameter_test_loss = defaultdict(float)

        for batch_size in batch_sizes:
            for num_epoch in num_epochs:
                for learning_rate in learning_rates:
                    for early_stopping_patience in early_stopping_patiences:
                        print(f"Training model with batch_size={batch_size}, num_epochs={num_epoch}, lr={learning_rate}, early_stopping={early_stopping_patience}", end="")

                        # Repeated cross validation (repeated kfold) for training and validation sets
                        rkf  = RepeatedKFold(n_splits=5, n_repeats=2, random_state=self.seed)
                        results = defaultdict(list) # {fold_index: [model, best_val_loss, test_loss]}}

                        for fold_index, (train_index, val_index) in enumerate(rkf.split(train_data.features)):
                            print(f"\nFold {fold_index + 1}", end="")

                            train_set = NNDataset(train_data.get_features_by_idx(train_index), train_data.get_targets_by_idx(train_index), train_data.get_sentences_by_idx(train_index), train_data.indices[train_index])

                            val_set = NNDataset(train_data.get_features_by_idx(val_index), train_data.get_targets_by_idx(val_index), train_data.get_sentences_by_idx(val_index), train_data.indices[val_index])

                            # Create data loaders
                            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
                            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

                            

                            # Create the model
                            model = self.create_model(input_size)
                            model = model.to(self.device)

                            # Create the optimizer
                            optimizer = self.create_optimizer(model, learning_rate)

                            # Train the model
                            model, optimizer, _ = self.train_model(model, optimizer, train_loader, val_loader, criterion, num_epoch, early_stopping_patience)

                            results[fold_index].append((model, optimizer))


                        # Evaluate the model on the test set
                        for fold_index, toup in results.items():
                            model, optimizer = toup[0]
                            print(f"\nFold {fold_index + 1}", end=": ")
                            
                            mean_test_loss = self.eval_model(dev_data, batch_size, model, criterion)

                            toup = (model, optimizer, mean_test_loss)
                            results[fold_index] = toup

                        # Save the mean test loss for this hyperparameter combination
                        mean_test_loss = np.mean([toup[2] for toup in results.values()])                      
                        hyperparameter_test_loss[(batch_size, num_epoch, learning_rate, early_stopping_patience)] = mean_test_loss


        # Find the best hyperparameter combination
        best_model = None
        best_test_loss = float('inf')
        best_hyperparameters = None
        for hyperparameters, mean_test_loss in hyperparameter_test_loss.items():
            
            if mean_test_loss < best_test_loss:
                best_test_loss = mean_test_loss
                best_hyperparameters = hyperparameters

        assert best_hyperparameters is not None, "No best hyperparameters found."

        print()
        print(f"Hyperparameters of best model: {best_hyperparameters}")
        
        ####################################################################################
        # Train the best model on the whole training set
        ####################################################################################

        batch_size, num_epoch, learning_rate, early_stopping_patience = best_hyperparameters

        full_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        full_test_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

        model = self.create_model(input_size)
        model = model.to(self.device)

        optimizer = self.create_optimizer(model, learning_rate)

        best_model, best_optimizer, validation_losses = self.train_model(model, optimizer, full_train_loader, full_test_loader, criterion, num_epoch, early_stopping_patience)

        return best_model, best_optimizer, best_hyperparameters
                        
    
    def create_model(self, total_input_size):
        """
        Creates the model with the specified parameters.
        """

        compressed_input_size = 0
        for n in self.n_list:
            compressed_input_size += len(self.ngrams_baseline[n])

        non_compressed_input_size = total_input_size - compressed_input_size

        model = CompressedInputNN(
            compressed_input_size=compressed_input_size,
            non_compressed_input_size=non_compressed_input_size
        )
        
        return model.to(self.device)
    
    def create_optimizer(self, model, learning_rate):
        """
        Creates the optimizer with the specified parameters.
        """

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        return optimizer
    
    def train_model(self, model, optimizer, train_loader, val_loader, criterion, num_epoch, early_stopping_patience):

        """
        Trains the model with the specified parameters.
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
    
            
    def get_pos_tag_sequence(self, sentence: str) -> list:
        """
        Identifies the grammatical structure of a given sentence. We define this as the order of the POS tags
        """

        doc = self.spacy_nlp(sentence)
        
        syntactical_structure = [token.pos_ for token in doc if token.pos_ not in self.banned_pos_tags]

        return syntactical_structure
    
    def generate_ngrams(self, syntactical_structure: list, n=2):

        ngrams = zip(*[syntactical_structure[i:] for i in range(n)]) # Generate all n-grams. The zip function returns a list of tuples with n elements each
        
        return ngrams
    
    def get_ngrams(self, sentence:str, n=2):
        """
        Return a list of the POS n-Grams in a sentence
        """

        pos_tags = self.get_pos_tag_sequence(sentence)
        ngrams = self.generate_ngrams(pos_tags, n=n)

        return list(ngrams)
    
    def get_tree_depth(self, node, depth):
        """
        recursively calculates the depth of a tree. The root is considered as height 0
        """
        
        if node.n_lefts + node.n_rights > 0:
            return max(self.get_tree_depth(child, depth + 1) for child in node.children)
        else:
            return depth
        
    def has_subordinate_clause(self, doc) -> int:
        """
        Checks if the sentence contains a subordinate clause.
        Entweder KOUI (unterordnende Konj. mit Infinitiv) oder KOUS (unterordnende Konj. mit Satz)
        Source: https://homepage.ruhr-uni-bochum.de/stephen.berman/Korpuslinguistik/Tagsets-STTS.html
        """
        for sent in doc.sents:
            for token in sent:
                if token.tag_=="KOUI" or token.tag_=="KOUS":
                    return 1
        return 0
    
    def is_passive(self, doc) -> int:
        """
        Returns if the sentence is in passive voice.
        A sentence is considered passive if either:
        - It contains a participle with werden as head
        - It contains a passivized subject
        """

        for sent in doc.sents:
            for token in sent:

                if "VerbForm=Part" in str(token.morph) and token.head.lemma_ == "werden":
                    return 1

                if token.dep_ == "sbp":
                    return 1

        return 0
        
    def calculate_base_features(self, doc):
        """
        Only works with the spacy pipeline
        """
        syntactical_structure = [token.pos_ for token in doc if token.pos_ not in self.banned_pos_tags]

        
        # Extract how often each n-Gram occurs in the sentence.
        ngrams = defaultdict(Counter)
        for n in self.n_list:
            new_ngrams = self.generate_ngrams(syntactical_structure, n=n)
            ngrams[n].update(new_ngrams)
            
        # Calculate the depth of the tree
        sentence_depth = [self.get_tree_depth(sent.root, 0) for sent in doc.sents][0]
        
        # Calculate the dependency distance
        mean_dependency_distance = doc._.dependency_distance["dependency_distance_mean"].item() # .item to convert to float

        # Find the POS-tag of the root of the sentence
        root = [sent.root for sent in doc.sents][0]
        sentence_root = root.pos_

        # Check if it is a passive sentence
        is_passive = self.is_passive(doc)
        
        # Check if the sentence contains a subordinate clause
        has_subordinate_clause = self.has_subordinate_clause(doc)

        return ngrams, sentence_depth, mean_dependency_distance, sentence_root, is_passive, has_subordinate_clause

    def train_baseline(self, training_data):
        """
        This function extracts all possible n-grams and sentence roots from the trainings data. We need this to know to which vector to expand the features in get_NN_features
        """
        ngrams = defaultdict(Counter)
        sentence_roots = defaultdict(Counter)

        training_data = training_data["text"].tolist()


        progress_bar = tqdm(total=len(training_data), desc=f"Training baseline", unit="sentence")

        for doc in self.spacy_nlp.pipe(training_data, batch_size=100):
            syntactical_structure = [token.pos_ for token in doc if token.pos_ not in self.banned_pos_tags]
            
            # Generate all n-grams
            for n in self.n_list:
                new_ngrams = self.generate_ngrams(syntactical_structure, n=n)
                ngrams[n].update(new_ngrams)

            # Generate the sentence root
            root = [sent.root for sent in doc.sents][0]
            root = root.pos_
            sentence_roots[root].update([root])

            progress_bar.update(1)

        progress_bar.close()

        ngrams = {n: list(set(ngrams[n].elements())) for n in self.n_list}

        sentence_roots = list(sentence_roots.keys())

        return ngrams, sentence_roots

    def get_NN_features(self, doc, flatten = True) -> pd.Series:
        """
        Flatten determines if the output is a simple series or a nested series
        """

        ngrams, sentence_depth, mean_dependency_distance, sentence_root, is_passive, has_subordinate_clause = self.calculate_base_features(doc)

        # Move each n-gram into a vector of zeros and counts
        ngrams_vector = defaultdict()

        for n in self.n_list:
            ngrams_vector[f"{n}_grams"] = pd.Series([0] * len(self.ngrams_baseline[n]), index=self.ngrams_baseline[n])

            ngrams_vector[f"{n}_grams"].update(ngrams[n])


        ngrams_final = pd.concat(ngrams_vector.values()) # aggregate all n-gram values in one vector

        sentence_root_vector = pd.Series([0] * len(self.sentence_roots), index=self.sentence_roots)
        sentence_root_vector.update(pd.Series(1, index=[sentence_root]))

        if flatten:

            base_features = pd.concat([
                ngrams_final, 
                pd.Series(sentence_depth, index=["sentence_depth"]), 
                pd.Series(mean_dependency_distance, index=["mean_dependency_distance"]), 
                sentence_root_vector, 
                pd.Series(is_passive, index=["is_passive"]), 
                pd.Series(has_subordinate_clause, index=["has_subordinate_clause"])
                ])

            assert isinstance(base_features, pd.Series), "Base features are not a pandas series"

            base_features = pd.Series(base_features, name = doc.text) 

            return base_features

        else:
            base_features = pd.Series({
                "sentence_depth": sentence_depth,
                "mean_dependency_distance": mean_dependency_distance,
                "sentence_root": sentence_root_vector,
                "is_passive": is_passive,
                "has_subordinate_clause": has_subordinate_clause
            }, name=doc.text)

            for n in self.n_list:
                base_features[f"{n}_grams"] = ngrams_vector[f"{n}_grams"]

            return base_features
       
    
    def get_features_from_sentence(self, sentence: str):
        """
        Generates the features for a sentence
        """

        doc = self.spacy_nlp(sentence)
        features = self.get_NN_features(doc)

        return features


    def batch_feature_extraction(self, data:ARA_Dataset, flatten = True):
        """
        Generates the features for a dataset
        """

        sentences = data.get_features()["text"].to_list()

        results = pd.DataFrame()

        progress_bar = tqdm(total=len(sentences), desc=f"Feature extraction", unit="sentence")

        for doc in self.spacy_nlp.pipe(sentences):

            features = self.get_NN_features(doc, flatten)
            # results = results.append(features)
            results = pd.concat([results, features], axis=1)


            progress_bar.update(1)
            
        results = results.transpose() # transpose the dataframe to have the features as columns
        
        progress_bar.close()
        
        return results
    
    @classmethod
    def from_saved(cls, model_path):
        """
        Loads the model from a specified path.
        """

        with open(model_path, 'r') as file:
            input = json.load(file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        NN_model_path = f'{model_path}_NN_model.pt'
        model = torch.load(NN_model_path, map_location=device, weights_only=False)

        # Load the optimizer
        NN_optimizer_path = f'{model_path}_NN_optimizer.pt'
        optimizer = torch.load(NN_optimizer_path, map_location=device, weights_only=False)

        # Load the criterion
        NN_criterion_path = f'{model_path}_NN_criterion.pt'
        with open(NN_criterion_path, 'rb') as file:
            criterion = pickle.load(file)

        # Load the settings
        banned_pos_tags = input['banned_pos_tags']
        n_list = input['n_list']
        sentence_roots = input['sentence_roots']
        batch_size = input['batch_size']

        # Load the ngrams baseline
        import_ngrams_baseline = input['ngrams_baseline']

        list_of_tuples = {n: [tuple(x) for x in import_ngrams_baseline[str(n)]] for n in n_list} # convert the key back to int and the nested list to [tuples]

        ngrams_baseline = list_of_tuples

        
        return cls(
            spacy_model=input['spacy_model'],
            banned_pos_tags=banned_pos_tags,
            n_list=n_list,
            ngrams_baseline=ngrams_baseline,
            sentence_roots=sentence_roots,
            batch_size=batch_size,
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )
    
    @classmethod
    def from_n_grams(cls, model_path, train_data, eval_data, batch_sizes, num_epochs, learning_rates, early_stopping_patiences):
        """
        Loads only the spacy model and its outputs (ngrams & sentence_roots) and then trains the model from scratch.
        """

        with open(model_path, 'r') as file:
            input = json.load(file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the settings
        banned_pos_tags = input['banned_pos_tags']
        n_list = input['n_list']
        sentence_roots = input['sentence_roots']

        # Load the ngrams baseline
        import_ngrams_baseline = input['ngrams_baseline']

        list_of_tuples = {n: [tuple(x) for x in import_ngrams_baseline[str(n)]] for n in n_list} # convert the key back to int and the nested list to [tuples]

        ngrams_baseline = list_of_tuples

        
        return cls(
            spacy_model=input['spacy_model'],
            banned_pos_tags=banned_pos_tags,
            train_data=train_data,
            eval_data=eval_data,
            n_list=n_list,
            batch_sizes=batch_sizes,
            num_epochs=num_epochs,
            learning_rates=learning_rates,
            early_stopping_patiences=early_stopping_patiences,
            ngrams_baseline=ngrams_baseline,
            sentence_roots=sentence_roots,
        )

    def save_model(self, model_path: str):
        """
        Saves the model to a specified path.
        """

        NN_model_path = f"{model_path}_NN_model.pt"
        NN_criterion_path = f"{model_path}_NN_criterion.pt"
        NN_optimizer_path = f"{model_path}_NN_optimizer.pt"

        output = {
            'banned_pos_tags': self.banned_pos_tags,
            'spacy_model': self.spacy_model,
            'n_list': self.n_list,
            'ngrams_baseline': self.ngrams_baseline,
            'sentence_roots': self.sentence_roots,
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

    def eval_model(self, test_data:ARA_Dataset | NNDataset, batch_size:int | None = None, model=None, criterion=None, file_path:str | None = None, csv_file_path = None, csv_row_index = None):
        """
        Evaluates the model on the test dataset.
        """

        if isinstance(test_data, ARA_Dataset):
            test_features = self.batch_feature_extraction(test_data)
            NN_test_data = NNDataset(test_features, test_data.target, test_data.sentences)
        
        elif isinstance(test_data, NNDataset):
            NN_test_data = test_data

        else:
            raise ValueError

        test_loader = DataLoader(NN_test_data, batch_size=batch_size, shuffle=False)        

        if model is None:
            model = self.model

        assert isinstance(model, nn.Module)
        
        model.to(self.device)
        model.eval()

        if criterion is None:
            criterion = nn.MSELoss()

        if batch_size is None:
            batch_size = self.batch_size



        predictions = []
        targets = []

        with torch.no_grad():
            for features, target, sentence in test_loader:
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
        loss = criterion(predictions_tensor, targets_tensor).item()

        if file_path:
            # Save the results
            evaluations = defaultdict(list)

            assert isinstance(test_data, ARA_Dataset)

            evaluations["ids"] = test_data.features.index
            evaluations["sentences"] = test_data.sentences
            evaluations["labels"] = test_data.target


            save_predictions = [float(sentence) for sentence in predictions]
            evaluations["predictions"] = save_predictions
            
            self.save_evaluations(evaluations, file_path)

        mse = sklearn.metrics.mean_squared_error(targets, predictions)
        print(f"Mean Squared Error: {mse}")

        print(f"RMSE; {np.sqrt(mse)}")

        mae = sklearn.metrics.mean_absolute_error(targets, predictions)
        print(f"Mean Absolute Error: {mae}")

        if file_path:
            kendall_tau, p_value = scipy.stats.kendalltau(evaluations["labels"], evaluations["predictions"], variant='b')
            print(f"Kendall Tau: {kendall_tau} (p-value: {p_value})")

        if csv_file_path:
            self.analyze_predictions(predictions, targets, csv_file_path, csv_row_index)

        return loss
    
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

            features = self.get_features_from_sentence(sent)
            features = features.to_frame().transpose() # convert to dataframe and transpose to for features as columns

            features = torch.tensor(features.values, dtype=torch.float32).to(self.device)

            self.model.eval()

            with torch.no_grad():
                prediction = self.model(features)

            scores.append(float(prediction.item()))

        return np.mean(scores)
          
    def expand_nested_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        dataframe with nested series to flat dataframe.
        """

        expanded_df = pd.DataFrame()

        for col in df.columns:
            if isinstance(df[col].iloc[0], pd.Series):
                expanded_col = df[col].apply(pd.Series)
                expanded_df = pd.concat([expanded_df, expanded_col], axis=1)
            else:
                expanded_df = pd.concat([expanded_df, df[col]], axis=1)

        # Convert object columns to float
        for col in expanded_df.columns:
            if expanded_df[col].dtype == 'object':
                expanded_df[col] = pd.to_numeric(expanded_df[col], errors='coerce').astype(float)

        return expanded_df
    
    def ablation_study(self, eval_data:ARA_Dataset, batch_size:int = 20, num_epoch:int = 40, learning_rate:float = 0.01, early_stopping_patience:int = 15):
        print("Starting ablation study...")

        train_data_features = self.batch_feature_extraction(self.train_data, flatten=False)
        dev_data_features = self.batch_feature_extraction(self.dev_data, flatten=False)
        eval_data_features = self.batch_feature_extraction(eval_data, flatten=False)

        feature_list = eval_data_features.columns.tolist()
        feature_list.append("no_ablation")


        for feature in feature_list:
            print(f"Ablating feature: {feature}")
            
            train_data_ablated = train_data_features.drop(columns=[feature], errors='ignore')
            dev_data_ablated = dev_data_features.drop(columns=[feature], errors='ignore')
            eval_data_ablated = eval_data_features.drop(columns=[feature], errors='ignore')


            # Expand the nested series
            train_data_ablated = self.expand_nested_series(train_data_ablated)
            dev_data_ablated = self.expand_nested_series(dev_data_ablated)
            eval_data_ablated = self.expand_nested_series(eval_data_ablated)



            NN_train_data = NNDataset(train_data_ablated, self.train_data.target, self.train_data.sentences)
            NN_dev_data = NNDataset(dev_data_ablated, self.dev_data.target, self.dev_data.sentences)
            NN_eval_data = NNDataset(eval_data_ablated, eval_data.target, eval_data.sentences)

            compressed_input_size = 0
            for n in self.n_list:
                if feature != f"{n}_grams":
                    compressed_input_size += len(self.ngrams_baseline[n])


            total_input_size = eval_data_ablated.shape[1]
            non_compressed_input_size = total_input_size - compressed_input_size

            model = CompressedInputNN(
                compressed_input_size=compressed_input_size, 
                non_compressed_input_size=non_compressed_input_size
            )
            model = model.to(self.device)

            # Create data loaders
            train_loader = DataLoader(NN_train_data, batch_size=batch_size, shuffle=False)
            dev_loader = DataLoader(NN_dev_data, batch_size=batch_size, shuffle=False)

            # Create the optimizer
            optimizer = self.create_optimizer(model, learning_rate)

            criterion = nn.MSELoss()

            # Train the model
            model, optimizer, _ = self.train_model(model, optimizer, train_loader, dev_loader, criterion, num_epoch, early_stopping_patience)

            print("Eval data")
            self.eval_model(NN_eval_data, batch_size, model, criterion, csv_file_path="results/ARA/Syntax_model_ablation_study_eval.csv", csv_row_index=feature)



 

if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    spacy.util.fix_random_seed(seed)


    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")


    # read settings
    with open("ARA_Syntax_model_settings.json", "r") as settings_file:
        settings = json.load(settings_file)

    parser = argparse.ArgumentParser(description="ARA RF Model")

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
        train_data = ARA_Dataset(data=train_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        dev_data = ARA_model.import_jsonl(settings["dev_data"])
        dev_data = ARA_Dataset(data=dev_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])


        ARA_Syntax_Model_instance = ARA_Syntax_Model(
            spacy_model = settings["spacy_model"], 
            banned_pos_tags= settings["banned_pos_tags"], 
            train_data = train_data,
            dev_data= dev_data,
            n_list=settings["n_list"],
            batch_sizes=settings["batch_sizes"],
            num_epochs=settings["num_epochs"],
            learning_rates=settings["learning_rates"],
            early_stopping_patiences=settings["early_stopping_patiences"]
        )

        # Save the model
        ARA_Syntax_Model_instance.save_model(model_path)

        eval_data = ARA_model.import_jsonl(settings["eval_data"])
        eval_data = ARA_Dataset(data=eval_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        
        print("Train Data")
        ARA_Syntax_Model_instance.eval_model(train_data, file_path='predictions/ARA_Syntax_model_train.jsonl')
        
        print("\nDev data")
        ARA_Syntax_Model_instance.eval_model(dev_data, file_path="predictions/ARA_Syntax_model_dev.jsonl")
        
        print("\nEval data")
        ARA_Syntax_Model_instance.eval_model(eval_data, file_path="predictions/ARA_Syntax_model_eval.jsonl", csv_file_path="results/ARA/Syntax_model_eval.csv", csv_row_index="Syntax")

        ARA_Syntax_Model_instance.ablation_study(eval_data)

    elif args.mode == "eval":

        ARA_Syntax_Model_instance = ARA_Syntax_Model.from_saved(model_path)
        
        # print("Train Data")
        # train_data = ARA_model.import_jsonl(settings["train_data"])
        # train_data = ARA_Dataset(data=train_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        # ARA_Syntax_Model_instance.eval_model(train_data, file_path='predictions/ARA_Syntax_model_train.jsonl')

        # print("\nDev data")
        # dev_data = ARA_model.import_jsonl(settings["dev_data"])
        # dev_data = ARA_Dataset(data=dev_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        # ARA_Syntax_Model_instance.eval_model(dev_data, file_path="predictions/ARA_Syntax_model_dev.jsonl")

        # print("\nEval data")
        eval_data = ARA_model.import_jsonl(settings["eval_data"])
        eval_data = ARA_Dataset(data=eval_data, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])
        ARA_Syntax_Model_instance.eval_model(eval_data, file_path="predictions/ARA_Syntax_model_eval.jsonl", csv_file_path="results/ARA/Syntax_model_eval.csv", csv_row_index="Syntax")


    elif args.mode == "cli":
        # Load the model
        ARA_Syntax_Model_instance = ARA_Syntax_Model.from_saved(model_path)

        ARA_Syntax_Model_instance.cli_mode(args.input_sentence, "ARA_Syntax_model")

    elif args.mode == "inference-timing":
        ARA_Syntax_Model_instance = ARA_Syntax_Model.from_saved(model_path)

        test_sentences = ARA_model.import_jsonl(settings["eval_data"])
        test_dataset = ARA_Dataset(data=test_sentences, columns_to_keep=settings["columns_to_keep"], target_column=settings["target"])

        ARA_Syntax_Model_instance.measure_inference_time(test_dataset)