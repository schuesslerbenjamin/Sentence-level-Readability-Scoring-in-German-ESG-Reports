import torch
import pandas as pd


class NNDataset(torch.utils.data.Dataset):
    def __init__(self, features, target, sentences, indices = None):
        """
        Initializes the dataset with features and target.
        """

        if isinstance(features, pd.DataFrame):
            self.indices = features.index
            features = features.values
            self.features = torch.tensor(features, dtype=torch.float32)
        
        elif isinstance(features, torch.Tensor): 
            if indices is None:
                raise ValueError
            self.indices = indices          
            self.features = features.detach().clone()

        else:
            raise ValueError
        
        
        if isinstance(target, pd.Series):
            target = target.values
            self.target = torch.tensor(target, dtype=torch.float32)

        elif isinstance(target, torch.Tensor):
            self.target = target.detach().clone()
        
        else:
            raise ValueError

        if isinstance(sentences, pd.Series):
            self.sentences = sentences
        else:
            raise ValueError

        
        if len(self.features) != len(self.target) or len(self.features) != len(self.sentences):
            raise ValueError
            

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Returns the features and target for a given index.
        """

        if isinstance(idx, str):
            int_idx = self.indices.get_loc(idx)
        elif isinstance(idx, int):
            int_idx = idx
        else:
            raise ValueError
        
        if isinstance(self.features, torch.Tensor):
            features = self.features[int_idx]
        elif isinstance(self.features, pd.DataFrame):
            features = self.features.iloc[int_idx]
        else:
            raise ValueError
        
        if isinstance(self.target, torch.Tensor):
            target = self.target[int_idx]
        elif isinstance(self.target, pd.Series):
            target = self.target.iloc[int_idx]
        else:
            raise ValueError
        
        if isinstance(self.sentences, pd.Series):
            sentence = self.sentences.iloc[int_idx]
        else:
            raise ValueError
        
        return features, target, sentence
        
    def get_data(self):
        """
        Returns the features and target of the dataset.
        """
        return self.features, self.target, self.sentences
    
    def get_features_by_idx(self, idx):
        """
        Returns the features for a given list of indices.
        """
        if isinstance(self.features, torch.Tensor):
            return self.features[idx]
        elif isinstance(self.features, pd.DataFrame):
            return self.features.iloc[idx]
        else:
            raise ValueError
    
    def get_targets_by_idx(self, idx):
        """
        Returns the targets for a given list of indices.
        """
        if isinstance(self.target, torch.Tensor):
            return self.target[idx]
        elif isinstance(self.target, pd.Series):
            return self.target.iloc[idx]
        else:
            raise ValueError
        
    def get_sentences_by_idx(self, idx):
        """
        Returns the sentences for a given list of indices.
        """
        if isinstance(self.sentences, pd.Series):
            return self.sentences.iloc[idx]
        else:
            raise ValueError("Sentences must be a pandas Series.")

    
    def get_num_features(self):
        """
        Returns the number of features in the dataset.
        """
        return self.features.size(1)