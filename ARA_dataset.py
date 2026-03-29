import pandas as pd
import torch


class ARA_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, columns_to_keep: list = ["label", "text"], target_column: str | None = None):
        """
        Initializes the dataset with the given data, target, and columns to keep.
        """
        
        # Convert the data to a DataFrame
        data = pd.DataFrame.from_dict(data, orient='index')
        
        # Extract the target column
        try:
            if target_column is None:
                target = data["label"]
            else:
                target = data[target_column]
        except:
            raise KeyError
        
        # Keep only the specified columns
        dfcols = data.columns
        columns_to_keep = [col for col in columns_to_keep if col in dfcols]      
        data = data[columns_to_keep]

        # Drop target column from features
        features = data.copy()
        try:
            features = features.drop(target_column, axis=1)
        except:
            pass   
        
        # Extract sentence texts
        try:
            sentences = data["text"]
        except:
            raise KeyError
  
        self.features = features
        self.target = target
        self.sentences = sentences
        self.columns_to_keep = columns_to_keep

    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: str) -> tuple:
        """
        Returns the features, target, and sentence for a given index.
        """
        return self.features[idx], self.target[idx], self.sentences[idx]
    
    def get_data(self) -> tuple:
        """
        Returns all features and targets of the dataset.
        """
        return self.features, self.target, self.sentences
    
    def get_features(self) -> pd.DataFrame:
        """
        Returns all features of the dataset.
        """
        return self.features
    
    def get_targets(self) -> pd.Series:
        """
        Returns all targets of the dataset.
        """
        return self.target
    
    def get_sentences(self) -> pd.Series:
        """
        Returns all sentences of the dataset.
        """
        return self.sentences