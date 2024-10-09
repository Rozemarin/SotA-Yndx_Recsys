import os
import shutil
import pandas as pd
import zipfile
import requests
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod


class BaseDatasetPreparer(ABC):
    def __init__(self, ratings_df=None, users_df=None, items_df=None, user_col='user_id', item_col='item_id', rating_col='rating', timestamp_col='timestamp'):
        self.ratings_df = ratings_df
        self.users_df = users_df
        self.items_df = items_df
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.preprocessed_ratings_df = None
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
    
    @abstractmethod
    def download_dataset(self):
        """Method to download and load the dataset."""
        pass
    
    def _encode_ids(self):
        """Encodes user and item IDs using LabelEncoders."""
        if self.users_df is not None:
            self.user_encoder.fit(self.users_df[self.user_col])
        else:
            self.user_encoder.fit(self.ratings_df[self.user_col])
        if self.items_df is not None:
            self.item_encoder.fit(self.items_df[self.item_col])
        else:
            self.item_encoder.fit(self.ratings_df[self.item_col])
            
        self.preprocessed_ratings_df[self.user_col] = self.user_encoder.transform(self.preprocessed_ratings_df[self.user_col])
        self.preprocessed_ratings_df[self.item_col] = self.item_encoder.transform(self.preprocessed_ratings_df[self.item_col])
    
    def preprocess_ratings_df(self, pcore=None, max_seq_len=None):
        """Filter out users and items with fewer than pcore interactions. Truncate to first max_seq_len interactions."""
        self.preprocessed_ratings_df = self.ratings_df.copy()
        self._encode_ids()
        
        if pcore is not None:
            while True:
                item_counts = self.preprocessed_ratings_df[self.item_col].value_counts()
                self.preprocessed_ratings_df = self.preprocessed_ratings_df[self.preprocessed_ratings_df[self.item_col].isin(item_counts[item_counts >= pcore].index)]
                user_counts = self.preprocessed_ratings_df[self.user_col].value_counts()
                self.preprocessed_ratings_df = self.preprocessed_ratings_df[self.preprocessed_ratings_df[self.user_col].isin(user_counts[user_counts >= pcore].index)]
                if len(user_counts[user_counts < pcore]) == 0 and len(item_counts[item_counts < pcore]) == 0:
                    break
                    
        if max_seq_len is not None:
            if self.timestamp_col:
                self.preprocessed_ratings_df = self.preprocessed_ratings_df.sort_values(by=[self.user_col, self.timestamp_col])
            else:
                self.preprocessed_ratings_df = self.preprocessed_ratings_df.sort_values(by=[self.user_col])
            self.preprocessed_ratings_df = self.preprocessed_ratings_df.groupby(self.user_col).head(max_seq_len)
            
    def train_test_split(self, method='timestamp', test_size=0.2):
        """Split data into train and test using the selected method."""
        if method == 'timestamp':
            return self._train_test_split_by_timestamp(test_size)
        elif method == 'user':
            return self._train_test_split_by_user(test_size)
        else:
            raise ValueError("Unknown method for train_test_split. Choose 'timestamp' or 'user'.")
    
    def _train_test_split_by_timestamp(self, test_size):
        """Split by timestamp: oldest data is for training, newest data is for testing."""
        ratings_df = self.ratings_df if self.preprocessed_ratings_df is None else self.preprocessed_ratings_df
        ratings_df = ratings_df.sort_values(by=self.timestamp_col)
        train_df, test_df = train_test_split(ratings_df, test_size=test_size, shuffle=False, stratify=None)
        return train_df, test_df

    def _train_test_split_by_user(self, test_size):
        """Split by user: some users go into test set, the rest into train."""
        ratings_df = self.ratings_df if self.preprocessed_ratings_df is None else self.preprocessed_ratings_df
        train_users, test_users = train_test_split(ratings_df[self.user_col].unique(), test_size=test_size)
        train_df = ratings_df[ratings_df[self.user_col].isin(train_users)]
        test_df = ratings_df[ratings_df[self.user_col].isin(test_users)]
        return train_df, test_df


class PrepareMovielens(BaseDatasetPreparer):
    def __init__(self, dataset='ml-1m'):
        super().__init__()
        self.dataset = dataset

    def download_dataset(self):
        dataset_urls = {
            'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'ml-20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
        }
        
        if self.dataset not in dataset_urls:
            raise ValueError(f"Dataset {self.dataset} not supported. Choose 'ml-1m' or 'ml-20m'.")
        
        url = dataset_urls[self.dataset]
        zip_filename = f'{self.dataset}.zip'
        
        if not os.path.exists(zip_filename):
            print(f'Downloading {self.dataset}...')
            response = requests.get(url)
            with open(zip_filename, 'wb') as f:
                f.write(response.content)
        
        extract_dir = self.dataset
        
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Handle extra folder level
        extracted_folder = os.path.join(extract_dir, self.dataset)
        if os.path.exists(extracted_folder):
            for file_name in os.listdir(extracted_folder):
                src_path = os.path.join(extracted_folder, file_name)
                dest_path = os.path.join(extract_dir, file_name)
                
                if not os.path.exists(dest_path):
                    shutil.move(src_path, dest_path)
            shutil.rmtree(extracted_folder)
        
        # Load dataset
        if self.dataset == 'ml-1m':
            self.users_df = pd.read_csv(
                f'{self.dataset}/users.dat',
                engine='python',
                sep="::",
                names=["user_id", "sex", "age_group", "occupation", "zip_code"],
            )
            self.ratings_df = pd.read_csv(
                f'{self.dataset}/ratings.dat',
                engine='python',
                sep="::",
                names=["user_id", "item_id", "rating", "timestamp"],
            )
            self.items_df = pd.read_csv(
                f'{self.dataset}/movies.dat',
                engine='python',
                sep="::",
                names=["item_id", "title", "genres"],
                encoding="iso-8859-1",
            )
        elif self.dataset == 'ml-20m':
            self.ratings_df = pd.read_csv(
                f'{self.dataset}/ratings.csv',
                sep=","
            )
            self.ratings_df.rename(
                columns={"userId": "user_id", "movieId": "item_id"},
                inplace=True
            )
            self.items_df = pd.read_csv(
                f'{self.dataset}/movies.csv',
                sep=",",
                encoding="iso-8859-1",
            )
            self.items_df.rename(columns={"movieId": "movie_id"}, inplace=True)


class PrepareAmazon(BaseDatasetPreparer):
    def __init__(self, category='Books'):
        super().__init__()
        self.category = category
        self.dataset_urls = {
            'Books': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv',
            'Video_Games': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv',
            'Beauty': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv',
        }
        self._saved_name = f'{self.category}_ratings.csv'
        
    def download_dataset(self):
        if self.category not in self.dataset_urls:
            raise ValueError(f"Category {self.category} not supported. Choose from {list(self.dataset_urls.keys())}.")
        
        url = self.dataset_urls[self.category]
        
        if not os.path.exists(self._saved_name):
            print(f'Downloading {self.category} dataset...')
            urlretrieve(url, self._saved_name)
        
        self.ratings_df = pd.read_csv(
            self._saved_name,
            sep=",",
            names=["user_id", "item_id", "rating", "timestamp"],
            header=0
        )
        
        
# prep = PrepareMovielens(dataset='ml-1m')
# prep.download_dataset()
# prep.preprocess_ratings_df(pcore=5, max_seq_len=100)
# train_df, test_df = prep.train_test_split(method='user', test_size=0.2)