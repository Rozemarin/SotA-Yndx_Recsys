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
    def __init__(self, ratings_df=None, user_col='user_id', item_col='item_id', rating_col='rating', timestamp_col='timestamp'):
        self.ratings_df = ratings_df
        self.preprocessed_ratings_df = None
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    @abstractmethod
    def download_dataset(self):
        pass

    def preprocess_ratings_df(self, pcore=None, max_seq_len=None):
        self.preprocessed_ratings_df = self.ratings_df.copy()

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
            self.preprocessed_ratings_df = self.preprocessed_ratings_df.groupby(self.user_col).tail(max_seq_len)

    def train_test_split(self, train_val_test_split_method='timestamp', train_val_test_ratio=[0.7, 0.2, 0.1], random_state=42):
        assert round(sum(train_val_test_ratio)) == 1.0, "Train, val, test ratios must sum up to 1."

        if train_val_test_split_method == 'timestamp':
            train_df, val_df, test_df = self._train_val_test_split_by_timestamp(train_val_test_ratio, random_state=random_state)
        elif train_val_test_split_method == 'user':
            train_df, val_df, test_df = self._train_val_test_split_by_user(train_val_test_ratio, random_state=random_state)
        else:
            raise ValueError("Unknown train_val_test_split_method. Choose 'timestamp' or 'user'.")
            
        # labelencoding after splitting
        train_df[self.user_col] = self.user_encoder.fit_transform(train_df[self.user_col])
        train_df[self.item_col] = self.item_encoder.fit_transform(train_df[self.item_col])
        val_df[self.user_col] = self.user_encoder.transform(val_df[self.user_col])
        val_df[self.item_col] = self.item_encoder.transform(val_df[self.item_col])
        test_df[self.user_col] = self.user_encoder.transform(test_df[self.user_col])
        test_df[self.item_col] = self.item_encoder.transform(test_df[self.item_col])
        
        return train_df, val_df, test_df

    def _train_val_test_split_by_timestamp(self, train_val_test_ratio, random_state=42):
        ratings_df = self.ratings_df if self.preprocessed_ratings_df is None else self.preprocessed_ratings_df
        ratings_df = ratings_df.sort_values(by=self.timestamp_col)
        
        train_ratio, val_ratio, test_ratio = train_val_test_ratio
        train_df, test_val_df = train_test_split(ratings_df, test_size=(val_ratio+test_ratio), random_state=random_state, shuffle=False)
        train_users = train_df[self.user_col].unique()
        train_items = train_df[self.item_col].unique()
        test_val_df = test_val_df[
            test_val_df[self.user_col].isin(train_users) &
            test_val_df[self.item_col].isin(train_items)
        ]

        relative_test_size = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(test_val_df, test_size=relative_test_size, random_state=random_state, shuffle=False)
        return train_df, val_df, test_df

    def _train_val_test_split_by_user(self, train_val_test_ratio, random_state=42):
        ratings_df = self.ratings_df if self.preprocessed_ratings_df is None else self.preprocessed_ratings_df
        train_ratio, val_ratio, test_ratio = train_val_test_ratio

        train_users, val_test_users = train_test_split(ratings_df[self.user_col].unique(), test_size=(val_ratio + test_ratio))
 
        relative_test_size = test_ratio / (val_ratio + test_ratio)
        val_users, test_users = train_test_split(val_test_users, test_size=relative_test_size, random_state=random_state)

        train_df = ratings_df[ratings_df[self.user_col].isin(train_users)]
        val_df = ratings_df[ratings_df[self.user_col].isin(val_users)]
        test_df = ratings_df[ratings_df[self.user_col].isin(test_users)]

        return train_df, val_df, test_df


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
        elif self.dataset == 'ml-20m':
            self.ratings_df = pd.read_csv(
                f'{self.dataset}/ratings.csv',
                sep=","
            )
            self.ratings_df.rename(
                columns={"userId": "user_id", "movieId": "item_id"},
                inplace=True
            )


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
        
        
def get_train_val_test_from_dataset(dataset_name='ml-1m', pcore=5, max_seq_len=200, train_val_test_split_method='timestamp', train_val_test_ratio=[0.7, 0.2, 0.1], random_state=42):
    # Все итоговые датасеты вида [user_id: int, item_id: int, rating: int from 1 to 5, timestamp: int] 
    # Из val_df и test_df после создания выбрасываются все пользовтаели и айтемы, которые не вошли в train_df
    if dataset_name in ['ml-1m', 'ml-2m']:
        prep = PrepareMovielens(dataset='ml-1m')
    elif dataset_name in ['Books', 'Video_Games', 'Beauty']:
        prep = PrepareAmazon(dataset_name)
    else:
        raise ValueError('Unknown dataset')
    prep.download_dataset()
    prep.preprocess_ratings_df(pcore=pcore, max_seq_len=max_seq_len)
    return prep.train_test_split(train_val_test_split_method=train_val_test_split_method, train_val_test_ratio=train_val_test_ratio, random_state=random_state)  