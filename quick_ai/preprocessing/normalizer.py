from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from ..base.preprocess import Preprocessor
from sklearn.preprocessing import MinMaxScaler


class Normalizer(Preprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def fit(self, data: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        data=data.apply(lambda x: x.astype(bool) if x.isin([0,1]).all() else x)
        normalized_numeric_cols = pd.DataFrame(self.scaler.fit_transform(data.select_dtypes(include=[np.number])),columns=self.scaler.get_feature_names_out())
        data = data.apply(lambda x: normalized_numeric_cols[x.name] if x.name in normalized_numeric_cols.columns else x)

        if target.isin([0,1]).all():
            target  = target.astype(bool)
        if np.issubdtype(target.dtype, np.number):
            target = pd.Series(self.target_scaler.fit_transform(np.transpose([target]))[:,0])
        return data, target

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data=data.apply(lambda x: x.astype(bool) if x.isin([0,1]).all() else x)
        normalized_numeric_cols = pd.DataFrame(
            self.scaler.transform(data.select_dtypes(include=[np.number])),
            columns=self.scaler.get_feature_names_out(),
        )
        return data.apply(
            lambda x: normalized_numeric_cols[x.name] if x.name in normalized_numeric_cols.columns else x
        )

    def inverse(self, data: pd.DataFrame) -> pd.DataFrame:
        data=data.apply(lambda x: x.astype(bool) if x.isin([0,1]).all() else x)
        normalized_numeric_cols = pd.DataFrame(
            self.scaler.inverse_transform(data.select_dtypes(include=[np.number])),
            columns=self.scaler.get_feature_names_out(),
        )
        return data.apply(
            lambda x: normalized_numeric_cols[x.name] if x.name in normalized_numeric_cols.columns else x
        )

    def preprocess_target(self, target: pd.Series) -> pd.Series:
        if target.isin([0,1]).all():
            target  = target.astype(bool)
        if np.issubdtype(target.dtype, np.number):
            target = self.target_scaler.transform(np.transpose([target]))
        return target

    def inverse_target(self, target: pd.Series) -> pd.Series:
        if target.isin([0,1]).all():
            target  = target.astype(bool)
        if np.issubdtype(target.dtype, np.number):
            target = self.target_scaler.inverse_transform(np.transpose([target]))
        return np.transpose(target)[0]


def main():

    dataset = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40],
        'C': ["a", "b", "c", "d"],
        'D': [0, 1, 0, 1],
        'E': [1,0.2,"c",0],
        'target': [45, 22, 69, 18],
    })
    data = dataset.drop(columns=['target'])
    target = dataset['target']
    print("Before Normalization:")
    print(data)
    print(target)
    print()
    normalizer = Normalizer()
    data, target = normalizer.fit(data, target)
    print("After Normalization:")
    print(data)
    print(target)
    print()
    data = pd.DataFrame({
        'A': [1.5, 2.5, 3.5, 4.5],
        'B': [11, 21, 31, 41],
        'C': ["a", "b", "c", "d"],
        'D': [0, 1, 0, 1],
        'E': [1,0.2,"c",0],
    })
    data = normalizer.preprocess(data)
    print("Normalizing more data:")
    print(data)
    print()
    target = normalizer.inverse_target(target)
    print("Inverse Target:")
    print(target)
    print("Equal to the original: ",all(target == [45, 22, 69, 18]))


if __name__ == '__main__':
    main()
