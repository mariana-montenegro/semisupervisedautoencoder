from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class DataLoader:
    @staticmethod
    def load():
        data = pd.read_excel('ML_final_results\data\merged_data.xlsx')
        data = data.drop(['ID','Sample','Old','Label Old','Type of Episode','FEP','Label', 'Label New'], axis=1)
        target_labels = ['CTR'] 
        x = data.drop(['new'], axis=1).reset_index(drop=True)
        y = (data['new'].isin(target_labels)).astype(int).reset_index(drop=True)
        y = 1 - y
        return x, y
    
    @staticmethod
    def split(x, y, split_size=0.3):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size, random_state=42, stratify=y)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def pre_process_data(x):
        scaler = StandardScaler().fit(x)
        x_std = scaler.transform(x)

        return x_std