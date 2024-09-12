from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class DataLoader:
    @staticmethod
    def load():
        data = pd.read_excel('UL_final_results\data\merged_data.xlsx')
        data = data.drop(['ID','Sample','Old','Label Old','Type of Episode','FEP','Label','new'], axis=1)
        
        patients_data = data[data['Label New'] != 0].reset_index(drop=True)
        x = patients_data.drop(['Label New'], axis=1)
        y = patients_data['Label New'].reset_index(drop=True)
        print(y.value_counts())

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