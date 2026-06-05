import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class SWaT:
    def __init__(self, options):
        """
        Initialize the SWaT dataset processing class with the given options.

        Parameters:
        - options (dict): A dictionary containing keys such as 'seed', 'num_vars',
                          'data_dir', 'window_size', and 'shuffle'.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']
        self.fault_id = options.get('fault_id')

    def _load_fault_events(self):
        events_file = os.path.join(self.data_dir, 'fault_events.csv')
        if not os.path.exists(events_file):
            raise FileNotFoundError(
                "SWaT fault_events.csv is missing. Create it from List_of_attacks_Final.xlsx first."
            )
        events = pd.read_csv(events_file)
        events = events.dropna(subset=['start_time', 'end_time', 'attack_points', 'fault_id']).copy()
        events['start_time'] = pd.to_datetime(events['start_time'])
        events['end_time'] = pd.to_datetime(events['end_time'])
        if self.fault_id:
            events = events.loc[events['fault_id'] == self.fault_id].copy()
        if events.empty:
            raise ValueError(f"No SWaT attack events mapped to fault_id={self.fault_id!r}")
        return events

    @staticmethod
    def _normalise_attack_point(point):
        return str(point).replace('-', '').strip().upper()

    def generate_example(self):
        """
        Generate examples by loading, cleaning, and processing the SWaT dataset.
        This method loads the label, normal, and abnormal data files, performs
        necessary cleaning, scaling, and window slicing operations, and stores
        the processed arrays in self.data_dict.
        """
        # ----------------------------
        # Load Normal and Abnormal Data
        # ----------------------------
        normal_csv = os.path.join(self.data_dir, 'SWaT_Normal.csv')
        abnormal_csv = os.path.join(self.data_dir, 'SWaT_Abnormal.csv')
        normal_excel = os.path.join(self.data_dir, 'SWaT_Dataset_Normal_v1.xlsx')
        abnormal_excel = os.path.join(self.data_dir, 'SWaT_Dataset_Attack_v0.xlsx')

        if os.path.exists(normal_csv) and os.path.exists(abnormal_csv):
            df_normal = pd.read_csv(normal_csv, header=0, index_col=0)
            df_abnormal = pd.read_csv(abnormal_csv, header=0, index_col=0)
        else:
            if not (os.path.exists(normal_excel) and os.path.exists(abnormal_excel)):
                raise FileNotFoundError(
                    "SWaT raw time-series files are required for scenario-closed-loop generation. "
                    "Place SWaT_Normal.csv/SWaT_Abnormal.csv or the original Excel files in datasets/swat."
                )
            df_normal = pd.read_excel(normal_excel, header=1)
            df_normal.to_csv(normal_csv)
            df_abnormal = pd.read_excel(abnormal_excel, header=1)
            df_abnormal.to_csv(abnormal_csv)

        # ----------------------------
        # Load mapped fault events
        # ----------------------------
        fault_events = self._load_fault_events()

        # ----------------------------
        # Clean Normal Data
        # ----------------------------
        # Select only rows marked as 'Normal'
        df_normal = df_normal.loc[df_normal['Normal/Attack'] == 'Normal']
        # Drop unnecessary columns and downsample by taking every 10th row
        df_normal.drop(columns=[' Timestamp', 'Normal/Attack'], inplace=True)
        df_normal = df_normal[::10].reset_index(drop=True)

        # ----------------------------
        # Clean Abnormal Data
        # ----------------------------
        # Remove any rows with missing values and reset index
        df_abnormal.dropna(how='any', inplace=True)
        df_abnormal.reset_index(drop=True, inplace=True)
        # Initialize label matrix with zeros; columns from 1 to second-last column are used
        labels = np.zeros(df_abnormal.values[:, 1:-1].shape)
        # Convert the timestamp column to datetime using the given format, then standardize its format
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(
            df_abnormal[' Timestamp'], format=' %d/%m/%Y %I:%M:%S %p'
        ).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(df_abnormal['Adjusted Timestamp'])

        # ----------------------------
        # Create Column Dictionary for Abnormal Data
        # ----------------------------
        # Create a mapping from cleaned column names (without leading spaces) to their index
        col_dic = {}
        for i in df_abnormal.columns.values[1:-2]:
            col_dic[self._normalise_attack_point(i)] = len(col_dic)

        # ----------------------------
        # Process Each Attack Event for Abnormal Data
        # ----------------------------
        test_x_lst = []
        test_label_lst = []

        for _, event in fault_events.iterrows():
            lower = event['start_time']
            upper = event['end_time']
            attack_lst = [
                self._normalise_attack_point(point)
                for point in str(event['attack_points']).replace(',', '|').split('|')
                if str(point).strip()
            ]
            attack_lst_ind = [col_dic[p] for p in attack_lst if p in col_dic]
            if not attack_lst_ind:
                continue
            # Find indices in abnormal data where the timestamp is within the attack interval and marked as 'Attack'
            index_lst = np.array(df_abnormal.loc[
                (df_abnormal['Adjusted Timestamp'] >= lower) &
                (df_abnormal['Adjusted Timestamp'] <= upper) &
                (df_abnormal['Normal/Attack'] == 'Attack')
            ].index.values)
            if len(index_lst) > 0:
                # Mark the corresponding attack points in the label matrix as 1 for these indices
                for j in attack_lst_ind:
                    labels[index_lst, j] = 1
                # Define the window for the example based on the minimum index in the attack interval
                start_idx = max(0, int(min(index_lst) - 2 * 10 * self.window_size))
                end_idx = min(len(df_abnormal), int(min(index_lst) + 1 * 10 * self.window_size))
                if end_idx <= start_idx:
                    continue
                # Slice the abnormal data and label arrays with a step of 10
                test_x_lst.append(
                    df_abnormal.iloc[start_idx:end_idx:10, 1:-2].values
                )
                test_label_lst.append(
                    labels[start_idx:end_idx:10]
                )

        if not test_x_lst:
            raise ValueError(
                f"No SWaT abnormal windows could be built for fault_id={self.fault_id!r}. "
                "Check fault_events.csv and the raw SWaT attack timestamps."
            )

        # ----------------------------
        # Process Normal Data: Split and Scale
        # ----------------------------
        # Split normal data into segments of 1000 rows each, ensuring each segment has exactly 1000 rows
        x_n_list = [
            df_normal.iloc[i:i + 1000].values
            for i in range(0, len(df_normal), 1000)
            if i + 1000 < len(df_normal)
        ]
        if not x_n_list:
            raise ValueError("No SWaT normal training windows could be built from the raw normal file.")
        # Initialize and fit the StandardScaler on the concatenated normal data segments
        scaler = StandardScaler()
        scaler.fit(np.concatenate(x_n_list, axis=0))
        # Transform each segment of normal data
        x_n_list = [scaler.transform(segment) for segment in x_n_list]
        # Transform each abnormal example using the same scaler
        test_x_lst = [scaler.transform(example) for example in test_x_lst]

        # ----------------------------
        # Store Processed Data in data_dict
        # ----------------------------
        self.data_dict['x_n_list'] = np.array(x_n_list)
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(test_label_lst)

    def save_data(self):
        """
        Save the processed data arrays to .npy files in the data directory.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        """
        Load the processed data arrays from .npy files in the data directory into data_dict.
        """
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=False)
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=True)
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=True)
