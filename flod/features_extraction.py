# Dato un dataframe, date delle dimensioni di sliding window per ogni feature



# definire funzione per ogni feature che raccoglie dimensione sliding window da dizionario costruttore. Altro argomento è variabile se usare dati normali o filtered

import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np

class SisFallFeaturesExtraction():

    dataframe = None

    def __init__(self, path: str, fall_begin_sample: int, fall_end_sample: int):
        
        # Read dataframe and keep only the first triaxial accellerometer data
        self.dataframe = pd.read_csv(path, header=None).iloc[:,:3]
        self.dataframe.columns = ['x', 'y', 'z']

        # Convert accellerometer data in accelleration (expressed in val * g, g stands for gravity force)
        self.dataframe['x'] = self.dataframe['x'] * (32/2**13)
        self.dataframe['y'] = self.dataframe['y'] * (32/2**13)
        self.dataframe['z'] = self.dataframe['z'] * (32/2**13)

        # Compute filtered columns with the filter spec given in SisFall paper
        self.dataframe['fx'] = SisFallFeaturesExtraction._filter_column(self.dataframe['x'])
        self.dataframe['fy'] = SisFallFeaturesExtraction._filter_column(self.dataframe['y'])
        self.dataframe['fz'] = SisFallFeaturesExtraction._filter_column(self.dataframe['z'])

        # Compute some basic properties of the samples
        self._sum_vector_magnitude()
        self._sum_vector_magnitude(True)
        self._sum_vector_magnitude_horizontal()
        self._sum_vector_magnitude_horizontal(True)

        self._label_data(fall_begin_sample, fall_end_sample)

    @staticmethod
    def _filter_column(column: pd.Series) -> pd.Series:
        # Filter order
        order = 4
        # Cutoff frequency
        cutoff = 5
        # Accellerometer (aka samples) original frequency
        fs = 200
        nyq = 0.5 * fs

        normal_cutoff = cutoff / nyq

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, column)


    def _sum_vector_magnitude(self, filtered: bool = False) -> pd.Series:
        col_name = 'f_svm' if filtered else 'svm'
        l_x, l_y, l_z =  ('fx', 'fy', 'fz') if filtered else ('x', 'y', 'z')

        x, y, z = self.dataframe[l_x], self.dataframe[l_y], self.dataframe[l_z]

        # Sums power of vector items and squares it, foreach vector
        self.dataframe[col_name] = [np.sqrt(np.sum(np.array([x[i], y[i], z[i]]) ** 2)) for i in range(len(x))]

        return self.dataframe[col_name]

    def _sum_vector_magnitude_horizontal(self, filtered: bool = False) -> pd.Series:
        col_name = 'f_svmh' if filtered else 'svmh'
        l_x, l_y, l_z =  ('fx', 'fy', 'fz') if filtered else ('x', 'y', 'z')

        x, z = self.dataframe[l_x], self.dataframe[l_z]

        # Sums power of vector items and squares it, foreach vector
        self.dataframe[col_name] = [np.sqrt(np.sum(np.array([x[i], z[i]]) ** 2)) for i in range(len(x))]

        return self.dataframe[col_name]

    def _window(self, k: int, size:int):
        begin = k - size
        if begin < 0 or k >= len(self.dataframe):
            return self.dataframe.iloc[0:0]
        return self.dataframe.iloc[begin:k]

    def c1(self, Nv: int, filtered: bool = False):
        dest_col = 'f_c1' if filtered else 'c1'
        src_col = 'f_svm' if filtered else 'svm'

        self.dataframe[dest_col] = 0

        k = Nv
        
        w = self._window(k, Nv)
        while not w.empty:
            rms = np.sqrt(np.mean(np.array(w[src_col]**2)))
            self.dataframe.iloc[k, self.dataframe.columns.get_loc(dest_col)] = rms
            k += 1
            w = self._window(k, Nv)

    def c2(self, Nv: int, filtered: bool = False):
        dest_col = 'f_c2' if filtered else 'c2'
        src_col = 'f_svmh' if filtered else 'svmh'

        self.dataframe[dest_col] = 0

        k = Nv
        
        w = self._window(k, Nv)
        while not w.empty:
            rms = np.sqrt(np.mean(np.array(w[src_col]**2)))
            self.dataframe.iloc[k, self.dataframe.columns.get_loc(dest_col)] = rms
            k += 1
            w = self._window(k, Nv)

    def c3(self, Nv: int, filtered: bool = False):
        dest_col = 'f_c3' if filtered else 'c3'
        src_col = 'f_svm' if filtered else 'svm'

        self.dataframe[dest_col] = 0

        k = Nv
        
        w = self._window(k, Nv)
        while not w.empty:
            ix_src_col = w.columns.get_loc(src_col)
            ix_dst_col = w.columns.get_loc(dest_col)
            ix_min = w[src_col].argmin()
            ix_max = w[src_col].argmax()
            ptp = w.iloc[ix_max, ix_src_col] - w.iloc[ix_min, ix_src_col]
            self.dataframe.iloc[k, ix_dst_col] = ptp
            k += 1
            w = self._window(k, Nv)

    # This window feature is computed only on its last sample
    def c4(self, Nv: int, filtered: bool = False):
        dest_col = 'f_c4' if filtered else 'c4'
        src_cols = ['fx', 'fy', 'fz'] if filtered else ['x', 'y', 'z']

        self.dataframe[dest_col] = 0

        k = Nv

        w = self._window(k, Nv)
        while not w.empty:
            latest_sample = np.array(w[src_cols].iloc[-1,:])
            angle = np.arctan2(np.sqrt(latest_sample[0]**2 + latest_sample[2]**2), -latest_sample[1])

            self.dataframe.iloc[k, self.dataframe.columns.get_loc(dest_col)] = angle
            k += 1
            w = self._window(k, Nv)

    def _label_data(self, fall_begin_sample: int, fall_end_sample: int):
        # Labels the data without caring about sliding windows

        self.dataframe['is_fall'] = 0

        is_fall_ix = self.dataframe.columns.get_loc('is_fall')

        self.dataframe.iloc[fall_begin_sample : fall_end_sample+1, is_fall_ix] = 1