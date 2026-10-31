import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ICCAD:
    def __init__(self):
        self.scale_delta = 600.0
        self.getData()
        self.test_batch_pointer = 0

    def printExactPattern(self, df_row):
        """
        真实的物理拉伸函数。
        由于加入了虚拟扫描线 (\Delta=0)，这部分物理还原时，
        0 宽度的块会自动消失，完美展现原始版图。
        """
        m = int(df_row['cX']) # 现在永远是 128
        n = int(df_row['cY']) # 现在永远是 128

        topo = df_row['topoSig']
        if isinstance(topo, bytes):
            topo = topo.decode('utf-8')

        Imarr = np.array(list(topo)).astype(int)

        if Imarr.max() > 1:
            Imarr[Imarr == 48] = 0
            Imarr[Imarr == 49] = 1

        Imarr = Imarr.reshape((n, m), order='C')

        Delarr_x = np.flipud(np.array(df_row['yDelta'].split(' ')).astype(int))
        Delarr_y = np.array(df_row['xDelta'].split(' ')).astype(int)

        b = np.tile(Imarr[0, :], (Delarr_x[0], 1))
        for i in np.arange(1, len(Delarr_x)):
            b = np.concatenate((b, np.tile(Imarr[i, :], (Delarr_x[i], 1))), axis=0)

        c = np.tile(b[:, 0], (Delarr_y[0], 1)).T
        for i in np.arange(1, len(Delarr_y)):
            c = np.concatenate((c, np.tile(b[:, i], (Delarr_y[i], 1)).T), axis=1)

        return c

    def squishImage(self, df_row):
        """
        纯粹的 Deep Squish Pattern 折叠器：
        128x128 矩阵 -> 切割为 32x32 个 4x4 块 -> 折叠通道生成 16x32x32 张量
        """
        m = int(df_row['cX'])
        n = int(df_row['cY'])

        topo_data = df_row['topoSig']
        if isinstance(topo_data, bytes):
            topo_data = topo_data.decode('utf-8')

        Imarr = np.array(list(topo_data)).astype(np.float32)

        if Imarr.max() > 1:
            Imarr[Imarr == 48] = 0
            Imarr[Imarr == 49] = 1

        # 1. 还原 128x128 矩阵
        temp_im = Imarr.reshape((n, m), order='C')

        # 2. 张量折叠 (Folding) [128, 128] -> [16, 32, 32]
        folded_tensor = temp_im.reshape(32, 4, 32, 4)
        folded_tensor = folded_tensor.transpose(1, 3, 0, 2)
        folded_tensor = folded_tensor.reshape(16, 32, 32)

        return folded_tensor

    def getData(self):
        train_path = './data/iccad16_fake/train.pkl'
        test_path = './data/iccad16_fake/test.pkl'

        print(f"Loading ORIGINAL generative dataset from {train_path}...")
        try:
            self.df_train = pd.read_pickle(train_path)
            self.train_data = np.stack([self.squishImage(self.df_train.iloc[i]) for i in range(len(self.df_train))])
        except Exception as e:
            print(f"Error loading train data: {e}")
            self.train_data = np.zeros((10, 16, 32, 32))

        if os.path.exists(test_path):
            self.df_test = pd.read_pickle(test_path)
            self.test_data = np.stack([self.squishImage(self.df_test.iloc[i]) for i in range(len(self.df_test))])
        else:
            print("No test.pkl found, mirroring train_data.")
            self.test_data = self.train_data

        self.train_hs_length = len(self.train_data)
        self.test_hs_length = len(self.test_data)
        print(f"Data Loaded: Train patterns: {self.train_hs_length}, Test patterns: {self.test_hs_length}")
        print(f"Data Shape: {self.train_data.shape[1:]} (Expected: 16, 32, 32)")

    def get_train_batch(self, half_batch_size, mode='detection', balance=False):
        batch_size = half_batch_size * 2
        idx = np.random.choice(self.train_hs_length, batch_size, replace=True)
        batch = self.train_data[idx]
        return batch, np.zeros(batch_size)

    def get_test_batch(self, batch_size=300, mode='detection', dtype='hs'):
        idx = np.random.choice(self.test_hs_length, batch_size, replace=True)
        return self.test_data[idx]