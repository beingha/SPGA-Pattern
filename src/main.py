from cdnsgen import DiffusionModel
import pandas as pd
import numpy as np
import msgpack


class NativeDataLoaderAdapter:
    def __init__(self, msgpack_path):
        with open(msgpack_path, 'rb') as f:
            data = msgpack.unpackb(f.read(), raw=False)
        self.df = pd.DataFrame(data)
        self.num_samples = len(self.df)
        print(f" 正在解析 {self.num_samples} 条官方原始数据...")

        all_folded = []
        target_size = 128
        dropped_count = 0

        for idx in range(self.num_samples):
            row = self.df.iloc[idx]
            topo_str = row['topoSig']

            try:
                cX = int(row['cX'])
                cY = int(row['cY'])
            except (KeyError, ValueError):
                dropped_count += 1
                continue

            if cX * cY != len(topo_str):
                dropped_count += 1
                continue
            if cX > target_size or cY > target_size:
                dropped_count += 1
                continue
            if cX == 0 or cY == 0:
                dropped_count += 1
                continue

            topo_arr = np.array([int(c) for c in topo_str], dtype=np.int64).reshape(cY, cX)

            #  数据增强 1：随机翻转
            if np.random.rand() > 0.5:
                topo_arr = np.flipud(topo_arr)
            if np.random.rand() > 0.5:
                topo_arr = np.fliplr(topo_arr)

             # [Stage 1] 随机 90° 旋转（奇数次旋转会交换 H/W，需同步交换 cX/cY）
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                topo_arr = np.rot90(topo_arr, k=k)
                if k % 2 == 1:
                    cX, cY = cY, cX

            #  数据增强 2：居中填充 + 微小抖动
            base_pad_top = (target_size - cY) // 2
            base_pad_left = (target_size - cX) // 2

            jitter_y = np.random.randint(-2, 3)
            jitter_x = np.random.randint(-2, 3)

            pad_top = max(0, min(target_size - cY, base_pad_top + jitter_y))
            pad_bottom = target_size - cY - pad_top
            pad_left = max(0, min(target_size - cX, base_pad_left + jitter_x))
            pad_right = target_size - cX - pad_left

            padded_topo = np.pad(topo_arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                                 constant_values=0)

            folded = padded_topo.reshape(32, 4, 32, 4).transpose(1, 3, 0, 2).reshape(16, 32, 32)
            all_folded.append(folded)

        self.data_cache = np.stack(all_folded, axis=0)
        self.valid_samples = len(self.data_cache)
        print(f" 数据清洗完毕！共拦截 {dropped_count} 条数据。")
        print(f" 成功加载 {self.valid_samples} 条数据，形状: {self.data_cache.shape}")

    def getTrainBatchDP(self, batch_size):
        indices = np.random.choice(self.valid_samples, batch_size, replace=True)
        return self.data_cache[indices]


def main():
    print("Initializing Native Data Adapter...")
    data = NativeDataLoaderAdapter('D:\\wyz\\deepattern-master\\data\\iccad16\\train.msgpack')

    print("Initializing EXACT Discrete Diffusion Model...")
    detector = DiffusionModel(img_channel=16,
                              img_size=32,
                              batch_size=128,
                              init_lr=2e-4,
                              max_iter=500000,
                              sv_itr=10000,
                              model_path='D:\\wyz\\deepattern-master\\models\\iccad_kongjianxianyan',
                              prior_strength=1.0)

    print("Starting Training...")
    # 新实验目录首次训练必须从头开始，避免旧 checkpoint 与新空间先验混用
    detector.run(data, resume=False)


if __name__ == "__main__":
    main()