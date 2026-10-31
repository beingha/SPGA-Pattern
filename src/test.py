from cdnsgen import DiffusionModel


def main():
    print("Initializing EXACT Discrete Diffusion Model for Testing...")

    detector = DiffusionModel(img_channel=16,
                              img_size=32,
                              model_path='D:\\wyz\\deepattern-master\\models\\iccad_kongjianxianyan',
                              fast_sample_m=5,
                              prior_strength=1.0)

    print("Starting Large-scale Generation...")
    detector.test(None)


if __name__ == "__main__":
    main()