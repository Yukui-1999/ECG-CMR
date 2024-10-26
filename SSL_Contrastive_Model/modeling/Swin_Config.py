class Config_base:
    class DATA:
        IMG_SIZE = 224  # 图像大小
    class MODEL:
        NUM_CLASSES = 256  # 分类数
        class SWIN:
            PATCH_SIZE = 4
            IN_CHANS = 50
            EMBED_DIM = 128
            DEPTHS = [2, 2, 18, 2]
            NUM_HEADS = [4, 8, 16, 32]
            WINDOW_SIZE = 7
            MLP_RATIO = 4.0
            QKV_BIAS = True
            QK_SCALE = None
            APE = False
            PATCH_NORM = True
        DROP_RATE = 0.1
        DROP_PATH_RATE = 0.1
    class TRAIN:
        USE_CHECKPOINT = False


class Config_base_win14:
    class DATA:
        IMG_SIZE = 224  # 图像大小
    class MODEL:
        NUM_CLASSES = 256  # 分类数
        class SWIN:
            PATCH_SIZE = 4
            IN_CHANS = 50
            EMBED_DIM = 128
            DEPTHS = [2, 2, 18, 2]
            NUM_HEADS = [4, 8, 16, 32]
            WINDOW_SIZE = 14
            MLP_RATIO = 4.0
            QKV_BIAS = True
            QK_SCALE = None
            APE = False
            PATCH_NORM = True
        DROP_RATE = 0.1
        DROP_PATH_RATE = 0.1
    class TRAIN:
        USE_CHECKPOINT = False

class Config_large:
    class DATA:
        IMG_SIZE = 224  # 图像大小
    class MODEL:
        NUM_CLASSES = 256  # 分类数
        class SWIN:
            PATCH_SIZE = 4
            IN_CHANS = 50
            EMBED_DIM = 192
            DEPTHS = [2, 2, 18, 2]
            NUM_HEADS = [6, 12, 24, 48]
            WINDOW_SIZE = 14
            MLP_RATIO = 4.0
            QKV_BIAS = True
            QK_SCALE = None
            APE = False
            PATCH_NORM = True
        DROP_RATE = 0.1
        DROP_PATH_RATE = 0.1
    class TRAIN:
        USE_CHECKPOINT = False


def build_swin_config(config_name):
    if config_name == 'swin_base':
        return Config_base()
    elif config_name == 'swin_base_win14':
        return Config_base_win14()
    elif config_name == 'swin_large':
        return Config_large()
    else:
        raise ValueError(f"config_name {config_name} not supported")