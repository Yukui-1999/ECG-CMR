from .registry import Registry, build_from_config

def build_func(cfg, registry, **kwargs):
    """
    Except for config, if passing a list of dataset config, then return the concat type of it
    """
    return build_from_config(cfg, registry, **kwargs)

AUTO_ENCODER = Registry("AUTO_ENCODER", build_func=build_func)
DATASETS = Registry("DATASETS", build_func=build_func)
DIFFUSION = Registry("DIFFUSION", build_func=build_func)
DISTRIBUTION = Registry("DISTRIBUTION", build_func=build_func)
EMBEDDER = Registry("EMBEDDER", build_func=build_func)
ENGINE = Registry("ENGINE", build_func=build_func)
INFER_ENGINE = Registry("INFER_ENGINE", build_func=build_func)
MODEL = Registry("MODEL", build_func=build_func)
PRETRAIN = Registry("PRETRAIN", build_func=build_func)
VISUAL = Registry("VISUAL", build_func=build_func)
EMBEDMANAGER = Registry("EMBEDMANAGER", build_func=build_func)
ECGCLIP = Registry("ECGCLIP", build_func=build_func)
ECGCLIPsa = Registry("ECGCLIPsa", build_func=build_func)
ECGCMRDATASET = Registry("ECGCMRDATASET", build_func=build_func)
ECGCMRDATASET_ECGlaCMR = Registry("ECGCMRDATASET_ECGlaCMR", build_func=build_func)
ECGCMRDATASET_ECGCMR = Registry("ECGCMRDATASET_ECGCMR", build_func=build_func)
ECGCMRDATASET_ECGlaCMRnew = Registry("ECGCMRDATASET_ECGlaCMRnew", build_func=build_func)
ECGCMRDATASET_UKB_SMALL = Registry("ECGCMRDATASET_UKB_SMALL", build_func=build_func)
ECGCMRDATASET_zheyi = Registry("ECGCMRDATASET_zheyi", build_func=build_func)