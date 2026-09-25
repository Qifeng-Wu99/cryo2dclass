"""cryo2dclass: unsupervised 2D classification of cryo-EM/ET particles.

A rotation-invariant autoencoder learns a low-dimensional embedding of every
particle while an exhaustive in-plane pose search (rotation, shift, mirror) is
built into the reconstruction loss. Particles are then clustered in the
embedding space.

Python API::

    from cryo2dclass import resolve_config, train, evaluate, visualize

    cfg = resolve_config(stage=1, overrides={"data": "particles.mrcs", "out": "runs/stage1"}, box_size=128)
    train(cfg)
"""

__version__ = "0.1.0"

__all__ = ["resolve_config", "TrainConfig", "train", "evaluate", "visualize", "__version__"]


def __getattr__(name):  # lazy imports keep `cryo2dclass --help` fast
    if name in ("resolve_config", "TrainConfig"):
        from . import config
        return getattr(config, name)
    if name == "train":
        from .train import train
        return train
    if name == "evaluate":
        from .evaluate import evaluate
        return evaluate
    if name == "visualize":
        from .plotting import visualize
        return visualize
    raise AttributeError(name)
