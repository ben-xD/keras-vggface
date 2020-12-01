__version__ = '0.6'

def pretty_versions():
    import tensorflow as tf
    t_version = tf.__version__
    return "keras-vggface : {}, tensorflow : {} ".format(__version__,t_version)