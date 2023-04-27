from models import p2b, bat, m2track,ppe


def get_model(name):
    model = globals()[name.lower()].__getattribute__(name.upper())
    return model
