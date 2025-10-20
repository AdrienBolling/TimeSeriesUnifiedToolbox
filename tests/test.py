from ray.tune import Trainable
from torch.nn import Module


class Test(Trainable, Module):
    def __init__(self):
        Trainable.__init__(self)
        Module.__init__(self)


t = Test()
print(Test.mro())
