

class AbstractTrainer:
    def __init__(self,*args,**kwargs):
        self._net = kwargs["net"]
        self._opt = kwargs["opt"]

