class AbstractService:
    def __init__(self, *args, **kwargs):
        self._ser_name = kwargs["name"]
        self._ser_type = kwargs["type"]

    @classmethod
    def create(cls, **kwargs):
        raise NotImplemented

    @property
    def service_name(self) -> str:
        return self._ser_name

    @property
    def service_type(self) -> str:
        return self._ser_type
