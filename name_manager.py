class NameManager:
    _counts = {}

    @staticmethod
    def reset():
        NameManager._counts = {}

    @staticmethod
    def _count(name):
        if name not in NameManager._counts:
            NameManager._counts[name] = 0
        count = NameManager._counts[name]
        return count

    @staticmethod
    def _inc_count(name):
        assert name in NameManager._counts, f'Name {name} is not registered.'
        NameManager._counts[name] += 1

    @staticmethod
    def new(name: str):
        count = NameManager._count(name)
        tensor_name = f"{name}:{count}"
        NameManager._inc_count(name)
        return tensor_name