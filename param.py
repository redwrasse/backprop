class Param(object):

    def __init__(self, name, init_value):
        self.name = name
        self.value = init_value

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value