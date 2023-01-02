from Commons.mod import resource_path

class Config:

    def __init__(self) -> None:
        self.__file = open(resource_path('./config.txt'), 'a+')
        self.data = {}
        self.__read()

    def __read(self):
        self.__file.seek(0)
        data = self.__file.read()
        data_list = data.split()
        for d in data_list:
            key_value = d.split('=')
            if len(key_value) < 2:
                continue
            self.data[key_value[0]] = key_value[1]

    def write(self):
        self.__file.truncate(0)
        text = ""
        for key in self.data:
            text = text + f"{key}={self.data[key]}\n"
        self.__file.write(text)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__file.close()

class ControllerState:
    def __init__(self) -> None:
        self.config = Config()
        self.mode = self.config.data["mode"] if "mode" in self.config.data else "relative"
        self.calibrating = False
        self.lefty = self.config.data["lefty"] == "True" if "lefty" in self.config.data else False
    
    def setMode(self, mode):
        self.mode = mode
        self.__set("mode", mode)

    def setLefty(self, lefty):
        self.lefty = lefty
        self.__set("lefty", lefty)

    def __set(self, key, value):
        self.config.data[key] = value
        self.config.write()


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.config.close()