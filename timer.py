import time

class Timer:

    def __init__(self):
        self.time = time.time()
        self.lap_time = []
    
    def count(self):
        now = time.time()
        self.lap_time.append(now - self.time)
        self.time = now