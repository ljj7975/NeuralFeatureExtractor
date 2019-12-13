from datetime import datetime

class Timer:
    def __init__(self):
        self.start_time = datetime.now()

    def lap(self):
        now = datetime.now()
        duration = now - self.start_time
        return duration.total_seconds()

    def reset(self):
        self.start_time = datetime.now()
