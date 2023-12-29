import torch

class MyMetric:
    def __init__(self, name):
        self.name = "m/" + name
        self.error = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

    def update_state(self, error):
        self.error += error
        self.count += 1.0

    def result(self):
        return self.error / self.count if self.count != 0 else torch.tensor(0.0)

    def reset(self):
        self.error = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

