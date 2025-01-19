import torch
from .model import ChessModel

class Controller:
    def __init__(self, model_file, device):
        self.device = device
        self.model = ChessModel().to(device)
        self.model.load_state_dict(torch.load(model_file))

    def save_model(self, model_file):
        torch.save(self.model.state_dict(), model_file)

    def update_model(self, new_model):
        self.model = new_model
        torch.save(self.model.state_dict(), "model/model.pth")
