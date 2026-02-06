import os
import torch


class BaseModel(torch.nn.Module):
    """Minimal base model definition for training/testing loops."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device("cuda:0" if opt.gpu_ids else "cpu")
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.model)

    @staticmethod
    def modify_options(parser, is_train):
        return parser

    def set_input(self, input_data):
        self.input = input_data

    def get_current_errors(self):
        return {}

    def update_learning_rate(self):
        pass

    def save_networks(self, label):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"net_{label}.pth")
        torch.save(self.state_dict(), save_path)
