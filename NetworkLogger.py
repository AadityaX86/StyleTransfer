# Tensorboard logger
from torch.utils.tensorboard import SummaryWriter
import os
import time
from ast import Constant

from typing import Dict

# Logger that logs network details like loss, accuracy, etc. to a file
class NetworkLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self.writer = SummaryWriter(log_path)
        self.writers: Dict[str, SummaryWriter] = {}


    def log_scalar(self, tag, scalar_value, global_step, component_name : Constant = "") -> None:
        self.get_writer(component_name).add_scalar(tag, scalar_value, global_step)

    def log_scalars(self, main_tag, tag_scalar_dict, global_step, component_name : Constant = "") -> None:
        self.get_writer(component_name).add_scalars(main_tag, tag_scalar_dict, global_step)

    def log_image(self, tag, img_tensor, global_step, component_name: Constant = "") -> None:
        self.get_writer(component_name).add_image(tag, img_tensor, global_step)


    def get_writer(self, component_name: Constant = "") -> SummaryWriter:
        if component_name == "":
            return self.writer

        if component_name not in self.writers:
            path = os.path.join(self.log_path, component_name)
            os.makedirs(path, exist_ok=True)
            self.writers[component_name] = SummaryWriter(path)

        return self.writers[component_name]