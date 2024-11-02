from operator import itemgetter

import numpy as np


class GenericParamControl(object):
    @staticmethod
    def init_module_list():
        raise NotImplementedError

    @staticmethod
    def init_param_group():
        raise NotImplementedError

    def get_param_group(self):
        raise NotImplementedError

    def release_param(self, model):
        raise NotImplementedError


class ResnetParamControl(GenericParamControl):
    r""""
    It controls what layers to release.
    """
    def __init__(self, trainer, gradual_release=1, release_count=8, backbone_mode="ir"):
        # It effects the model and optimizer of trainer.
        self.trainer = trainer

        # 0 to disable the gradual release.
        self.gradual_release = gradual_release

        # How many times to release.
        self.release_count = release_count

        # The backbone mode of the Resnet, usually ir or ir_se.
        self.backbone_mode = backbone_mode

        # The layer indices to be grouped. One group will be released at a time.
        self.module_list = self.init_module_list()

        # Obtain the parameters according to the indices.
        self.module_stack = self.init_param_group()

        # A flag indicating whether to halt the trainer.
        self.early_stop = False

    def init_module_list(self):

        # 4-10 are the output layers. 163-187 are the fourth block of Resnet 50. 142-163 are half of the third block of Resnet 50.
        module_list = [[(4, 10)], [(163, 187)], [(142, 163)]]

        # ir_se has extra squeeze-excitation layer so that it has more layers.
        if self.backbone_mode == "ir_se":
            module_list = [[(4, 10)], [(205, 235)], [(187, 205)]]

        if self.backbone_mode == "vggish":
            module_list =  [[(12, 18)], [(10, 12)], [(8, 10)]]

        return module_list

    def init_param_group(self):
        # Get the parameters according to the module list.
        module_stack = []
        for groups in self.module_list:
            slice_range = []
            if len(groups) > 1:
                for group in groups:
                    slice_range += list(np.arange(*group))
            else:
                slice_range = list(np.arange(*groups[0]))

            module_stack.append(slice_range)
        return module_stack

    def get_param_group(self):
        # Pop out the last element of the stack.
        modules_to_release = self.module_stack.pop(0)
        return modules_to_release

    def get_current_lr(self):
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        return current_lr

    def release_param(self, model, epoch=0):
        if self.gradual_release:
            if self.release_count > 0:
                indices = self.get_param_group()

                for param in list(itemgetter(*indices)(list(model.parameters()))):
                    param.requires_grad = True

                self.trainer.init_optimizer_and_scheduler(epoch=epoch)
                self.release_count -= 1
                self.trainer.early_stopping_counter = self.trainer.early_stopping
            else:
                print("Early stopped since no further parameters to release!")
                self.early_stop = True

    def load_trainer(self, trainer):

        # If started from a checkpoint, then load the trainer.
        self.trainer = trainer


