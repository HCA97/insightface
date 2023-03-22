import torch.nn as nn
import open_clip

class ViT_CLIP(nn.Module):
    def __init__(self, model_name, model_data):
        super(ViT_CLIP, self).__init__()     

        backbone, _, _ = open_clip.create_model_and_transforms(model_name, model_data)   
        self.encoder = backbone.visual

        if model_name == 'ViT-L-14':
            self.learning_rate = {'8': 1.25e-6, '16': 2.5e-6, '20': 5e-6, '24': 10e-6}
        else:
            self.learning_rate = {'10': 1.25e-6, '20': 2.5e-6, '26': 5e-6, '32': 10e-6} 
        self.weight_decay = 1e-3

    def forward(self, x):
        x = self.encoder(x)
        return x

  
    def get_parameters(self):

        parameter_settings = [] 
        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.encoder.named_parameters()], 
                lr=self.learning_rate, 
                wd=self.weight_decay
            )
        ) 

        return parameter_settings

    def get_parameter_section(self, parameters, lr=None, wd=None): 
        parameter_settings = []


        lr_is_dict = isinstance(lr, dict)
        wd_is_dict = isinstance(wd, dict)

        layer_no = None
        for no, (n,p) in enumerate(parameters):
            
            for split in n.split('.'):
                if split.isnumeric():
                    layer_no = int(split)
            
            if not layer_no:
                layer_no = 0
            
            if lr_is_dict:
                for k,v in lr.items():
                    if layer_no < int(k):
                        temp_lr = v
                        break
            else:
                temp_lr = lr

            if wd_is_dict:
                for k,v in wd.items():
                    if layer_no < int(k):
                        temp_wd = v
                        break
            else:
                temp_wd = wd
                
            weight_decay = 0.0 if 'bias' in n else temp_wd
            parameter_setting = {"params" : p, "lr" : temp_lr, "weight_decay" : temp_wd}
            parameter_settings.append(parameter_setting)

        return parameter_settings