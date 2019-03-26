import torch
import torch.nn as nn

class DownsampleA(nn.Module):  

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__() 
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   

  def forward(self, x):   
    x = self.avg(x)  
    return torch.cat((x, x.mul(0)), 1)  

class DownsampleC(nn.Module):     

  def __init__(self, nIn, nOut, stride):
    super(DownsampleC, self).__init__()
    assert stride != 1 or nIn != nOut
    self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

  def forward(self, x):
    x = self.conv(x)
    return x

class DownsampleD(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleD, self).__init__()
    assert stride == 2
    self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
    self.bn   = nn.BatchNorm2d(nOut)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x
#
#class SelectiveSequential(nn.Module):
#    def __init__(self, to_select, modules_dict):
#        super(SelectiveSequential, self).__init__()
#        for key, module in modules_dict.items():
#            self.add_module(key, module)
#        self._to_select = to_select
#    
#    def forward(self,x):
#        list = []
#        for name, module in self._modules.items():
#            x = module(x)
#            if name in self._to_select:
#                list.append(x)
#        return list
#    
#class FeatureExtractor(nn.Module):
#    def __init__(self, submodule, extracted_layers):
#        super(FeatureExtractor, self).__init__()
#        self.submodule = submodule
#
#    def forward(self, x):
#        outputs = []
#        for name, module in self.submodule._modules.items():
#            x = module(x)
#            if name in self.extracted_layers:
#                outputs += [x]
#        return outputs + [x]
#    
#original_model = torchvision.models.alexnet(pretrained=True)
#
#class AlexNetConv4(nn.Module):
#    def __init__(self):
#        super(AlexNetConv4, self).__init__()
#        self.features = nn.Sequential(
#            # stop at conv4
#            *list(original_model.features.children())[:-3]
#        )
#    def forward(self, x):
#        x = self.features(x)
#        return x
#
#model = AlexNetConv4()
#
#
#extract_feature = {}
#count = 0
#def save_hook(module, input, output):
##    global hook_key
#    global count
#    temp = torch.zeros(output.size())
#    temp.copy_(output.data)
#    extract_feature[count] = temp
#    count += 1
#    print(extract_feature)
#    
#class Myextract(nn.Module):
#    
#    def __init__(self, model):
#        super(Myextract, self).__init__()
#        self.model = model
#        self.extract_feature = {}
#        self.hook_key = {}
#        self.count= 0
#        
#    def add_hook(self):  
#        for key, module in model._modules.items():
#            if 'stage' in key:
#                i = 1
#                for block in module.children():
##                    self.get_key( key + '_block_' + str(i))
#                    self.add_hook_block(key + '_block_' + str(i), module)
#                    i = i+1
#                print(i)
#            else:
#                self.get_key (key)
#                module.register_forward_hook (save_hook)
#            print('add hook  done')
#
#    def add_hook_block(self,key,module):
##        module.bn_a.register_forward_hook (self.save(key+'_bn_a'))
##        module.bn_b.register_forward_hook (self.save(key+'_bn_b'))
#        self.get_key(key+'_bn_a')
#        module.bn_a.register_forward_hook (save_hook)
#        self.get_key(key+'_bn_b')
#        module.bn_b.register_forward_hook (save_hook)
#        print('add hook block done')
#        
#
#    def get_key(self, key):
#        self.count += 1
#        self.hook_key[self.count] = key
#
#    def run(self):
#        self.add_hook()
#        
#        
#model.layer2.conv1.register_forward_hook (hook)