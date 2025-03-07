import torch
import torch.nn as nn
import torch.nn.functional as F
from flags import FLAGS


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        
        # Ensure we have a list
        if not isinstance(num_features_list, list):
            num_features_list = [num_features_list]
        
        self.num_features_list = [int(i) for i in num_features_list]
        self.num_features = max(self.num_features_list)
        self.width_mult = 1.0
        # Create separate BN for each possible width
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(i) for i in self.num_features_list]
        )
        
    def forward(self, input):
        # Find index in width_mult_list for the current width
        idx = FLAGS.width_mult_list.index(self.width_mult)
        # Use the corresponding BN
        y = self.bn[idx](input)
        
        return y

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                kernel_size, stride=1, padding=0, dilation=1,
                groups_list=[1], bias=True):
        
        # Ensure we have lists
        if not isinstance(in_channels_list, list):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, list):
            out_channels_list = [out_channels_list]
        if not isinstance(groups_list, list):
            groups_list = [groups_list]
        
        # Convert float channel counts to integers
        in_channels_list = [int(c) for c in in_channels_list]
        out_channels_list = [int(c) for c in out_channels_list]
        groups_list = [int(g) for g in groups_list]
        
        # Initialize with maximum sizes
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list), kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias
        )
        
        # Store channel options
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list if len(groups_list) == len(in_channels_list) else [1] * len(in_channels_list)
        
        # Default to maximum channels
        self.width_mult = 1.0
        self.current_in_channels = max(in_channels_list)
        self.current_out_channels = max(out_channels_list)
        self.current_groups = max(groups_list)
        
    def forward(self, input):
        # Use sliced weights according to current width
        weight = self.weight[
            0:self.current_out_channels, 0:self.current_in_channels, :, :]
        
        if self.bias is not None:
            bias = self.bias[0:self.current_out_channels]
        else:
            bias = self.bias
            
        return nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.current_groups)
        
class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)
        
class Block(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = SlimmableConv2d(inplanes, planes, kernel_size=3,
                                    padding=1, stride=stride, bias=False)
        self.batch_norm1 = SwitchableBatchNorm2d(planes)
        
        self.conv2 = SlimmableConv2d(planes, planes, kernel_size=3,
                                    padding=1, stride=1, bias=False)
        self.batch_norm2 = SwitchableBatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        
        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = SlimmableConv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = SwitchableBatchNorm2d(planes)
        
        self.conv2 = SlimmableConv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = SwitchableBatchNorm2d(planes)
        
        self.conv3 = SlimmableConv2d(planes, [p*self.expansion for p in planes], kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = SwitchableBatchNorm2d([p*self.expansion for p in planes])
        
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        out = self.batch_norm3(self.conv3(out))
        
        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add identity
        out += identity
        out = self.relu(out)
        
        return out


class SlimmableResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(SlimmableResNet, self).__init__()
        
        # Default width multiplier
        self.width_mult = 1.0
        
        # Calculate channel counts for each width
        self.width_mults = FLAGS.width_mult_list
        base_channels = 64
        self.inplanes = [int(base_channels * w) for w in self.width_mults]
        
        # First conv layer
        self.conv1 = SlimmableConv2d([3] * len(self.width_mults), self.inplanes,
                                    kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = SwitchableBatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet stages
        self.layer1 = self._make_layer(block, [base_channels * w for w in self.width_mults], layers[0])
        self.layer2 = self._make_layer(block, [base_channels*2 * w for w in self.width_mults], layers[1], stride=2)
        self.layer3 = self._make_layer(block, [base_channels*4 * w for w in self.width_mults], layers[2], stride=2)
        self.layer4 = self._make_layer(block, [base_channels*8 * w for w in self.width_mults], layers[3], stride=2)
        
        # Classification Head
        # [B, C, H, W] -> [B, C, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        
        final_channels_list = [
            int(base_channels * 8 * block.expansion * w)
            for w in self.width_mults
        ]
        self.fc = SlimmableLinear(
            in_features_list=final_channels_list,
            out_features_list=[num_classes] * len(final_channels_list),
            bias=True
        )
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        # Create downsample path if needed
        if stride != 1 or self.inplanes[0] != planes[0] * block.expansion:
            expanded_planes = [int(p * block.expansion) for p in planes]
            
            downsample = nn.Sequential(
                SlimmableConv2d(
                    self.inplanes, expanded_planes,
                    kernel_size=1, stride=stride, bias=False),
                SwitchableBatchNorm2d(expanded_planes)
            )
        
        layers = []
        # First block has stride and downsample
        layers.append(block(self.inplanes, planes, downsample, stride))
        
        # Update inplanes for next blocks
        self.inplanes = [int(p * block.expansion) for p in planes]
        
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)  # Shape: [B, C, 1, 1]
        x = torch.flatten(x, 1)  # Shape: [B, C]

        x = self.fc(x)  # Shape: [B, num_classes]
        
        return x
    
    def switch_to_width(self, width_mult):
        """Set the network to use a specific width multiplier"""
        # Set the model-level width multiplier
        self.width_mult = width_mult
        
        def set_width_mult(m, width_mult):
            """Set width multiplier for a slimmable module"""
            if hasattr(m, 'width_mult'):
                m.width_mult = width_mult
                if hasattr(m, 'in_channels_list'):  # For SlimmableConv2d
                    idx = FLAGS.width_mult_list.index(width_mult)
                    m.current_in_channels = m.in_channels_list[idx]
                    m.current_out_channels = m.out_channels_list[idx]
                    m.current_groups = m.groups_list[idx]
            
        self.apply(lambda m: set_width_mult(m, width_mult))

def SlimmableResNet18(num_classes=1000):
    return SlimmableResNet(Block, [2,2,2,2], num_classes=num_classes)
    
def SlimmableResNet34(num_classes=1000):
    return SlimmableResNet(Block, [3,4,6,3], num_classes=num_classes)      
        
def SlimmableResNet50(num_classes=1000):
    return SlimmableResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)
    
def SlimmableResNet101(num_classes=1000):
    return SlimmableResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def SlimmableResNet152(num_classes=1000):
    return SlimmableResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)