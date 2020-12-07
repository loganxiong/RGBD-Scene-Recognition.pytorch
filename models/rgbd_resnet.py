import torch.nn as nn
from torch.nn import Conv2d,Parameter,Softmax
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
import pdb,os
from torchvision.models import resnet

__all__ = ['ResNet', 'resnet18']

class SA_Module(nn.Module):
    """ Spatial attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(SA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x, hha):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        #pdb.set_trace()
        rgb = x
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        attend_out = out.view(m_batchsize, C, height, width)

        #out = attend_out + x
        rgb_out = self.gamma*attend_out + rgb
        hha_out = self.gamma*attend_out + hha
        #print(self.gamma)
        return attention, rgb_out, hha_out
        
class Network(nn.Module):
	def __init__( self, config ):
		super(Network, self).__init__()
		self.num_classes = config['n_classes']

		
		self.rgb_fc6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 384),
            nn.ReLU(inplace=True),
            nn.Dropout()
		)
		
		self.hha_fc6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 384),
            nn.ReLU(inplace=True),
            nn.Dropout()
		)
		self.aux_rgb_classifier = nn.Sequential(
		        nn.Linear(384, self.num_classes)
		    )
		self.aux_d_classifier = nn.Sequential(
            nn.Linear(384, self.num_classes)
        )
		self.concat_classifier = nn.Sequential(
            nn.Linear(768, self.num_classes)
        )
		self.model_rgb = resnet.resnet18()
		self.model_hha = resnet.resnet18()
		
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		
		
		self.rgb_hha_transform = SA_Module(512)
		
		self.num_node = 16
		self.model_rgb = self.load_pretrained_model(self.model_rgb)
		self.model_hha = self.load_pretrained_model(self.model_hha)
		
		
	def load_pretrained_model(self, model):
		del model.fc
		state_dict = model.state_dict()
		pretrained_path = './models/resnet18_places365.pth.tar'
		resnet_places_params = torch.load(pretrained_path)['state_dict']
		pre_parames = resnet_places_params
		pre_keys = list(resnet_places_params.keys())
		#pdb.set_trace()
		for _,k in enumerate(state_dict):
			ckey = 'module.' + k
			if ckey in pre_keys:
				state_dict[k] = pre_parames[ckey]
		model.load_state_dict(state_dict)
		return model
      
	def forward(self, x, hha):
		rgb_spatial = self.model_rgb(x)
		depth_spatial = self.model_hha(hha)
		
		# Attention
		_, rgb_spatial, depth_spatial = self.rgb_hha_transform(rgb_spatial, depth_spatial)
		# GCN
		# ASK
		
		rgb_embedding = self.avgpool(rgb_spatial).view(x.size(0),-1)   #64,512,15,20
		hha_embedding = self.avgpool(depth_spatial).view(hha.size(0),-1)   #64,512,15,20
	  
		rgb_encode_out = self.rgb_fc6(rgb_embedding)
		hha_encode_out = self.hha_fc6(hha_embedding)
		concat_input = torch.cat([rgb_encode_out, hha_encode_out],1)
		model_out = self.concat_classifier(concat_input)
		aux_rgb_out = self.aux_rgb_classifier(rgb_encode_out)
		aux_d_out = self.aux_d_classifier(hha_encode_out)
		model_out = (model_out + aux_rgb_out + aux_d_out) / 3.
		
		return model_out, aux_rgb_out, aux_d_out, concat_input
		
if __name__=='__main__':
    model = Network({'n_classes':10}).cuda()
    tt = torch.ones([2,3,224,224]).cuda()
    tt2 = torch.ones([2,3,224,224]).cuda()
    print(model(tt,tt2)[0].shape)