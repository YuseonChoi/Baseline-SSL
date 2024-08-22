import torch
import torch.nn as nn
import torch.nn.functional as F

import Module as at_module

class FNblock(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.2, is_online=False, is_first=False):
        """ The fullband and narrowband fusion block
            * LSTM input: (batch_size, sequence_length, input_size)
            * LSTM output: (batch_size, sequence_length, D*H_out) 
        """
        super(FNblock, self).__init__()
        self.input_size = input_size
        self.full_hidden_size =  hidden_size // 2
        self.is_first = is_first
        self.is_online = is_online
        if self.is_online:
            self.narr_hidden_size = hidden_size
        else:
            self.narr_hidden_size = hidden_size  // 2
        self.dropout = dropout

        self.dropout_full =  nn.Dropout(p=self.dropout)
        self.dropout_narr = nn.Dropout(p=self.dropout)
        self.fullLstm = nn.LSTM(input_size=self.input_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        if self.is_first:
              self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size+self.input_size, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)
        else:
            self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)
        
    def forward(self, x, nb_skip=None, fb_skip=None):
        nb,nt,nf,nc = x.shape
        nb_skip = x.permute(0,2,1,3).reshape(nb*nf,nt,-1)
        x = x.reshape(nb*nt,nf,-1)
        if not self.is_first:
            x = x + fb_skip
        x, _ = self.fullLstm(x)
        fb_skip = x
        x = self.dropout_full(x)
        x = x.view(nb,nt,nf,-1).permute(0,2,1,3).reshape(nb*nf,nt,-1)
        if self.is_first:  
            x = torch.cat((x,nb_skip),dim=-1)
        else:
            x = x + nb_skip
        x, _ = self.narrLstm(x)
        nb_skip = x
        x = self.dropout_narr(x)
        x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
        ## torch.Size([2, 298, 256, 256]) torch.Size([596, 256, 256]) torch.Size([512, 298, 256])
        return x, fb_skip, nb_skip


class ConvBlock(nn.Module):
    """ the convolutional block
        * N: 배치 크기 / C_in: in_channels 값과 일치해야 함 / H_in: 2D input tensor 높이 / W_in: 너비
        * input: (N, C_in, H_in, W_in)
        * output: (N, C_out, H_out, W_out)
    """
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(8,1))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(8,1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=(8,1))
        self.pooling1 = nn.AvgPool2d(kernel_size=(4,1))
        self.pooling2 = nn.AvgPool2d(kernel_size=(3,1))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        print("연산 전:", x.shape)
        ## causal하게 만들어주기 위해 kernel size에 맞게 패딩 적용
        x = self.relu(self.pooling1(self.conv1(F.pad(x, (0, 0, 7, 0)))))
        print("conv1 연산 후:", x.shape)  ## torch.Size([2,128,74,256])
        x = self.relu(self.pooling2(self.conv2(F.pad(x, (0, 0, 7, 0)))))
        print("conv2 연산 후:", x.shape)  ## torch.Size([2,128,24,256])
        x = self.tanh(self.conv3(F.pad(x, (0, 0, 7, 0))))  
        print("conv3 연산 후:", x.shape) ## torch.Size([2,4,24,256])
        return x

       
class IPDnet(nn.Module):
    def __init__(self,input_size=4,hidden_size=128,is_online=True,is_doa=False):
        """ IPDnet
        """
        super(IPDnet, self).__init__()
        self.is_online = is_online
        self.is_doa = is_doa
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = FNblock(input_size=self.input_size,is_online=self.is_online, is_first=True)       
        self.block_2 = FNblock(input_size=self.hidden_size,is_online=self.is_online, is_first=False)
        self.convblock = ConvBlock()
        self.tanh = nn.Tanh()
        if self.is_doa:
            self.ipd2doa = nn.Linear(512,180)
    def forward(self,x):
        ## nb:2 (한번에 처리할 오디오 개수) /nc:4 (투채널 오디오 real,imag) /nf:256/nt:298
        ## nb:2/nt:298/nf:256/nc:4
        x = x.permute(0,3,2,1)
        nb,nt,nf,nc = x.shape
        x, fb_skip, nb_skip = self.block_1(x)
        x, fb_skip, nb_skip = self.block_2(x,fb_skip=fb_skip, nb_skip=nb_skip)
        ## torch.Size([2, 298, 256, 256]) nb,nt,nf,d torch.Size([596, 256, 256]) torch.Size([512, 298, 256])
        ## cnn input: (N, C_in, H_in, W_in)
        ## nb nc nt nf
        x = x.permute(0,3,1,2)
        x = self.convblock(x)
        return x
    
    
if __name__ == "__main__":
	import torch
    ## 마이크 M=2, K=2 라고 가정해보겠음
    ## 마이크 개수랑 소스 수는 인자로 넣어줘야 하는지?
    ## input: nb:2/nc:4/nf:256/nt:298
	input = torch.randn((2,4,256,298)).cuda()
	net = IPDnet().cuda()
	ouput = net(input)
    ## output: torch.Size([2, 24, 512])
	print(ouput.shape)
	print('# parameters:', sum(param.numel() for param in net.parameters()))
