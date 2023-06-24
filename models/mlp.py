import torch
   
   
class MLP(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hiddens, num_outputs), 
        )

    def forward(self, img):
        flat = img.view(img.shape[0], -1)
        return self.net(flat)
   
    
net = MLP(num_inputs=784, num_hiddens=256, num_outputs=10)
print(net)
print('parameters:', sum(param.numel() for param in net.parameters()))