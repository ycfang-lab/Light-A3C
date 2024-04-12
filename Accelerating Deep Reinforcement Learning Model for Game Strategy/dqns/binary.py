from layers.binary import BinNoisy
from torch import nn
from torch.nn import functional as F


class BinaryDQN(nn.Module):
    def __init__(self, args, action_space):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.conv1 = nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.fc_h_v = BinNoisy(3136, args.hidden_size)
        self.fc_h_a = BinNoisy(3136, args.hidden_size)
        self.fc_z_v = BinNoisy(args.hidden_size, self.atoms)
        self.fc_z_a = BinNoisy(args.hidden_size, action_space * self.atoms)

    def forward(self, x, log=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()
