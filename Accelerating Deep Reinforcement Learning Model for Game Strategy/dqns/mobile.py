from layers.noisy import NoisyLinear
from torch import nn
from torch.nn import functional as F


class MobileDQN(nn.Module):
    def __init__(self, args, action_space):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 4, stride=2, groups=32, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, groups=64, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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