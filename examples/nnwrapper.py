import torch
from torch import nn
from torch.autograd.functional import jacobian
from torch.func import jvp as torch_jvp
from torch.func import vjp as torch_vjp


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


class NNWrapper():

    def __init__(self):

        # load saved weights into network
        data = torch.load("auto.pth", weights_only=True)
        weights = data["weights"]
        model = NeuralNetwork()
        model.load_state_dict(weights)
        model.eval()

        # this can be as complex as desired.  in this case we just normalize input
        # call a single network, then unnormalize the output
        def mynetwork(x):
            xscale = (x - data["fmean"]) / data["fstd"]
            yscale = model(xscale)
            return yscale * data["tstd"] + data["tmean"]

        self.networkfn = mynetwork
        self.dtype = torch.float32


    def eval(self, x):
        x = torch.tensor(x, dtype=self.dtype)
        y = self.networkfn(x)

        return y.detach().numpy()

    def jacobian(self, x):
        x = torch.tensor(x, dtype=self.dtype, requires_grad=True)
        dydx = jacobian(self.networkfn, x)

        return dydx.detach().numpy()

    # both the jvp and vjp could written more efficiently
    # since y and ydot are computed in one call
    # I could cache them, compare whether x has changed, and reuse therm
    # but keeping it simple ane short for this demo.
    def jvp(self, x, v):
        x = torch.tensor(x, dtype=self.dtype, requires_grad=True)
        v = torch.tensor(v, dtype=self.dtype)

        y, ydot = torch_jvp(self.networkfn, (x,), (v,))

        return ydot.detach().numpy()

    def vjp(self, x, v):
        x = torch.tensor(x, dtype=self.dtype, requires_grad=True)
        v = torch.tensor(v, dtype=self.dtype)

        y, vjpfunc = torch_vjp(self.networkfn, x)
        xbar = vjpfunc(v)[0]

        return xbar.detach().numpy()

