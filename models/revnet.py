import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

CUDA = torch.cuda.is_available()

def size_after_residual(size, out_channels, kernel_size, stride, padding, dilation):
    """Calculate the size of the output of the residual function
    """
    N, C_in, H_in, W_in = size

    H_out = math.floor(
        (H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    W_out = math.floor(
        (W_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    return N, out_channels, H_out, W_out


def possible_downsample(x, in_channels, out_channels, stride=1, padding=1,
                        dilation=1):
    _, _, H_in, W_in = x.size()

    _, _, H_out, W_out = size_after_residual(x.size(), out_channels, 3, stride, padding, dilation)

    # Downsample image
    if H_in > H_out or W_in > W_out:
        out = F.avg_pool2d(x, 2*dilation+1, stride, padding)

    # Pad with empty channels
    if in_channels < out_channels:

        try: out
        except: out = x

        pad = Variable(torch.zeros(
            out.size(0),
            (out_channels - in_channels) // 2,
            out.size(2), out.size(3)
        ), requires_grad=True)

        if CUDA:
            pad = pad.cuda()

        temp = torch.cat([pad, out], dim=1)
        out = torch.cat([temp, pad], dim=1)

    # If we did nothing, add zero tensor, so the output of this function
    # depends on the input in the graph
    try: out
    except:
        injection = Variable(torch.zeros_like(x.data), requires_grad=True)

        if CUDA:
            injection.cuda()

        out = x + injection

    return out


class RevBlockFunction(Function):
    @staticmethod
    def residual(x, in_channels, out_channels, params, buffers, training,
                 stride=1, padding=1, dilation=1, no_activation=False):
        """Compute a pre-activation residual function.

        Args:
            x (Variable): The input variable
            in_channels (int): Number of channels of x
            out_channels (int): Number of channels of the output

        Returns:
            out (Variable): The result of the computation

        """
        out = x

        if not no_activation:
            out = F.batch_norm(out, buffers[0], buffers[1], params[0],
                               params[1], training)
            out = F.relu(out)

        out = F.conv2d(out, params[-6], params[-5], stride, padding=padding,
                       dilation=dilation)

        out = F.batch_norm(out, buffers[-2], buffers[-1], params[-4],
                           params[-3], training)
        out = F.relu(out)
        out = F.conv2d(out, params[-2], params[-1], stride=1, padding=1,
                       dilation=1)

        return out

    @staticmethod
    def _forward(x, in_channels, out_channels, training, stride, padding,
                 dilation, f_params, f_buffs, g_params, g_buffs,
                 no_activation=False):

        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.no_grad():
            x1 = Variable(x1.contiguous())
            x2 = Variable(x2.contiguous())

            if CUDA:
                x1.cuda()
                x2.cuda()

            x1_ = possible_downsample(x1, in_channels, out_channels, stride,
                                      padding, dilation)
            x2_ = possible_downsample(x2, in_channels, out_channels, stride,
                                      padding, dilation)

            f_x2 = RevBlockFunction.residual(
                x2,
                in_channels,
                out_channels,
                f_params,
                f_buffs, training,
                stride=stride,
                padding=padding,
                dilation=dilation,
                no_activation=no_activation
            )

            y1 = f_x2 + x1_

            g_y1 = RevBlockFunction.residual(
                y1,
                out_channels,
                out_channels,
                g_params,
                g_buffs,
                training
            )

            y2 = g_y1 + x2_

            y = torch.cat([y1, y2], dim=1)

            del y1, y2
            del x1, x2

        return y

    @staticmethod
    def _backward(output, in_channels, out_channels, f_params, f_buffs,
                  g_params, g_buffs, training, padding, dilation, no_activation):

        y1, y2 = torch.chunk(output, 2, dim=1)
        with torch.no_grad():
            y1 = Variable(y1.contiguous())
            y2 = Variable(y2.contiguous())

            x2 = y2 - RevBlockFunction.residual(
                y1,
                out_channels,
                out_channels,
                g_params,
                g_buffs,
                training=training
            )

            x1 = y1 - RevBlockFunction.residual(
                x2,
                in_channels,
                out_channels,
                f_params,
                f_buffs,
                training=training,
                padding=padding,
                dilation=dilation
            )

            del y1, y2
            x1, x2 = x1.data, x2.data

            x = torch.cat((x1, x2), 1)
        return x

    @staticmethod
    def _grad(x, dy, in_channels, out_channels, training, stride, padding,
              dilation, activations, f_params, f_buffs, g_params, g_buffs,
              no_activation=False, storage_hooks=[]):
        dy1, dy2 = torch.chunk(dy, 2, dim=1)

        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.enable_grad():
            x1 = Variable(x1.contiguous(), requires_grad=True)
            x2 = Variable(x2.contiguous(), requires_grad=True)
            x1.retain_grad()
            x2.retain_grad()

            if CUDA:
                x1.cuda()
                x2.cuda()

            x1_ = possible_downsample(x1, in_channels, out_channels, stride,
                                      padding, dilation)
            x2_ = possible_downsample(x2, in_channels, out_channels, stride,
                                      padding, dilation)

            f_x2 = RevBlockFunction.residual(
                x2,
                in_channels,
                out_channels,
                f_params,
                f_buffs,
                training=training,
                stride=stride,
                padding=padding,
                dilation=dilation,
                no_activation=no_activation
            )

            y1_ = f_x2 + x1_

            g_y1 = RevBlockFunction.residual(
                y1_,
                out_channels,
                out_channels,
                g_params,
                g_buffs,
                training=training
            )

            y2_ = g_y1 + x2_

            dd1 = torch.autograd.grad(y2_, (y1_,) + tuple(g_params), dy2,
                                      retain_graph=True)
            dy2_y1 = dd1[0]
            dgw = dd1[1:]
            dy1_plus = dy2_y1 + dy1
            dd2 = torch.autograd.grad(y1_, (x1, x2) + tuple(f_params), dy1_plus,
                                      retain_graph=True)
            dfw = dd2[2:]

            dx2 = dd2[1]
            dx2 += torch.autograd.grad(x2_, x2, dy2, retain_graph=True)[0]
            dx1 = dd2[0]

            for hook in storage_hooks:
                x = hook(x)

            activations.append(x)

            y1_.detach_()
            y2_.detach_()
            del y1_, y2_
            dx = torch.cat((dx1, dx2), 1)

        return dx, dfw, dgw

    @staticmethod
    def forward(ctx, x, in_channels, out_channels, training, stride, padding,
                dilation, no_activation, activations, storage_hooks, *args):
        """Compute forward pass including boilerplate code.

        This should not be called directly, use the apply method of this class.

        Args:
            ctx (Context):                  Context object, see PyTorch docs
            x (Tensor):                     4D input tensor
            in_channels (int):              Number of channels on input
            out_channels (int):             Number of channels on output
            training (bool):                Whethere we are training right now
            stride (int):                   Stride to use for convolutions
            no_activation (bool):           Whether to compute an initial
                                            activation in the residual function
            activations (List):             Activation stack
            storage_hooks (List[Function]): Functions to apply to activations
                                            before storing them
            *args:                          Should contain all the Parameters
                                            of the module
        """

        if not no_activation:
            f_params = [Variable(x) for x in args[:8]]
            g_params = [Variable(x) for x in args[8:16]]
            f_buffs = args[16:20]
            g_buffs = args[20:]
        else:
            f_params = [Variable(x) for x in args[:6]]
            g_params = [Variable(x) for x in args[6:14]]
            f_buffs = args[14:16]
            g_buffs = args[16:]

        if CUDA:
            for var in f_params:
                var.cuda()
            for var in g_params:
                var.cuda()

        # if the images get smaller information is lost and we need to save the input
        _, _, H_in, W_in = x.size()
        _, _, H_out, W_out = size_after_residual(x.size(), out_channels, 3, stride, padding, dilation)
        if H_in > H_out or W_in > W_out or no_activation:
            activations.append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        ctx.save_for_backward(*[x.data for x in f_params],
                              *[x.data for x in g_params])
        ctx.f_buffs = f_buffs
        ctx.g_buffs = g_buffs
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.training = training
        ctx.no_activation = no_activation
        ctx.storage_hooks = storage_hooks
        ctx.activations = activations
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels

        y = RevBlockFunction._forward(
            x,
            in_channels,
            out_channels,
            training,
            stride,
            padding,
            dilation,
            f_params, f_buffs,
            g_params, g_buffs,
            no_activation=no_activation
        )

        return y.data

    @staticmethod
    def backward(ctx, grad_out):
        saved_tensors = list(ctx.saved_tensors)
        if not ctx.no_activation:
            f_params = [Variable(p, requires_grad=True) for p in saved_tensors[:8]]
            g_params = [Variable(p, requires_grad=True) for p in saved_tensors[8:16]]
        else:
            f_params = [Variable(p, requires_grad=True) for p in saved_tensors[:6]]
            g_params = [Variable(p, requires_grad=True) for p in saved_tensors[6:14]]

        in_channels = ctx.in_channels
        out_channels = ctx.out_channels

        # Load or reconstruct input
        if ctx.load_input:
            ctx.activations.pop()
            x = ctx.activations.pop()
        else:
            output = ctx.activations.pop()
            x = RevBlockFunction._backward(
                output,
                in_channels,
                out_channels,
                f_params, ctx.f_buffs,
                g_params, ctx.g_buffs,
                ctx.training,
                ctx.padding,
                ctx.dilation,
                ctx.no_activation
            )

        dx, dfw, dgw = RevBlockFunction._grad(
            x,
            grad_out,
            in_channels,
            out_channels,
            ctx.training,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.activations,
            f_params, ctx.f_buffs,
            g_params, ctx.g_buffs,
            no_activation=ctx.no_activation,
            storage_hooks=ctx.storage_hooks
        )

        num_buffs = 2 if ctx.no_activation else 4

        return ((dx, None, None, None, None, None, None, None, None, None) + tuple(dfw) +
                tuple(dgw) + tuple([None]*num_buffs) + tuple([None]*4))


class RevBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activations, stride=1,
                 padding=1, dilation=1, no_activation=False, storage_hooks=[]):
        super(RevBlock, self).__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.no_activation = no_activation
        self.activations = activations
        self.storage_hooks = storage_hooks

        if not no_activation:
            self.register_parameter(
                'f_bw1',
                nn.Parameter(torch.Tensor(self.in_channels))
            )
            self.register_parameter(
                'f_bb1',
                nn.Parameter(torch.Tensor(self.in_channels))
            )

        self.register_parameter(
            'f_w1',
            nn.Parameter(torch.Tensor(
                self.out_channels,
                self.in_channels,
                3, 3
            ))
        )
        self.register_parameter(
            'f_b1',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'f_bw2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'f_bb2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'f_w2',
            nn.Parameter(torch.Tensor(
                self.out_channels,
                self.out_channels,
                3, 3
            ))
        )
        self.register_parameter(
            'f_b2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )

        self.register_parameter(
            'g_bw1',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_bb1',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_w1',
            nn.Parameter(torch.Tensor(
                self.out_channels,
                self.out_channels,
                3, 3
            ))
        )
        self.register_parameter(
            'g_b1',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_bw2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_bb2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_w2',
            nn.Parameter(torch.Tensor(
                self.out_channels,
                self.out_channels,
                3, 3
            ))
        )
        self.register_parameter(
            'g_b2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )

        if not no_activation:
            self.register_buffer('f_rm1', torch.zeros(self.in_channels))
            self.register_buffer('f_rv1', torch.ones(self.in_channels))
        self.register_buffer('f_rm2', torch.zeros(self.out_channels))
        self.register_buffer('f_rv2', torch.ones(self.out_channels))

        self.register_buffer('g_rm1', torch.zeros(self.out_channels))
        self.register_buffer('g_rv1', torch.ones(self.out_channels))
        self.register_buffer('g_rm2', torch.zeros(self.out_channels))
        self.register_buffer('g_rv2', torch.ones(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        f_stdv = 1 / math.sqrt(self.in_channels * 3 * 3)
        g_stdv = 1 / math.sqrt(self.out_channels * 3 * 3)

        if not self.no_activation:
            self._parameters['f_bw1'].data.uniform_()
            self._parameters['f_bb1'].data.zero_()
        self._parameters['f_w1'].data.uniform_(-f_stdv, f_stdv)
        self._parameters['f_b1'].data.uniform_(-f_stdv, f_stdv)
        self._parameters['f_w2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['f_b2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['f_bw2'].data.uniform_()
        self._parameters['f_bb2'].data.zero_()

        self._parameters['g_w1'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_b1'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_w2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_b2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_bw1'].data.uniform_()
        self._parameters['g_bb1'].data.zero_()
        self._parameters['g_bw2'].data.uniform_()
        self._parameters['g_bb2'].data.zero_()

        if not self.no_activation:
            self._buffers['f_rm1'].zero_()
            self._buffers['f_rv1'].fill_(1)
        self.f_rm2.zero_()
        self.f_rv2.fill_(1)

        self.g_rm1.zero_()
        self.g_rv1.fill_(1)
        self.g_rm2.zero_()
        self.g_rv2.fill_(1)

    def forward(self, x):
        return RevBlockFunction.apply(
            x,
            self.in_channels,
            self.out_channels,
            self.training,
            self.stride,
            self.padding,
            self.dilation,
            self.no_activation,
            self.activations,
            self.storage_hooks,
            *self._parameters.values(),
            *self._buffers.values(),
        )


class RevBottleneck(nn.Module):
    # TODO: Implement metaclass and function
    pass


class RevNet(nn.Module):
    def __init__(self,
                 units,
                 filters,
                 strides,
                 classes,
                 bottleneck=False):
        """
        Args:
            units (list-like): Number of residual units in each group

            filters (list-like): Number of filters in each unit including the
                inputlayer, so it is one item longer than units

            strides (list-like): Strides to use for the first units in each
                group, same length as units

            bottleneck (boolean): Wether to use the bottleneck residual or the
                basic residual
        """
        super(RevNet, self).__init__()
        self.name = self.__class__.__name__

        self.activations = []

        if bottleneck:
            self.Reversible = RevBottleneck     # TODO: Implement RevBottleneck
        else:
            self.Reversible = RevBlock

        self.layers = nn.ModuleList()

        # Input layer
        # self.layers.append(nn.Conv2d(3, filters[0], 3, padding=1))
        self.layers.append(nn.Conv2d(3, filters[0], 7, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))

        for i, group_i in enumerate(units):
            self.layers.append(self.Reversible(
                filters[i], filters[i + 1],
                stride=strides[i],
                no_activation=True,
                activations=self.activations
            ))

            for unit in range(1, group_i):
                self.layers.append(self.Reversible(
                    filters[i + 1],
                    filters[i + 1],
                    activations=self.activations
                ))

        self.fc = nn.Linear(filters[-1], classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # Save last output for backward
        self.activations.append(x.data)

        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def free(self):
        """Clear saved activation residue and thereby free memory."""
        del self.activations[:]
