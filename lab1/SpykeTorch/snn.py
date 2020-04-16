import torch
import torch.nn as nn
import torch.nn.functional as fn
from . import functional as sf
from torch.nn.parameter import Parameter
from torch.distributions.bernoulli import Bernoulli
from .utils import to_pair


###################### DON'T MODIFY HERE ############################
class Convolution(nn.Module):
    r"""Performs a 2D convolution over an input spike-wave composed of several input
    planes. Current version only supports stride of 1 with no padding.

    The input is a 4D tensor with the size :math:`(T, C_{{in}}, H_{{in}}, W_{{in}})` and the crresponsing output
    is of size :math:`(T, C_{{out}}, H_{{out}}, W_{{out}})`, 
    where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
    :math:`H`, and :math:`W` are the hight and width of the input/output planes.

    * :attr:`in_channels` controls the number of input planes (channels/feature maps).

    * :attr:`out_channels` controls the number of feature maps in the current layer.

    * :attr:`kernel_size` controls the size of the convolution kernel. It can be a single integer or a tuple of two integers.

    * :attr:`weight_mean` controls the mean of the normal distribution used for initial random weights.

    * :attr:`weight_std` controls the standard deviation of the normal distribution used for initial random weights.

    .. note::

        Since this version of convolution does not support padding, it is the user responsibility to add proper padding
        on the input before applying convolution.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        weight_mean (float, optional): Mean of the initial random weights. Default: 0.8
        weight_std (float, optional): Standard deviation of the initial random weights. Default: 0.02
    """
    def __init__(self, in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.02):
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_pair(kernel_size)
        #self.weight_mean = weight_mean
        #self.weight_std = weight_std

        # For future use
        self.stride = 1
        self.bias = None
        self.dilation = 1
        self.groups = 1
        self.padding = 0

        # Parameters
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.weight.requires_grad_(False) # We do not use gradients
        self.reset_weight(weight_mean, weight_std)

    def reset_weight(self, weight_mean=0.8, weight_std=0.02):
        """Resets weights to random values based on a normal distribution.

        Args:
            weight_mean (float, optional): Mean of the random weights. Default: 0.8
            weight_std (float, optional): Standard deviation of the random weights. Default: 0.02
        """
        self.weight.normal_(weight_mean, weight_std)

    def load_weight(self, target):
        """Loads weights with the target tensor.

        Args:
            target (Tensor=): The target tensor.
        """
        self.weight.copy_(target)	

    def forward(self, input):
        return fn.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Pooling(nn.Module):
    r"""Performs a 2D max-pooling over an input signal (spike-wave or potentials) composed of several input
    planes.

    .. note::

        Regarding the structure of the spike-wave tensors, application of max-pooling over spike-wave tensors results
        in propagation of the earliest spike within each pooling window.

    The input is a 4D tensor with the size :math:`(T, C, H_{{in}}, W_{{in}})` and the crresponsing output
    is of size :math:`(T, C, H_{{out}}, W_{{out}})`, 
    where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
    :math:`H`, and :math:`W` are the hight and width of the input/output planes.

    * :attr:`kernel_size` controls the size of the pooling window. It can be a single integer or a tuple of two integers.

    * :attr:`stride` controls the stride of the pooling. It can be a single integer or a tuple of two integers. If the value is None, it does pooling with full stride.

    * :attr:`padding` controls the amount of padding. It can be a single integer or a tuple of two integers.

    Args:
        kernel_size (int or tuple): Size of the pooling window
        stride (int or tuple, optional): Stride of the pooling window. Default: None
        padding (int or tuple, optional): Size of the padding. Default: 0
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(Pooling, self).__init__()
        self.kernel_size = to_pair(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = to_pair(stride)
        self.padding = to_pair(padding)

        # For future use
        self.dilation = 1
        self.return_indices = False
        self.ceil_mode = False

    def forward(self, input):
        return sf.pooling(input, self.kernel_size, self.stride, self.padding)

class STDP(nn.Module):
    r"""Performs STDP learning rule over synapses of a convolutional layer based on the following formulation:

    .. math::
        \Delta W_{ij}=
        \begin{cases}
            a_{LTP}\times \left(W_{ij}-W_{LB}\right)\times \left(W_{UP}-W_{ij}\right) & \ \ \ t_j - t_i \leq 0,\\
            a_{LTD}\times \left(W_{ij}-W_{LB}\right)\times \left(W_{UP}-W_{ij}\right) & \ \ \ t_j - t_i > 0,\\
        \end{cases}

    where :math:`i` and :math:`j` refer to the post- and pre-synaptic neurons, respectively,
    :math:`\Delta w_{ij}` is the amount of weight change for the synapse connecting the two neurons,
    and :math:`a_{LTP}`, and :math:`a_{LTD}` scale the magnitude of weight change. Besides,
    :math:`\left(W_{ij}-W_{LB}\right)\times \left(W_{UP}-W_{ij}\right)` is a stabilizer term which
    slowes down the weight change when the synaptic weight is close to the weight's lower (:math:`W_{LB}`)
    and upper (:math:`W_{UB}`) bounds.

    To create a STDP object, you need to provide:

    * :attr:`conv_layer`: The convolutional layer on which the STDP should be applied.

    * :attr:`learning_rate`: (:math:`a_{LTP}`, :math:`a_{LTD}`) rates. A single pair of floats or a list of pairs of floats. Each feature map has its own learning rates.

    * :attr:`use_stabilizer`: Turns the stabilizer term on or off.

    * :attr:`lower_bound` and :attr:`upper_bound`: Control the range of weights.

    To apply STDP for a particular stimulus, you need to provide:

    * :attr:`input_spikes` and :attr:`potentials` that are the input spike-wave and corresponding potentials, respectively.

    * :attr:`output_spikes` that is the output spike-wave.

    * :attr:`winners` or :attr:`kwta` to find winners based on the earliest spike then the maximum potential.

    * :attr:`inhibition_radius` to inhibit surrounding neurons (in all feature maps) within a particular radius.

    Args:
        conv_layer (snn.Convolution): Reference convolutional layer.
        learning_rate (tuple of floats or list of tuples of floats): (LTP, LTD) rates for STDP.
        use_stabilizer (boolean, optional): Turning stabilizer term on or off. Default: True
        lower_bound (float, optional): Lower bound of the weight range. Default: 0
        upper_bound (float, optional): Upper bound of the weight range. Default: 1
    """
    def __init__(self, conv_layer, learning_rate, use_stabilizer = True, lower_bound = 0, upper_bound = 1):
        super(STDP, self).__init__()
        self.conv_layer = conv_layer
        if isinstance(learning_rate, list):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = [learning_rate] * conv_layer.out_channels
        for i in range(conv_layer.out_channels):
            self.learning_rate[i] = (Parameter(torch.tensor([self.learning_rate[i][0]])),
                            Parameter(torch.tensor([self.learning_rate[i][1]])))
            self.register_parameter('ltp_' + str(i), self.learning_rate[i][0])
            self.register_parameter('ltd_' + str(i), self.learning_rate[i][1])
            self.learning_rate[i][0].requires_grad_(False)
            self.learning_rate[i][1].requires_grad_(False)
        self.use_stabilizer = use_stabilizer
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_pre_post_ordering(self, input_spikes, output_spikes, winners):
        r"""Computes the ordering of the input and output spikes with respect to the position of each winner and
        returns them as a list of boolean tensors. True for pre-then-post (or concurrency) and False for post-then-pre.
        Input and output tensors must be spike-waves.

        Args:
            input_spikes (Tensor): Input spike-wave
            output_spikes (Tensor): Output spike-wave
            winners (List of Tuples): List of winners. Each tuple denotes a winner in a form of a triplet (feature, row, column).

        Returns:
            List: pre-post ordering of spikes
        """
        # accumulating input and output spikes to get latencies
        input_latencies = torch.sum(input_spikes, dim=0)
        output_latencies = torch.sum(output_spikes, dim=0)
        result = []
        for winner in winners:
            # generating repeated output tensor with the same size of the receptive field
            out_tensor = torch.ones(*self.conv_layer.kernel_size, device=output_latencies.device) * output_latencies[winner]
            # slicing input tensor with the same size of the receptive field centered around winner
            # since there is no padding, there is no need to shift it to the center
            in_tensor = input_latencies[:,winner[-2]:winner[-2]+self.conv_layer.kernel_size[-2],winner[-1]:winner[-1]+self.conv_layer.kernel_size[-1]]
            result.append(torch.ge(in_tensor,out_tensor))
        return result

    # simple STDP rule
    # gets prepost pairings, winners, weights, and learning rates (all shoud be tensors)
    def forward(self, input_spikes, potentials, output_spikes, winners=None, kwta = 1, inhibition_radius = 0):
        if winners is None:
            winners = sf.get_k_winners(potentials, kwta, inhibition_radius, output_spikes)
        pairings = self.get_pre_post_ordering(input_spikes, output_spikes, winners)

        lr = torch.zeros_like(self.conv_layer.weight)
        for i in range(len(winners)):
            f = winners[i][0]
            lr[f] = torch.where(pairings[i], *(self.learning_rate[f]))

        self.conv_layer.weight += lr * ((self.conv_layer.weight-self.lower_bound) * (self.upper_bound-self.conv_layer.weight) if self.use_stabilizer else 1)
        self.conv_layer.weight.clamp_(self.lower_bound, self.upper_bound)

    def update_learning_rate(self, feature, ap, an):
        r"""Updates learning rate for a specific feature map.

        Args:
            feature (int): The target feature.
            ap (float): LTP rate.
            an (float): LTD rate.
        """
        self.learning_rate[feature][0][0] = ap
        self.learning_rate[feature][1][0] = an

    def update_all_learning_rate(self, ap, an):
        r"""Updates learning rates of all the feature maps to a same value.

        Args:
            ap (float): LTP rate.
            an (float): LTD rate.
        """
        for feature in range(self.conv_layer.out_channels):
            self.learning_rate[feature][0][0] = ap
            self.learning_rate[feature][1][0] = an

#############################################################

################## WRITE CODE BELOW #########################

### *** WRITE THIS FUNCTION TO PERFORM LOCAL CONVOLUTION *** ###

### This class should implement a Local Convolution layer without weight sharing ###
# It should perform a local receptive-field-wise 2D convolution over an input spike-wave composed of several input
# planes. It doesn't involve weight-sharing, which is more biologically plausible compared to regular convolution
# with weight sharing.
# You can view this class as building a layer consisting of columns arranged in a 2D 'rows x cols' grid. In our case, it will be # just a 1 x 1 grid of 1 column. But DON'T HARDCODE THIS.
# Each such column has 'out_channels' number of neurons. Each such neuron looks at a Receptive Field which has 'in_channels'
# number of channels, with each channel being a 2D grid of size 'kernel_size x kernel_size'
# Note that with the current cumulative spike wave mapping, a 2D convolution of input with weights implements nothing but a
# step-no-leak neuron
class LocalConvolution(nn.Module):

    # __init__ function is called when you instantiate this class.
    # Args: input_size   - A value denoting the height and width of input. Here, it will be RF size
    #       in_channels  - Number of input channels. Here, it is 2 since we use OnOff encoding
    #       out_channels - The number of neurons in a column
    #       kernel_size  - A value denoting the size of a RF taken as input by a single column
    #       stride       - Stride for convolving the kernel across the image
    # This function does not return anything.

    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride):
        super(LocalConvolution, self).__init__()
        self.input_size = to_pair(input_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_pair(kernel_size)
        self.stride = stride
        
        self.bias = None
        self.padding = 0
        self.dilation = 1
        self.groups = 1
        
        ########## UNCOMMENT AND COMPLETE THIS PART ##########
#         self.rows = ??
#         self.cols = ??
#         self.weight = ??
        ######################################################
        
        self.reset_weight()
    
    def reset_weight(self):
        # Resets weights to zero
        
        self.weight.zero_().int()

    def load_weight(self, target):
        # Loads weights with the target tensor
        # Args: target (Tensor=) - The target tensor
        
        self.weight.copy_(target)

    def forward(self, input):

        print(input.shape)
        print(input)

        return
    # forward function is called when you pass data (input) into the already instantiated class
    # Args: input - 4D spike wave tensor that was input to the Excitatory neurons.
    #               Its dimensions are (time,in_channels,height,width).
    #               Height and width are nothing but Receptive Field's height and width.
    #
    # Returns: out - output potential tensor corresponding to this layer's excitatory neuron potentials after convolving the
    #                synaptic weights with the input spike wave tensor (step-no-leak response).
    #                It should be a 4D tensor with dimensions (time, out_channels, rows, cols).
    
    # Since we don't use weight sharing, you have to work around the usual striding convolution essentially by manually
    # taking kernel_size patches from input and convolving it with the same size kernel. So, typically, you have to manually
    # stride across the input to create multiple columns.
        
        ### *** WRITE YOUR CONVOLUTION FUNCTION HERE *** ###
        
        
    


### This class should implement the STDP learning rule based on the decision tree branches given in your handout ###
class ModSTDP(nn.Module):
    
    # __init__ function is called when you instantiate this class.
    # Args: layer     - The layer for which this STDP classs will be instantiated.This is useful when you have deep SNNs.
    #       ucapture  - The 'capture' probability parameter
    #       uminus    - The 'minus' probability parameter
    #       usearch   - The 'search' probability parameter
    #       ubackoff  - The 'backoff' probability parameter
    #       umin      - The 'min' probability parameter used in weight stabilization
    #       maxweight - The maximum value/resolution of weights (weights can only be integers here)
    # This function does not return anything.
    
    def __init__(self, layer, ucapture, uminus, usearch, ubackoff, umin, maxweight):
        super(ModSTDP, self).__init__()
        # Initialize your variables here, including any Bernoulli Random Variable distributions
        
    


    # forward function is called when you pass data (input and output spikes) into the already instantiated class
    # Args: input_spikes - 4D spike wave tensor that was input to the Excitatory neurons. Its dimensions are
    #                      (time,in_channels,height,width). Height and width are nothing but Receptive Field's height and width
    #       output_spikes - 4D spike wave tensor that is the output after Lateral Inhibition
    # This function does not need to return anything.
    
    def forward(self, input_spikes, output_spikes):
        # Actual training rule goes here
        # Modify the weights of the corresponding layer in place
        return

class HopfieldNetwork(nn.Module):

    def __init__(self):
        super(HopfieldNetwork, self).__init__()

    def train_weights(self, train_patterns):
        self.num_neurons = train_patterns[0].size(0)*train_patterns[0].size(1)
        self.num_data = 1#len(train_patterns)
        self.W = torch.zeros(self.num_neurons,self.num_neurons)

        rho = torch.sum(torch.tensor([torch.sum(pattern) for pattern in train_patterns]))/(self.num_neurons*self.num_data)

        ones = torch.ones(self.num_neurons)
        minus_ones = -1*torch.ones(self.num_neurons)
        
        for pattern in train_patterns:
            pattern = pattern.reshape(self.num_neurons) - rho
            #pattern = torch.where(pattern>0,ones,minus_ones) #- rho
            self.W += torch.ger((pattern-4),(pattern-4))

        #print(self.W)
        self.W = self.W - torch.diag(torch.diag(self.W))
        self.W /= self.num_data
        #self.W = torch.clamp(self.W,0,1)

    def energy(self,state):
        return -0.5 * torch.matmul(torch.matmul(state.T,self.W),state) + torch.sum(state)

    def forward(self, input, num_iter=20):
        predicted=[]
        timesteps=torch.Tensor([-4.  , -3, -2 , -1,  0.  ,  1,  2 ,  3,  4])#[-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75, 1.])
        self.dim_size=int(torch.sqrt(torch.Tensor([self.num_neurons]))[0])
        for init_state in input:
            state=init_state
            energy=self.energy(state)
            #print(energy)
            print(torch.unique(init_state))

            activations=[]
            for step in range(num_iter):
                indices=torch.randperm(self.num_neurons)
                for idx in indices:
                    unnorm_activation = torch.matmul(self.W[idx],state)
                    print(unnorm_activation)
                    norm_activation = (unnorm_activation*4)/((self.num_neurons-1)*torch.max(self.W)) # Fix this normalization
                    activations.append(norm_activation)
                    print(norm_activation)
                    distance = torch.abs(norm_activation-timesteps)
                    print(distance)
                    closest_timestep_idx = torch.argmin(distance)
                    print(closest_timestep_idx)
                    state[idx] = timesteps[closest_timestep_idx]
                    print(state[idx])
                
                updated_energy = self.energy(state)
                #print(updated_energy)

                if energy == updated_energy:
                    predicted.append(state.reshape(self.dim_size,self.dim_size))
                    print(state)
                    break

                energy = updated_energy

            if len(predicted)==0:
                predicted.append(state.reshape(self.dim_size,self.dim_size))
                
            print(max(activations))
            print(min(activations))
            print(torch.max(self.W))
            print(torch.sum(self.W))
            print(self.num_neurons)

        return torch.stack(predicted)



        
############# YOUR CODE ENDS HERE ##########################
