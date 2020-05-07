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
        self.rows = int(((self.input_size[0]-self.kernel_size[0])/self.stride)+1) # fix later
        self.cols = int(((self.input_size[1]-self.kernel_size[1])/self.stride)+1) # fix later
        self.weight = torch.zeros(self.rows,self.cols,self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
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
        output = []
        output = torch.zeros(input.size(0),self.out_channels,self.rows,self.cols)
        for i in range(0,self.rows,self.stride):
            for j in range(0,self.cols,self.stride):

                result = fn.conv2d(input[:,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]],self.weight[i,j,:,:,:,:],bias=self.bias,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

                output[:,:,i,j] = result.squeeze(3).squeeze(2)

        return output
        
    


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
        self.layer = layer
        self.ucapture = Bernoulli(ucapture)
        self.uminus = Bernoulli(uminus)
        self.usearch = Bernoulli(usearch)
        self.ubackoff = Bernoulli(ubackoff)
        self.umin = Bernoulli(umin)
        self.maxweight = maxweight

    # forward function is called when you pass data (input and output spikes) into the already instantiated class
    # Args: input_spikes - 4D spike wave tensor that was input to the Excitatory neurons. Its dimensions are
    #                      (time,in_channels,height,width). Height and width are nothing but Receptive Field's height and width
    #       output_spikes - 4D spike wave tensor that is the output after Lateral Inhibition
    # This function does not need to return anything.
    
    def forward(self, all_input_spikes, all_output_spikes):
        # Actual training rule goes here
        # Modify the weights of the corresponding layer in place

        all_weights = self.W.copy()
        all_weights = torch.where(all_weights<0,torch.zeros(all_weights.size()),all_weights)
        all_weights = torch.where(all_weights>8,8*torch.ones(all_weights.size()),all_weights)

        all_output_spikes = 8-torch.sum(all_output_spikes,0)
        all_input_spikes = 8-torch.sum(all_input_spikes,0)

        #for r in range(0,self.layer.rows,self.layer.stride):
        #    for c in range(0,self.layer.cols,self.layer.stride):
        for neuron in range(self.num_neurons):
            #input_spikes = input_spikes[:,r:r+self.layer.kernel_size[0],c:c+self.layer.kernel_size[1]].unsqueeze(0).expand(self.layer.out_channels,-1,-1,-1)
            #output_spikes = output_spikes[:,r,c].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1,self.layer.in_channels,self.layer.kernel_size[0],self.layer.kernel_size[1])
            #weights = weights[r,c,:,:,:,:]
            input_spikes = all_input_spikes[neuron]
            output_spikes = all_output_spikes[neuron]
            weights = all_weights[neuron]
            
            fplus = Bernoulli((weights/8)*(2-(weights/8))).sample()
            fminus = Bernoulli((1-(weights/8))*(1+(weights/8))).sample()

            branch_one = torch.where((input_spikes < 8) & (output_spikes < 8) & (input_spikes <= output_spikes),weights+self.ucapture.sample(sample_shape=weights.size())*torch.max(fplus,self.umin.sample(sample_shape=weights.size())),weights)

            branch_two = torch.where((input_spikes < 8) & (output_spikes < 8) & (input_spikes > output_spikes),branch_one-self.uminus.sample(sample_shape=weights.size())*torch.max(fminus,self.umin.sample(sample_shape=weights.size())),branch_one)

            branch_three = torch.where((input_spikes < 8) & (output_spikes == 8),branch_two+self.usearch.sample(sample_shape=weights.size())*torch.max(fplus,self.umin.sample(sample_shape=weights.size())),branch_two)

            branch_four = torch.where((input_spikes == 8) & (output_spikes < 8),branch_three-self.ubackoff.sample(sample_shape=weights.size())*torch.max(fminus,self.umin.sample(sample_shape=weights.size())),branch_three)
            
            all_weights[neuron] = weights

        self.W = all_weights
        

class HopfieldNetwork(nn.Module):

    def __init__(self, ucapture, uminus, usearch, ubackoff, umin, maxweight):
        super(HopfieldNetwork, self).__init__()
        # self.norm_factor
        #self.resolution = resolution
        self.ucapture = Bernoulli(ucapture)
        self.uminus = Bernoulli(uminus)
        self.usearch = Bernoulli(usearch)
        self.ubackoff = Bernoulli(ubackoff)
        self.umin = Bernoulli(umin)
        self.maxweight = maxweight

    def hebbian_train_weights(self, train_patterns):
        self.num_neurons = train_patterns[0].size(0)*train_patterns[0].size(1)
        self.num_data = len(train_patterns)
        self.W = torch.zeros(self.num_neurons,self.num_neurons)

        #zeros=torch.zeros(784)
        #ones=torch.ones(784)

        for pattern in train_patterns:
            pattern = pattern.reshape(self.num_neurons) #- rho
            #pattern = torch.where(pattern>4,ones,zeros) #- rho
            self.W += torch.ger(pattern,pattern)

        #print(self.W)
        self.W = self.W - torch.diag(torch.diag(self.W))
        self.W /= self.num_data
        self.W /= 8#= torch.clamp(self.W,0,8)
        #self.W = torch.round_(self.W)


    def train_weights(self, train_patterns):
        self.hebbian_train_weights(train_patterns); return #remove this line if you want weight initialization of 0
        self.num_neurons = train_patterns[0].size(0)*train_patterns[0].size(1)
        self.num_data = len(train_patterns)
        self.W = torch.zeros(self.num_neurons,self.num_neurons)

        #uncomment next line for random weight initialization
        #self.W = torch.randint(0,8,(self.num_neurons,self.num_neurons)).type(torch.FloatTensor)


    def STDP_weights(self, all_input_spikes, all_output_spikes):
        # Actual training rule goes here

        all_weights = self.W.clone()

        all_output_spikes = 8-torch.sum(all_output_spikes,0)
        all_input_spikes = 8-torch.sum(all_input_spikes,0)


        for neuron in range(self.num_neurons):
            input_spikes = all_input_spikes[neuron]
            output_spikes = all_output_spikes[neuron]
            weights = all_weights[neuron]
            
            fplus = Bernoulli((weights/8)*(2-(weights/8))).sample()
            fminus = Bernoulli((1-(weights/8))*(1+(weights/8))).sample()

            branch_one = torch.where((input_spikes < 8) & (output_spikes < 8) & (input_spikes <= output_spikes),weights+self.ucapture.sample(sample_shape=weights.size())*torch.max(fplus,self.umin.sample(sample_shape=weights.size())),weights)

            branch_two = torch.where((input_spikes < 8) & (output_spikes < 8) & (input_spikes > output_spikes),branch_one-self.uminus.sample(sample_shape=weights.size())*torch.max(fminus,self.umin.sample(sample_shape=weights.size())),branch_one)

            branch_three = torch.where((input_spikes < 8) & (output_spikes == 8),branch_two+self.usearch.sample(sample_shape=weights.size())*torch.max(fplus,self.umin.sample(sample_shape=weights.size())),branch_two)

            branch_four = torch.where((input_spikes == 8) & (output_spikes < 8),branch_three-self.ubackoff.sample(sample_shape=weights.size())*torch.max(fminus,self.umin.sample(sample_shape=weights.size())),branch_three)
            
            all_weights[neuron] = branch_four

        all_weights = torch.clamp(all_weights,0,8)
        self.W = all_weights


    def energy(self,state):
        return -0.5 * torch.matmul(torch.matmul(state.T,self.W),state) + torch.sum(state)

    def forward(self, input, threshold=0, num_iter=20, noSTDP=False):
        predicted=[]
        max_timestep=8

        self.dim_size=28
        #thresholds=[240,240,240,500,400,300,200,100]
        for init_state in input:
            state=init_state.clone()
            energy=self.energy(torch.sum(state,dim=0))

            for step in range(num_iter):
                indices=torch.randperm(self.num_neurons)
                for idx in indices:
                    #print('index:', idx)
                    output_time=-1
                    new_neuron_state=torch.zeros(state.size(0))
                    for time in range(max_timestep):
                        current_membrane_potential = torch.matmul(self.W[idx],state[time])
                        #print('time:',time)
                        #print(state[time])
                        if current_membrane_potential >= threshold: #(max_timestep-time)*threshold:
                            output_time = time
                            #print('index:',idx)
                            #print('time', output_time)
                            break
                    if output_time != -1:
                        new_neuron_state[output_time:max_timestep] = 1

                    state[:,idx]=new_neuron_state
                                    
                updated_energy = self.energy(torch.sum(state,dim=0))

                if energy == updated_energy:
                    predicted.append(state.reshape(8,self.dim_size,self.dim_size))
                    print("took \t ", step, " iterations")
                    break

                
                energy = updated_energy

            if len(predicted)==0:
                predicted.append(state.reshape(8,self.dim_size,self.dim_size))
                
            # print(torch.max(self.W))
            # print(torch.min(self.W))
            # print(torch.sum(self.W))
            #print(timesteps)
            if (noSTDP):
                continue
            self.STDP_weights(init_state,predicted[0].reshape(8,784))


        return torch.stack(predicted)

