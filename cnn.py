import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


# einsum  rollingwindow, asstrided, reshape


class Relu():
    def forward(self, input):
        return np.maximum(0,input)

    def backward(self, input, grad_output):
        return (1 * (input > 0))*grad_output


class Softmax():
    def forward(self, input):
        e_x = np.exp(input - np.max(input))

        return e_x / np.sum(e_x, axis=0)

    def backward(self, input, grad_output):
        out = input * (1 - input)

        return out * grad_output

class SGD:
    def __init__(self, learning_rate, clip_range=10) -> None:

        self.learning_rate = learning_rate
        self.clip_range = clip_range

    def update_weights(self, weights: np.ndarray, weights_grad: np.ndarray) -> np.ndarray:

        clipped_grad = np.clip(weights_grad, -self.clip_range, self.clip_range)
        updated_weights = weights - self.learning_rate * clipped_grad
        return updated_weights


class CrossEntropyLoss:
    def __init__(self, stability=1e-10) -> None:

        self.stability = stability

    def forward(self, predictions: np.ndarray, target: np.ndarray) -> float:

        loss = -np.sum(target * np.log(predictions+self.stability))
 
        grad = -target/(predictions+self.stability)
        loss = np.mean(loss)
        grad = np.mean(grad)

        return loss, grad 
    

class Conv_layer():
    """
    input_shape - rozmiar wejscia
    filter_num - ilosc filtrow
    filter_size - wilekosc filtra
    stride - krok, czyli o ile pikseli jądro powinno zostać przesunięte na raz
    padding - sposób obsługi obramowania próbki, dodanie dodatkowych wag na krawędziach 
    umozliwia otrzymanie rozmiaru wyjścia takiego samego jak rozmiar wejścia jesli krok=1
    
    """
    def __init__(self, input_size, filter_num, filter_size, stride, padding, activation) -> None:
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding  
        self.activation = activation
        self.input_size = input_size
        self.filters = np.random.randn(self.filter_num, self.input_size, self.filter_size, self.filter_size)
        #self.filters = np.array([[[[1, 0, -1],[2, 0, -2],[1, 0, -1]]]])
        self.bias = np.zeros(self.out_channels)

    def get_windows(self, input):

        batch_size, channels, height, width = input.shape

        if self.padding != 0:
            input = np.pad(input, pad_width=((0,), (0,), (self.padding,), (self.padding,)), mode='constant', constant_values=(0.,))
        batch_stride, channels_stride, r_stride, k_stride = input.strides

        out_h = int(((height + (2 * self.padding) - self.filter_size)/self.stride))+1
        out_w = int(((width + (2 * self.padding) - self.filter_size)/self.stride))+1

        window_shape = (batch_size, channels, out_h, out_w, self.filter_size, self.filter_size)
        strides_shape = (batch_stride, channels_stride, self.stride * r_stride, self.stride * k_stride, r_stride, k_stride)

        return np.lib.stride_tricks.as_strided(input, window_shape, strides_shape)


    def forward(self, input):
        self.input = input
        #channels, height, width= input.shape


        # wzor na output -> output shape = inputshape + 2*padding - filter_size/stride + 1 
        # numpy colvolve, rollingwindow
        # slow ver
        """
        out_hw = int(((height + (2 * self.padding) - self.filter_size)/self.stride))+1

        output = np.zeros((self.filter_num, out_hw, out_hw))

        for f in range(self.filter_num):
            
            for h in range(out_hw):
                height_start = h * self.stride
                height_end = height_start + self.filter_size

                for w in range(out_hw):
                    width_start = w * self.stride
                    width_end = width_start + self.filter_size

                    update = input[:, height_start:height_end, width_start:width_end]
                    output[f, h, w] = np.sum(update * self.filters)
        """

        # fast version einsum

        windows = self.get_windows(input)
        self.windows = windows

        output = np.einsum('bchwkt,fckt->bfhw', windows, self.filters)
        output += self.bias[None, :, None, None]

        output = self.activation.forward(output)
        self.output = output
        return output
    

    def backward(self, grad_out, lr):
        
        # slow ver
        """
        batch_size, channels, height, width= grad_out.shape    

        grad_out = self.activation.backward(self.output, grad_out)

        grad_input = np.zeros(self.input.shape)
        grad_filter = np.zeros(self.filters.shape)

        for i in range (batch_size):
            for c in range(channels):
                
                for h in range(height):
                    height_start = h * self.stride
                    height_end = height_start + self.filter_size

                    for w in range(width):               
                        width_start = w * self.stride
                        width_end = width_start + self.filter_size

                        update = self.input[:, height_start:height_end, width_start:width_end]
                        grad_filter[c] += np.sum(grad_out[i, c, h, w]*update)

                        grad_input[:, :, height_start:height_end, width_start:width_end] += grad_out[i, c, h, w] * self.filters[c]
        """
        # fast ver

        grad_windows = self.get_windows(grad_out)
        rot_filter = np.rot90(self.filters, 2, axes=(2, 3))

        grad_filters = np.einsum('bchwkt,bfhw->fckt', self.windows, grad_out)
        grad_input = dx = np.einsum('bfhwkt,fckt->bchw', grad_windows, rot_filter)
        grad_bias = db = np.sum(grad_out, axis=(0, 2, 3))


        self.filters -= lr * grad_filters
        self.bias -= lr * grad_bias

        return grad_input
    

class Pooling_layer():
    def __init__(self, stride, size):
        self.size = size
        self.stride = stride
        self.padding = 0

    def get_windows(self, input):

        batch_size, channels, height, width = input.shape
        batch_stride, channels_stride, r_stride, k_stride = input.strides

        out_h = int(((height + (2 * self.padding) - self.size)/self.stride))+1
        out_w = int(((width + (2 * self.padding) - self.size)/self.stride))+1

        window_shape = (batch_size, channels, out_h, out_w, self.size, self.size)
        strides_shape = (batch_stride, channels_stride, self.stride * r_stride, self.stride * k_stride, r_stride, k_stride)

        return np.lib.stride_tricks.as_strided(input, window_shape, strides_shape)

    def forward(self, input):
        self.input = input

        # slow ver
        """
        channels, height, width= input.shape

        # 2x2 output 2 razy mniejszy
        out_hw = int((height - self.size)/self.stride)+1

        output_pool = np.zeros((channels, out_hw, out_hw))


        for i in range(channels):

            for h in range(out_hw):
                height_start = h * self.stride
                height_end = height_start + self.size                

                for w in range(out_hw):
                    width_start = w * self.stride
                    width_end = width_start + self.size

                    update = input[i, height_start:height_end, width_start:width_end]
                    # max pooling
                    output_pool[i, h, w] = np.max(update)
                    # avg
                    #output_pool[i, h, w] = np.mean(update)
        """
        # fast ver
        windows_pool = self.get_windows(input)
        out_pooling = windows_pool.max(axis=(4,5))

        return out_pooling
    

    def backward(self, grad_out):
        
        # do dodania fast unpulling

        grad_input = np.zeros(self.input.shape)
        batch_size, channels, height, width= grad_out.shape

        for i in range (batch_size):
            for c in range(channels):
                        
                for h in range(height):
                    height_start = h * self.stride
                    height_end = height_start + self.size

                    for w in range(width):               
                        width_start = w * self.stride
                        width_end = width_start + self.size

                        update = self.input[i, c, height_start:height_end, width_start:width_end]

                        x = update == np.max(update)
                        grad_input[i, c, height_start:height_end, width_start:width_end] = grad_out * x

        return grad_input
    

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation, optimizer) -> None:

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.optimizer = optimizer
        
        self.weights = np.random.randn(input_size, output_size)
        self.bias = 0

    def forward(self, input): 
        self.input = input

        intermediate = np.dot(input, self.weights) + self.bias
        #intermediate = (self.weights.T * input) + self.bias
        self.intermediate = intermediate
        output = self.activation.forward(self,intermediate)
        return output
    
    def backward(self, grad_output):

        grad = self.activation.backward(self, self.intermediate, grad_output)
        input_grad = np.dot(grad, self.weights.T)
        weight_grad = np.dot(self.input.T, grad)
        bias_grad = np.sum(grad)

        self.weights = self.optimizer.update_weights(self.weights, weight_grad)
        self.bias = self.optimizer.update_bias(self.bias, bias_grad)


        return input_grad



class Full_Net:
    def __init__(
        self, input_size, output_size, optimizer, lr, 
        hidden_layers=[128,64], 
        output_activation=Softmax, 
        activation=Relu,
    ) -> None:

        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.learnig_rate = lr
        self.activation = activation
        self.output_activation = output_activation

        self.loss_object = CrossEntropyLoss()

        
        self.layers = hidden_layers
        # conv_layer (input_size, filter_num, filter_size, stride, padding, activation)
        self.conv1 = Conv_layer(3, 64, 3, 1, 1, self.activation)
        self.conv2 = Conv_layer(64, 128, 3, 1, 1, self.activation)
        self.pooling1 = Pooling_layer(2,2)
        self.pooling2 = Pooling_layer(2,2)
        self.fclayer_1 = FullyConnectedLayer(input_size, self.output_size, self.activation, self.optimizer)
        self.fclayer_2 = FullyConnectedLayer(self.layers[0], self.layers[1], self.activation, self.optimizer)
        self.fclayer_3 = FullyConnectedLayer(self.layers[1], self.output_size, self.output_activation, self.optimizer)


    def forward(self, input):

        conv1 = self.conv1.forward(input)       # (batch_size, 64, 32, 32)
        pool1 = self.pooling1.forward(conv1)    # (batch_size, 64, 16, 16)   

        conv2 = self.conv1.forward(pool1)       # (batch_size, 128, 16, 16)
        pool2 = self.pooling2.forward(conv2)    # (batch_size, 128, 8, 8)

        self.shape_after_pool = pool2.shape
        flatten = pool2.reshape(self.pool2_shape[0], -1)    # (batch_size, 128*8*8)

        fclayer1 = self.fclayer_1.forward(flatten)      # (batch_size, 128)
        fclayer2 = self.fclayer_2.forward(fclayer1)     # (batch_size, 64)
        fclayer3 = self.fclayer_3.forward(fclayer2)     # (batch_size, 10)

        return fclayer3
    

    def backward(self, output, target):

        loss, gradient = self.loss_object.forward(output, target)

        grad = self.fclayer_3.backward(gradient)
        grad = self.fclayer_2.backward(grad)
        grad = self.fclayer_1.backward(grad)

        grad = grad.reshape(self.shape_after_pool)
        grad = self.pooling2.backward(grad)
        grad = self.conv2.backward(grad, self.learnig_rate)
        grad = self.pooling1.backward(grad)
        grad = self.conv1.backward(grad, self.learnig_rate)

        return loss



