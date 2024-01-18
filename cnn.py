import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


# einsum  rollingwindow, asstrided, reshape



#dolnoprzpustowy
filter1 = np.array([[[1, 1, 1],
                   [1, 4, 1],
                   [1, 1, 1]]])

#gornoprzepustowy
filter2 = np.array([[[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]]])

#sobel
filter3 = np.array([[[1,   2,  1],
                   [0,   0,  0],
                   [-1, -2, -1]]])

#negativ
filter4 = np.array([[[-1, -1, -1],
                    [-1, -1, -1],
                    [-1, -1, -1]]])

#sobel_l
filter_5 = np.array([[
                    [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]])



class Relu():
    def forward(self, input):
        return np.maximum(0,input)

    def backward(self, input):
        return (1 * (input > 0))


class Softmax():
    def forward(self, input):
        e_x = np.exp(input - np.max(input))

        return e_x / np.sum(e_x, axis=0)

    def backward(self, input, grad_output):
        e_x = np.exp(input - np.max(input))
        out = e_x / np.sum(e_x, axis=0)

        return (out*(1-out))*grad_output

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
    def __init__(self, filter_num, filter_size, stride, padding, activation) -> None:
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        #self.filters = np.random.randn(filter_num, input_size, filter_size, filter_size)
        self.filters = np.array([[[[1, 0, -1],[2, 0, -2],[1, 0, -1]]]])

    def get_windows(self, input):

        batch_size, channels, height, width = input.shape
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

        output = np.einsum('bchwkt,fckt->bfhw', windows, self.filters)


        #output = self.activation.forward(output)
        return output
    

    def backward(self, grad_out, lr):
        
        batch_size, channels, height, width= grad_out.shape    

        grad_out = self.activation.backward(grad_out)

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

        self.filters -= lr * grad_filter

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
        windows_pool = self.get_windows(input)
        out_pooling = windows_pool.max(axis=(4,5))

        return out_pooling
    
    def backward(self, grad_out):
        
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
        input = input.flatten() 
        self.input = input
        #print(input.shape)
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
        self, input_size, output_size, optimizer, 
        hidden_layers=[128], 
        output_activation=Softmax, 
        conv_activation=Relu
    ) -> None:

        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer

        self.conv_activation = conv_activation
        self.output_activation = output_activation

        self.loss_object = CrossEntropyLoss()

        
        self.layers = hidden_layers
        self.layer_1 = Conv_layer(1,3,1,0,conv_activation)
        self.layer_2 = Pooling_layer(2,2)
        self.layer_3 = FullyConnectedLayer(input_size, self.output_size, self.output_activation, self.optimizer)

    def forward(self, input):

        out1 = self.layer_1.forward(input)
        out2 = self.layer_2.forward(out1)
        out3 = self.layer_3.forward(out2)
        return out3
    

    def backward(self, output, target):

        loss, gradient = self.loss_object.forward(output, target)

        grad_1 = self.layer_3.backward(gradient)
        grad_2 = self.layer_2.backward(grad_1)
        grad_3 = self.layer_1.backward(grad_2)

        return loss



test_image = img.imread('Lenna_gray.jpg')
plt.imshow(test_image, cmap='gray')
plt.show()
test_image = np.expand_dims(test_image, axis=2)
test_image = np.expand_dims(test_image, axis=3)
test_image = test_image.T

print("Img shape: ", test_image.shape)
print("Filter shape: ", filter1.shape)


convolution = Conv_layer(1,3,1,0,Relu)


out_image = convolution.forward(test_image)

print("\nOut conv img shape: ", out_image.shape)
out_image = np.squeeze(out_image, axis=0)
print("\nOut conv img shape: ", out_image.shape)
out_image = out_image.T

plt.imshow(out_image, cmap='gray')
plt.show()

pooling = Pooling_layer(2,2)
out_image = np.expand_dims(out_image, axis=3)
out_image = out_image.T


out_pooling = pooling.forward(out_image)
print("\nOut pooling img shape: ", out_pooling.shape)
out_pooling = np.squeeze(out_pooling, axis=0)
print("\nOut conv img shape: ", out_pooling.shape)
out_pooling = out_pooling.T

plt.imshow(out_pooling, cmap='gray')
plt.show()

