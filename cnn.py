import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

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

def Relu(input):
    return np.maximum(0,input)



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

        #self.filters = np.random.randn(filter_num, filter_size, filter_size)
        self.filters = np.array([[[1, 0, -1],[2, 0, -2],[1, 0, -1]]])

    def forward(self, input):

        channels, height, width= input.shape


        # wzor na output -> output shape = inputshape + 2*padding - filter_size/stride + 1 
        # numpy colvolve, rollingwindow


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


        #output = self.activation(output)
        return output
    

    def backward(self, grad_out):
        pass


class Pooling_layer():
    def __init__(self, stride, size):
        self.size = size
        self.stride = stride

    def forward(self, input):
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

        return output_pool
    
    def backward(self, grad_out):
        pass

test_image = img.imread('Lenna_gray.jpg')
plt.imshow(test_image, cmap='gray')
plt.show()
test_image = np.expand_dims(test_image, axis=2)
test_image = test_image.T

print("Img shape: ", test_image.shape)
print("Filter shape: ", filter1.shape)


convolution = Conv_layer(1,3,1,0,Relu)


out_image = convolution.forward(test_image)

print("\nOut conv img shape: ", out_image.shape)
out_image = out_image.T

plt.imshow(out_image, cmap='gray')
plt.show()

pooling = Pooling_layer(2,2)
out_image = out_image.T

out_pooling = pooling.forward(out_image)
print("\nOut pooling img shape: ", out_pooling.shape)
out_pooling = out_pooling.T

plt.imshow(out_pooling, cmap='gray')
plt.show()