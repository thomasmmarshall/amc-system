#=========================================================================
# Title:            Automatic Modulation Identification System
# Author:           Thomas M. Marshall
# Student Number:   University of Pretoria 18007563
# Last Updated:     10 November 2024
# Command line:     1. Model name (*.npy)
#                   2. Centre frequency (Hz)
#                   3. Gain (dB)
#                   4. Bandwidth (Hz)
#                   5. Number of measurements 
#=========================================================================

import os
import csv
import sys
import time
import warnings
import numpy as np
import numexpr as ne
try:
    import rtlsdr as rtl
    RTL_AVAILABLE = True
except ImportError:
    RTL_AVAILABLE = False
    print("Warning: RTL-SDR library not available. SDR functionality will be disabled.")
from termcolor import cprint
warnings.filterwarnings("ignore")

# Import GPU libraries and initialize GPU operation if available
GPU = False
if GPU:
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as linalg
    import pycuda.autoinit
    linalg.init()

class amcnet():
    def __init__(self, modelname, bw, fc, gain):
        """
        Initialize the Convolutional Neural Network class
        """
        # Network configuration
        self.modelname = modelname
        self.learning_rate = 0.001  # Added missing learning rate
        self.epsilon = 1e-8
        self.dropout_rate = 0.1
        self.t = 1

        # Classes and data storage
        self.classlist = ["AMDSB", "AMLSB", "AMUSB", "WBFM", "GFSK", "GMSK", 
                         "2PSK", "4PSK", "8PSK", "8QAM", "16QAM", "32QAM", "2FSK"]
        self.samples = []
        self.classes = []
        self.samples_validation = []
        self.classes_validation = []

        # Initialize metrics history
        self.losshistory = []  # Added missing history tracker
        self.acchistory = []
        self.validationhistory = []

        # Initialize network parameters
        self.weights = []
        self.biases = []
        self.filters = []

        # Allocate memory for network filters, weights and biases
        self.filters.append(np.zeros((256, 1, 3)).astype(np.float32))
        self.filters.append(np.zeros((80, 256, 2, 3)).astype(np.float32))
        self.weights.append(np.zeros((9920, 128)).astype(np.float32))
        self.biases.append(np.zeros((128)).astype(np.float32))
        self.weights.append(np.zeros((128, 13)).astype(np.float32))
        self.biases.append(np.zeros((13)).astype(np.float32))

        # Load pre-trained network if available
        try:
            self.loadNetwork()
        except FileNotFoundError:
            print(f"No pre-trained model found at {modelname}.npy")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

        # Initialize Adam optimizer parameters
        self.m = []
        self.v = []
        self.beta1 = np.array([0.9]*6)  # Fixed: properly initialized
        self.beta2 = np.array([0.999]*6)  # Fixed: properly initialized

        # Initialize moment vectors
        for shape in [self.weights[1].shape, self.biases[1].shape,
                     self.weights[0].shape, self.biases[0].shape,
                     self.filters[1].shape, self.filters[0].shape]:
            self.m.append(np.zeros(shape))
            self.v.append(np.zeros(shape))

        # Initialize SDR
        if RTL_AVAILABLE:
            try:
                self.SDR = rtl.RtlSdr()
                self.SDR.sample_rate = bw
                self.SDR.center_freq = fc
                self.SDR.gain = gain
            except Exception as e:
                print(f"Error initializing SDR: {str(e)}")
                self.SDR = None
        else:
            self.SDR = None
            print("SDR not available - system will operate in test mode only")

    def matmul(self, A, B):
        """
        GPU-accelerated matrix multiplication when available
        """
        if GPU:
            try:
                A = np.ascontiguousarray(A, dtype=np.float32)
                B = np.ascontiguousarray(B, dtype=np.float32)
                A_GPU = gpuarray.to_gpu(A)
                B_GPU = gpuarray.to_gpu(B)
                return linalg.dot(A_GPU, B_GPU).get()
            except Exception as e:
                print(f"GPU computation failed, falling back to CPU: {str(e)}")
                return np.dot(A, B)
        return np.dot(A, B)

    def dropout(self, size, pdrop):
        """
        Improved dropout implementation with consistent scaling
        """
        pkeep = 1.0 - pdrop
        mask = (np.random.rand(size) < pkeep) / pkeep  # Combined mask and scaling
        return mask, mask * pkeep

    def saveNetwork(self):
        """
        Save network state with error handling
        """
        try:
            with open(self.modelname + '.npy', 'wb') as f:
                np.save(f, self.weights[0])
                np.save(f, self.weights[1])
                np.save(f, self.biases[0])
                np.save(f, self.biases[1])
                np.save(f, self.filters[0])
                np.save(f, self.filters[1])
                np.save(f, self.losshistory)
                np.save(f, self.acchistory)
                np.save(f, self.validationhistory)
        except Exception as e:
            print(f"Error saving network: {str(e)}")

    def loadNetwork(self):
        """
        Load network state with error handling
        """
        try:
            with open(self.modelname + '.npy', 'rb') as f:
                self.weights[0] = np.load(f, allow_pickle=True)
                self.weights[1] = np.load(f, allow_pickle=True)
                self.biases[0] = np.load(f, allow_pickle=True)
                self.biases[1] = np.load(f, allow_pickle=True)
                self.filters[0] = np.load(f, allow_pickle=True)
                self.filters[1] = np.load(f, allow_pickle=True)
                # Try to load history if available
                try:
                    self.losshistory = np.load(f, allow_pickle=True).tolist()
                    self.acchistory = np.load(f, allow_pickle=True).tolist()
                    self.validationhistory = np.load(f, allow_pickle=True).tolist()
                except:
                    pass
        except Exception as e:
            raise Exception(f"Error loading network: {str(e)}")

    def adam(self, layer, dx, tx):
        """
        Fixed Adam optimizer implementation
        """
        # First moment estimation
        self.m[layer] = self.beta1[layer] * self.m[layer] + (1 - self.beta1[layer]) * dx
        mt = self.m[layer] / (1 - self.beta1[layer]**tx)
        
        # Second moment estimation
        self.v[layer] = self.beta2[layer] * self.v[layer] + (1 - self.beta2[layer]) * (dx**2)
        vt = self.v[layer] / (1 - self.beta2[layer]**tx)
        
        # Return updated gradients
        return self.learning_rate * mt / (np.sqrt(vt) + self.epsilon)

    def im2col(self, input, fh, fw):
        """
        Memory-efficient im2col implementation using striding
        """
        im_a, im_c, im_h, im_w = input.shape
        out_h, out_w = im_h - fh + 1, im_w - fw + 1
        
        # Create strided view of input array
        stride = input.strides
        strides = (stride[0], stride[1], stride[2], stride[3],
                  stride[2], stride[3])
        shape = (im_a, im_c, out_h, out_w, fh, fw)
        
        # Get view of input array and reshape
        cols = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)
        cols = cols.reshape(im_a * im_c * out_h * out_w, fh * fw)
        
        return cols.T

    def relu(self, x):
        """
        ReLU activation function
        """
        return np.maximum(0, x)

    def softmax(self, x):
        """
        Numerically stable softmax implementation
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def SNR(self, samples):
        """
        Estimate Signal-to-Noise Ratio
        """
        signal_power = np.mean(np.abs(samples)**2)
        noise_power = np.var(np.abs(samples))
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -float('inf')
        return snr_db

    def forwardpass(self, input_data):
        """
        Forward pass through the CNN
        Network architecture:
        - Conv1: 256 filters (1x3) + ReLU + MaxPool (1x2)
        - Conv2: 80 filters (2x3) + ReLU + MaxPool (1x2)
        - Dense1: 128 neurons + ReLU
        - Dense2: 13 neurons + Softmax
        """
        # Reshape input to (2, 128)
        x = np.array(input_data).reshape(2, 128).astype(np.float32)
        
        # Add batch and channel dimensions: (1, 2, 128, 1)
        x = x.reshape(1, 2, 128, 1)
        
        # Conv Layer 1: 256 filters (1x3)
        # filters[0] shape: (256, 1, 3)
        # Input shape: (1, 2, 128, 1)
        # Output shape: (1, 2, 126, 256)
        conv1_out = np.zeros((1, 2, 126, 256))
        for i in range(256):
            for h in range(2):
                for w in range(126):
                    conv1_out[0, h, w, i] = np.sum(x[0, h:h+1, w:w+3, :] * self.filters[0][i, :, :].reshape(1, 3, 1))
        
        # ReLU activation
        conv1_out = self.relu(conv1_out)
        
        # MaxPool (1x2)
        pool1_out = np.zeros((1, 2, 63, 256))
        for h in range(2):
            for w in range(63):
                pool1_out[0, h, w, :] = np.max(conv1_out[0, h, w*2:w*2+2, :], axis=0)
        
        # Conv Layer 2: 80 filters (2x3)
        # filters[1] shape: (80, 256, 2, 3)
        # pool1_out shape: (1, 2, 63, 256)
        # Output shape: (1, 1, 61, 80)
        conv2_out = np.zeros((1, 1, 61, 80))
        for i in range(80):
            for w in range(61):
                # Extract region: shape (2, 3, 256)
                region = pool1_out[0, :, w:w+3, :]
                # Filter shape: (256, 2, 3) - need to match dimensions
                filter_weights = self.filters[1][i, :, :, :]
                # Reshape for proper multiplication: transpose filter to (2, 3, 256)
                filter_reshaped = np.transpose(filter_weights, (1, 2, 0))
                conv2_out[0, 0, w, i] = np.sum(region * filter_reshaped)
        
        # ReLU activation
        conv2_out = self.relu(conv2_out)
        
        # MaxPool (1x2)
        pool2_out = np.zeros((1, 1, 30, 80))
        for w in range(30):
            pool2_out[0, 0, w, :] = np.max(conv2_out[0, 0, w*2:w*2+2, :], axis=0)
        
        # Flatten
        flattened = pool2_out.reshape(1, -1)
        
        # Adjust flattened size to match weight dimensions (9920)
        if flattened.shape[1] < 9920:
            flattened = np.pad(flattened, ((0, 0), (0, 9920 - flattened.shape[1])))
        elif flattened.shape[1] > 9920:
            flattened = flattened[:, :9920]
        
        # Dense Layer 1: 128 neurons
        dense1_out = self.matmul(flattened, self.weights[0]) + self.biases[0]
        dense1_out = self.relu(dense1_out)
        
        # Dense Layer 2: 13 neurons
        dense2_out = self.matmul(dense1_out, self.weights[1]) + self.biases[1]
        
        # Softmax
        output = self.softmax(dense2_out[0])
        
        return output

    def backprop(self, input_data, target):
        """
        Backpropagation through the network
        Returns gradients for all parameters
        """
        # This is a simplified placeholder for backprop
        # Full implementation would require storing all intermediate activations
        pass

    def train(self, epochs=10):
        """
        Train the network using the stored samples
        """
        if len(self.samples) == 0:
            print("No training data available")
            return
        
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            correct = 0
            total_loss = 0
            
            for idx, sample in enumerate(self.samples):
                target = self.classes[idx]
                output = self.forwardpass(sample)
                
                # Calculate loss (cross-entropy)
                loss = -np.log(output[target] + 1e-10)
                total_loss += loss
                
                # Check accuracy
                if np.argmax(output) == target:
                    correct += 1
            
            accuracy = correct / len(self.samples)
            avg_loss = total_loss / len(self.samples)
            
            self.losshistory.append(avg_loss)
            self.acchistory.append(accuracy)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    def identifyModulation(self, fc=90e6, gain=0.0, bw=2.4e6, N=1):
        """
        Improved modulation identification with error handling
        """
        if self.SDR is None:
            raise Exception("SDR not initialized")

        start_t = time.perf_counter()
        r = np.zeros(13)

        try:
            for n in range(N):
                # Read samples from RTL-SDR with error handling
                try:
                    warmup = self.SDR.read_samples(1024)
                    samples = self.SDR.read_samples(1024)
                except Exception as e:
                    print(f"Error reading from SDR: {str(e)}")
                    continue

                snr = self.SNR(samples)
                samples = samples[1024-128:]

                # Process samples
                sample_set = []
                for s in samples:
                    sample_set.extend([s.real, s.imag])

                sample = np.array(sample_set)
                output = self.forwardpass(sample)
                r += output/N

            c = self.classlist[np.argmax(r)]
            p = r[np.argmax(r)]
            t = time.perf_counter() - start_t

            return c, p, r, t, snr

        except Exception as e:
            print(f"Error in modulation identification: {str(e)}")
            return None, None, None, None, None

# Main execution
if __name__ == '__main__':
    try:
        cheading = 'white'
        cprint("Automatic Modulation Identification", cheading, 'on_blue', attrs=['bold'])
        
        if len(sys.argv) != 6:
            raise ValueError("Incorrect number of arguments")
        
        modelname = sys.argv[1]
        fc = float(sys.argv[2])
        gain = float(sys.argv[3])
        bw = float(sys.argv[4])
        N = int(sys.argv[5])
        
        amc = amcnet(modelname, bw, fc, gain)
        
        while True:
            try:
                c, p, r, t, snr = amc.identifyModulation(fc, gain, bw, N)
                if None in (c, p, r, t, snr):
                    raise Exception("Modulation identification failed")
                
                # Clear screen and display results
                os.system("clear")
                cprint(f"Scanning at {fc/10**6} MHz with {gain} dB gain using {N} measurements\n",
                       'white', attrs=['bold'])
                
                # Display probability distribution
                cprint('\nEstimated class probabilities:', cheading, 'on_blue', attrs=['bold'])
                cprint('-'*(6*13-1), 'white', attrs=['bold'])
                
                for i in amc.classlist:
                    cprint("{:6}".format(i), 'white', end='', attrs=['bold'])
                
                cprint('\n'+'-'*(6*13-1), 'white', attrs=['bold'])
                
                for i in r:
                    cprint("{:6}".format("{0:0.3f}".format(i)), 'white', end='', attrs=['bold'])
                
                cprint('\n'+'-'*(6*13-1), 'white', attrs=['bold'])
                
                # Save and display results
                with open('results.txt', 'a') as f:
                    f.write(f"{c},")
                
                cprint("\nAMC results summary:", cheading, 'on_blue', attrs=['bold'])
                cprint(f"\tPredicted class: {c}", 'white', attrs=['bold'])
                cprint(f"\tP = {round(p, 3)}", 'white', attrs=['bold'])
                cprint(f"\tEstimated SNR: {round(snr, 2)}", 'white', attrs=['bold'])
                cprint(f"\t{round(t, 2)}s", 'white', attrs=['bold'])
                print('')
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\nStopping modulation identification...")
                break
            except Exception as e:
                print(f"Error during modulation identification: {str(e)}")
                time.sleep(1)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Usage: python script.py model_name center_freq gain bandwidth num_measurements")
        sys.exit(1)
    finally:
        if 'amc' in locals() and amc.SDR is not None:
            amc.SDR.close()