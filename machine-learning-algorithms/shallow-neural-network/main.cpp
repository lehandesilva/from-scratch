#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Neural network from scratch, pure C++. Built for XOR, generalized to any
// fully-connected architecture (any number of layers / neurons per layer).
// Architecture is chosen by the layers you push into `network` in main().
// Every layer uses sigmoid; output loss is binary cross-entropy.

// --- Activations ---

// Sigmoid squashes any real number into (0, 1).
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// Sigmoid's derivative, expressed via its own output a = sigmoid(z).
double sigmoid_derivative(double a) {
    return a * (1.0 - a);
}

// --- Loss ---

// Binary cross-entropy for one output: y is the target, a is the prediction.
double cross_entropy(double y, double a) {
    return -(y * std::log(a) + (1.0 - y) * std::log(1.0 - a));
}

// --- Layer ---

// One fully-connected layer: a matrix of weights plus a bias per neuron.
struct Layer {
    std::vector<std::vector<double>> W;  // W[j][i]: weight from input i into neuron j
    std::vector<double> b;               // b[j]: bias of neuron j
    std::vector<double> input;           // cached inputs from the last forward pass
    std::vector<double> output;          // cached activations from the last forward pass
};

// Build a layer with random small weights (random init breaks neuron symmetry).
Layer makeLayer(int numInputs, int numNeurons, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Layer layer;
    layer.W.resize(numNeurons, std::vector<double>(numInputs));  // numNeurons rows, numInputs cols
    layer.b.resize(numNeurons);
    for (int j = 0; j < numNeurons; j++) {
        for (int i = 0; i < numInputs; i++) {
            layer.W[j][i] = dist(rng);  // random weight per connection
        }
        layer.b[j] = dist(rng);         // random bias per neuron
    }
    return layer;
}

// --- Forward pass ---

// Run one layer: weighted sum + bias, then sigmoid, for each neuron.
std::vector<double> forwardLayer(Layer& layer, const std::vector<double>& input) {
    layer.input = input;                       // cache inputs; backprop needs them
    layer.output.resize(layer.W.size());       // one activation per neuron
    for (size_t j = 0; j < layer.W.size(); j++) {
        double z = layer.b[j];                 // start the sum at the bias
        for (size_t i = 0; i < input.size(); i++) {
            z += layer.W[j][i] * input[i];     // accumulate weight * input (dot product)
        }
        layer.output[j] = sigmoid(z);          // squash to (0, 1)
    }
    return layer.output;
}

// Run the whole network: feed each layer's output into the next.
std::vector<double> forward(std::vector<Layer>& network, const std::vector<double>& x) {
    std::vector<double> a = x;                 // first layer's input is the raw data
    for (Layer& layer : network) {
        a = forwardLayer(layer, a);            // output of this layer feeds the next
    }
    return a;                                  // final layer's output = prediction
}

// --- Backward pass (backpropagation + gradient descent update) ---

// Compute gradients layer by layer (output -> input) and update weights in place.
void backward(std::vector<Layer>& network, const std::vector<double>& y, double lr) {
    // Output-layer error: sigmoid + cross-entropy collapse dL/dz to (a - y).
    Layer& last = network.back();
    std::vector<double> delta(last.output.size());
    for (size_t j = 0; j < delta.size(); j++) {
        delta[j] = last.output[j] - y[j];
    }

    // Walk layers from last to first.
    for (int l = static_cast<int>(network.size()) - 1; l >= 0; l--) {
        Layer& layer = network[l];
        std::vector<double> deltaPrev(layer.input.size(), 0.0);  // error to route to the previous layer

        for (size_t j = 0; j < layer.W.size(); j++) {
            for (size_t i = 0; i < layer.input.size(); i++) {
                deltaPrev[i] += delta[j] * layer.W[j][i];        // read old weight BEFORE updating it
                layer.W[j][i] -= lr * delta[j] * layer.input[i]; // gradient = error * input; step downhill
            }
            layer.b[j] -= lr * delta[j];                         // bias gradient is just the error
        }

        // Turn deltaPrev into the previous layer's delta via its sigmoid slope.
        if (l > 0) {
            Layer& prev = network[l - 1];
            for (size_t i = 0; i < deltaPrev.size(); i++) {
                deltaPrev[i] *= sigmoid_derivative(prev.output[i]);
            }
            delta = deltaPrev;                                   // becomes the error for the next iteration
        }
    }
}

int main() {
    // XOR dataset: inputs X and targets Y (Y rows are vectors to allow multiple outputs).
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

    // Build the network: change these push_back lines to change the architecture.
    std::mt19937 rng(42);                      // fixed seed = reproducible runs
    std::vector<Layer> network;
    network.push_back(makeLayer(2, 4, rng));   // hidden layer: 2 inputs -> 4 neurons
    network.push_back(makeLayer(4, 1, rng));   // output layer: 4 inputs -> 1 neuron

    double lr = 0.5;                           // learning rate (step size)
    int epochs = 10000;                        // full passes over the dataset

    // Training loop.
    for (int e = 0; e < epochs; e++) {
        double total_loss = 0.0;
        for (size_t n = 0; n < X.size(); n++) {
            std::vector<double> out = forward(network, X[n]);          // predict
            for (size_t k = 0; k < out.size(); k++) {
                total_loss += cross_entropy(Y[n][k], out[k]);          // measure error
            }
            backward(network, Y[n], lr);                               // learn from it
        }
        if (e % 1000 == 0) {
            std::cout << "epoch " << e << " loss " << total_loss / X.size() << std::endl;
        }
    }

    // Final check: predictions should be near the targets.
    std::cout << "\nFinal predictions:" << std::endl;
    for (size_t n = 0; n < X.size(); n++) {
        std::vector<double> out = forward(network, X[n]);
        std::cout << X[n][0] << " XOR " << X[n][1]
                  << " = " << out[0]
                  << "  (target " << Y[n][0] << ")" << std::endl;
    }
    return 0;
}
