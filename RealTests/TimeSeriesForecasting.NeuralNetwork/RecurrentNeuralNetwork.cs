using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class RecurrentNeuralNetwork : NetworkModel
    {
        /* 
         * The number of features in the hidden state. The hidden state is the "memory" of a recurrent neural network,
         * so having a lot of features in it allows to capture more complex patterns. However, the computational
         * complexity of the training can become enormous really fast, so a good compromise must be taken. Using 64
         * for time series forecasting should be a good heuristic choice.
         */
        private const int HiddenSize = 64;
        /*
         * The number of layers in the network is an inner hyperparameter of the network, independent from hyperparameters
         * of the training algorithm. How to deal with this hyperparameter is still unclear.
         */
        private const int Layers = 3;

        private readonly RNN _rnn;
        private readonly Linear _linear;

        /*
         * Create a new RNN model with weights and biases initialized with this distribution:
         * U(−sqrt(k), sqrt(k)), where k = 1 / hidden_size.
         */
        public RecurrentNeuralNetwork(long inputFeatures, long outputTimeSteps, long outputFeatures) : base(nameof(RNN))
        {
            _rnn = RNN(inputFeatures, HiddenSize, numLayers: Layers, batchFirst: true);
            _linear = Linear(HiddenSize, outputFeatures);
            RegisterComponents();
        }

        /*
         * Initialize the model loading its weights and biases from the given file.
         */
        public RecurrentNeuralNetwork(long inputFeatures, long outputTimeSteps, long outputFeatures, string path) 
            : this(inputFeatures, outputTimeSteps, outputFeatures) => load(path);

        public override Tensor forward(Tensor input)
        {
            // Initialize hidden state. Hidden state is a Tensor of shape (1, batch_size, hidden_size).
            Tensor initialHiddenState = zeros(Layers, input.size(0), HiddenSize);
            // For time series forecasting, it is better to use the final hidden state instead of the final output.
            (Tensor _, Tensor finalHiddenState) = _rnn.forward(input, initialHiddenState);
            // The final hidden state of the last layer is passed to a fully connected linear layer to calculate the output values.
            return _linear.forward(finalHiddenState[-1]);
        }
    }
}
