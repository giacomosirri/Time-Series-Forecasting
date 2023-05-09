using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class RecurrentNeuralNetwork : NetworkModel
    {
        /* 
         * The number of features in the hidden state, i.e. the number of neurons in a layer.
         * The hidden state is the "memory" of a recurrent neural network, so having a lot of features in it 
         * allows to capture more complex patterns. However, the computational complexity of the training can 
         * become enormous really fast, so a good compromise must be taken. Using 16 for time series forecasting 
         * should be a good heuristic choice.
         */
        private const int HiddenSize = 16;
        /*
         * The number of layers in the network is an inner hyperparameter of the network, independent from hyperparameters
         * of the training algorithm. How to deal with this hyperparameter is still unclear.
         */
        internal int Layers { get; } = 1;

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
            // The Recurrent Neural Network's hidden state is initialized to zeros because the second parameter is null.
            // For time series forecasting, it is better to use the final hidden state instead of the final output.
            (Tensor _, Tensor finalHiddenState) = _rnn.forward(input, null);
            // The final hidden state of the last layer is passed to a fully connected linear layer to calculate the output values.
            return _linear.forward(finalHiddenState[-1]);
        }
    }
}
