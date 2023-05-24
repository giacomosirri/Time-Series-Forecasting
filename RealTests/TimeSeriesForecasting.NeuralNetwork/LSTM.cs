using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class LSTM : NetworkModel
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

        private readonly TorchSharp.Modules.LSTM _rnn;
        private readonly Linear _linear;
        private readonly long _outputTimeSteps;
        private readonly long _outputFeatures;

        /*
         * Create a new RNN model with weights and biases initialized with this distribution:
         * U(−sqrt(k), sqrt(k)), where k = 1 / hidden_size.
         */
        public LSTM(long inputFeatures, long outputTimeSteps, long outputFeatures) : base(nameof(LSTM))
        {
            _outputTimeSteps = outputTimeSteps;
            _outputFeatures = outputFeatures;
            _rnn = LSTM(inputFeatures, HiddenSize, numLayers: Layers, batchFirst: true);
            _linear = Linear(HiddenSize, _outputTimeSteps * _outputFeatures);
            RegisterComponents();
        }

        /*
         * Initialize the model loading its weights and biases from the given file.
         */
        public LSTM(long inputFeatures, long outputTimeSteps, long outputFeatures, string path)
            : this(inputFeatures, outputTimeSteps, outputFeatures) => load(path);

        public override Tensor forward(Tensor input)
        {
            // The second parameter of the forward method call is null, so RNN's hidden state is initialized to zeros.
            // output.shape = [batchSize, inputTimeSteps, hiddenSize].
            (Tensor output, Tensor finalHiddenState, Tensor finalCellState) = _rnn.forward(input, null);
            // Note that: LastTimeStepOutput = finalHiddenState[-1].
            // See https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm for reference.
            // lastTimeStepOutput.shape = [batchSize, hiddenSize] --> the input time step dimension is narrowed to 1 and then squeezed.
            var lastTimeStepOutput = output.narrow(1, output.size(1) - 1, 1).squeeze();
            // The last time step output is passed to a linear layer to get the final output of the RNN.
            // finalOutput.shape = [batchSize, outputTimeStep, outputFeatures].
            return _linear.forward(lastTimeStepOutput).reshape(-1, _outputTimeSteps, _outputFeatures);
        }
    }
}
