using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class RecurrentNeuralNetwork : NetworkModel
    {
        private readonly RNN _rnn;
        private readonly SimpleNeuralNetwork _linear;

        public RecurrentNeuralNetwork(long inputFeatures, long outputObservations, long outputFeatures, long layers) : base(nameof(RNN))
        {
            _rnn = RNN(inputFeatures, 64, numLayers: layers, batchFirst: true);
            _linear = new SimpleNeuralNetwork(64, 1, outputObservations, outputFeatures);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            // Item1 is the predicted output, Item2 is the final hidden state, which can be ignored.
            Tensor output = _rnn.forward(input, null).Item1;
            Tensor lastOutput = output.slice(1, output.shape[1]-1, output.shape[1], 1);
            return _linear.forward(lastOutput);
        }
    }
}
