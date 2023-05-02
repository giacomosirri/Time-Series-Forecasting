using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class RecurrentNeuralNetwork : NetworkModel
    {
        private readonly RNN _rnn;
        private readonly Baseline _linear;

        public RecurrentNeuralNetwork(int inputObservations, int inputFeatures, int outputObservations, int outputFeatures) 
            : base(nameof(RNN))
        {
            _rnn = RNN(inputFeatures, 64, numLayers: 1, batchFirst: true);
            _linear = new Baseline(64, 1, outputObservations, outputFeatures);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            // Item1 is the predicted output, Item2 is the final hidden state, which can be ignored.
            Tensor output = _rnn.forward(input, null).Item1;
            Console.WriteLine(string.Join(",", output.shape));
            Tensor lastOutput = output.slice(1, output.shape[1]-1, output.shape[1], 1);
            return _linear.forward(lastOutput);
        }
    }
}
