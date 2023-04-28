using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class Baseline : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _linear;
        private readonly string _filePath = "C:\\Users\\sirri\\Desktop\\Coding\\Tirocinio\\TorchSharp\\RealTests\\Logs\\weights_biases.txt";

        public Baseline(long inputObservations, long inputFeatures, long outputObservations, long outputFeatures) : base(nameof(Baseline))
        {
            _linear = Linear(inputObservations * inputFeatures, outputObservations * outputFeatures);
            RegisterComponents();
            LogState("Initial parameters");
        }

        public void LogState(string message)
        {
            var tl = new TensorLogger(_filePath);
            _linear.state_dict().AsEnumerable().ToList().ForEach(state => tl.Log(state.Value, message));
            tl.Dispose();
        }

        public override Tensor forward(Tensor input)
        {
            // Input tensor is a 3D Tensor of shape (batch_size, observations, features).
            Tensor flattenedInput = input.flatten(start_dim: 1);
            // Flattened input Tensor has shape (batch_size, observations * features), so it is compatible with the linear layer.
            return _linear.forward(flattenedInput).type_as(input);
        }
    }
}