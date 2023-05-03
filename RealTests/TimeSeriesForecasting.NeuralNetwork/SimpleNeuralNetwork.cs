using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class SimpleNeuralNetwork : NetworkModel
    {
        private readonly Linear _linear;

        public SimpleNeuralNetwork(long inputTimeSteps, long inputFeatures, long outputTimeSteps, long outputFeatures) : base(nameof(SimpleNeuralNetwork))
        {
            _linear = Linear(inputTimeSteps * inputFeatures, outputTimeSteps * outputFeatures);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            // Input tensor is a 3D Tensor of shape (batch_size, observations, features).
            Tensor flattenedInput = input.flatten(start_dim: 1);
            // Flattened input Tensor has shape (batch_size, observations * features), so it is compatible with the linear layer.
            return functional.relu(_linear.forward(flattenedInput).type_as(input));
        }
    }
}