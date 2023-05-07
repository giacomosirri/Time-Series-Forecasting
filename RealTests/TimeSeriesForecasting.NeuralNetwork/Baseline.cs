using TorchSharp;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    internal class Baseline : NetworkModel
    {
        public Baseline() : base(nameof(Baseline))
        {
        }

        public override Tensor forward(torch.Tensor input)
        {
            throw new NotImplementedException();
        }
    }
}
