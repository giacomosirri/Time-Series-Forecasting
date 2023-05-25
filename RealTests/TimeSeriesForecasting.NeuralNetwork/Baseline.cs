using TorchSharp;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.ANN
{
    internal class Baseline : NeuralNetwork
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
