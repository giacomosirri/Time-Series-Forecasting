using TorchSharp;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    internal class LSTM : NetworkModel
    {
        public LSTM() : base(nameof(LSTM))
        {
        }

        public override Tensor forward(torch.Tensor input)
        {
            throw new NotImplementedException();
        }
    }
}
