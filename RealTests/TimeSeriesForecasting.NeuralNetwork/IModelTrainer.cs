using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public interface IModelTrainer
    {
        public void Fit(Tensor x, Tensor y, int epochs);
    }
}
