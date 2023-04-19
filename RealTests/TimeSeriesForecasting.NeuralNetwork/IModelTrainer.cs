using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public interface IModelTrainer
    {
        public bool IsTrained { get; }
        public double CurrentLoss { get; }
        public void Fit(Tensor x, Tensor y, int epochs);
    }
}
