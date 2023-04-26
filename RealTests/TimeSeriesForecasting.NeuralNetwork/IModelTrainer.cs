using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public interface IModelTrainer
    {
        public bool IsTrained { get; }
        public float CurrentLoss { get; }
        public void Fit(Tensor x, Tensor y, int epochs);
    }
}
