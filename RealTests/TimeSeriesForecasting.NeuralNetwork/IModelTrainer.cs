using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public interface IModelTrainer
    {
        public bool IsTrained { get; }
        public float CurrentLoss { get; }
        public void TuneHyperparameters(Tensor x, Tensor y);
        public void Fit(Tensor x, Tensor y);
        public IDictionary<string, double> EvaluateAccuracy(Tensor x, Tensor y, IList<string> metrics);
        public Tensor Predict(Tensor x);
    }
}
