using System.ComponentModel;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public enum AccuracyMetric
    {
        [Description("Mean Squared Error")]
        MSE,
        [Description("Root Mean Squared Error")]
        RMSE,
        [Description("Mean Absolute Error")]
        MAE,
        [Description("Mean Absolute Percentage Error")]
        MAPE,
        [Description("R-Squared")]
        R2
    }

    public interface IModelManager
    {
        public bool IsTrained { get; }

        public IList<float> LossProgress { get; }

        public void Fit(Tensor trainX, Tensor trainY, Tensor validX, Tensor validY);

        public void Fit(Tensor x, Tensor y);

        public IDictionary<AccuracyMetric, double> EvaluateAccuracy(Tensor x, Tensor y);

        public Tensor Predict(Tensor x);

        public void Save(string directory);
    }
}
