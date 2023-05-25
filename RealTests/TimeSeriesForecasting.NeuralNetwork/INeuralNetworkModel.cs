using System.ComponentModel;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.ANN
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

    public interface INeuralNetworkModel
    {
        public bool IsTrained { get; }

        public IList<float> LossProgress { get; }

        public void Fit(Tensor trainX, Tensor trainY, Tensor validX, Tensor validY, int epochs, int batchSize, double learningRate);

        public void Fit(Tensor x, Tensor y, int epochs, int batchSize, double learningRate);

        public IDictionary<AccuracyMetric, IList<double>> Evaluate(Tensor predictedOutput, Tensor expectedOutput);

        public IDictionary<AccuracyMetric, IList<double>> PredictAndEvaluate(Tensor x, Tensor expectedOutput);

        public Tensor Predict(Tensor x);

        public Tensor Predict(Tensor x, int batchSize);

        public string Summarize();

        public void Save(string directory);
    }
}
