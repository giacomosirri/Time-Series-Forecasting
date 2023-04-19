using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    internal interface IModelTraining
    {
        Module Loss { get; set; }
        Module Optimizer { get; set; }
        double LearningRate { get; set; }

        public void Fit();
    }
}
