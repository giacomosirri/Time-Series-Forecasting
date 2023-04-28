using Apache.Arrow;
using TimeSeriesForecasting.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class ModelTrainer : IModelTrainer
    {
        private const int BatchSize = 32;
        private const double Arrest = 10e-5;
        private const string FilePath = "C:\\Users\\sirri\\Desktop\\Coding\\Tirocinio\\TorchSharp\\RealTests\\Logs\\loss.txt";

        private readonly NetworkModel _model;
        private readonly Optimizer _optimizer;
        private readonly double _learningRate = 10e-8;
        // Type of x (features), type of y (labels) --> type of the result.
        private readonly Loss<Tensor, Tensor, Tensor> _lossFunction;
        private readonly IList<float> _losses = new List<float>();
        private readonly LossLogger _logger;

        public bool IsTrained { get; private set; } = false;
        public float CurrentLoss
        { 
            get
            {
                if (IsTrained)
                {
                    // Return the final loss of the trained model.
                    return _losses[^1];
                }
                else
                {
                    throw new InvalidOperationException("Current loss is unavailable because the model has never been trained.");
                }
            } 
        }

        public ModelTrainer(NetworkModel model)
        {
            _model = model;
            _optimizer = new SGD(_model.parameters(), _learningRate);
            _lossFunction = new MSELoss();
            _logger = new LossLogger(FilePath);
        }

        public void Fit(Tensor x, Tensor y, int epochs)
        {
            int i = 0;
            Tensor[] batched_x = x.split(BatchSize);
            Tensor[] batched_y = y.split(BatchSize);
            Tensor previousOutput = tensor(float.MaxValue);
            for (; i < epochs; i++)
            {
                Tensor output = empty(1);
                for (int j = 0; j < batched_x.Length; j++)
                {
                    // Compute the loss.
                    output = _lossFunction.forward(_model.forward(batched_x[j]), batched_y[j].flatten(start_dim: 1));
                    // Clear the gradients before doing the back-propagation.
                    _model.zero_grad();
                    // Do back-progatation, which computes all the gradients.
                    output.backward();
                    _optimizer.step();
                }
                _losses.Add(output.item<float>());
                if (Math.Abs(previousOutput.item<float>() - output.item<float>()) < Arrest)
                {
                    break;
                }
                else
                {
                    previousOutput = output;
                }
            }
            IsTrained = true;
            // Log the computed losses to file.
            _logger.Log(_losses.AsEnumerable().Select((value, index) => (index, value)).ToList(), 
                $"MSE with learning rate {_learningRate} and batch size {BatchSize}:");
            if (i < epochs)
            {
                _logger.LogComment($"The training converges after {i} epochs.");
            }
            _logger.Dispose();
        }
    }
}
