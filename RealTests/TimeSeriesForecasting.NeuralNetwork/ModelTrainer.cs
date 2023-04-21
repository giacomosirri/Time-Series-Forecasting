using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class ModelTrainer : IModelTrainer
    {
        private const int BatchSize = 32;

        private readonly Module<Tensor, Tensor> _model;
        private readonly Optimizer _optimizer;
        private readonly double _learningRate = 0.01;
        // Type of x (features), type of y (labels) --> type of the result.
        private readonly Loss<Tensor, Tensor, Tensor> _loss;
        private double _currentLoss = double.MaxValue;

        public bool IsTrained { get; private set; } = false;
        public double CurrentLoss
        { 
            get
            {
                if (IsTrained)
                {
                    return _currentLoss;
                }
                else
                {
                    throw new InvalidOperationException("Current loss is unavailable because the model has never been trained.");
                }
            } 
        }

        public ModelTrainer(Module<Tensor, Tensor> model) 
        { 
            _model = model;
            _optimizer = new SGD(_model.parameters(), _learningRate);
            _loss = new MSELoss();
        }

        public void Fit(Tensor x, Tensor y, int epochs)
        {
            Tensor output = zeros(1);
            Tensor[] batched_x = x.split(BatchSize);
            Tensor[] batched_y = y.split(BatchSize);
            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < batched_x.Length; j++)
                {
                    // Compute the loss.
                    output = _loss.forward(_model.forward(batched_x[j]), batched_y[j].flatten(start_dim: 1));
                    // Clear the gradients before doing the back-propagation.
                    _model.zero_grad();
                    // Do back-progatation, which computes all the gradients.
                    output.backward();
                    _optimizer.step();
                }
                _currentLoss = output.item<float>();
            }
            IsTrained = true;
        }
    }
}
