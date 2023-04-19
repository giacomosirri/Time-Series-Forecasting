using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace TimeSeriesForecasting.NeuralNetwork
{
    internal class ModelTraining : IModelTraining
    {
        private const int BatchSize = 32;

        private readonly Module<Tensor, Tensor> _model;

        public Optimizer Optimizer { get; set; }
        public Loss<Tensor, Tensor, Tensor> Loss { get; set; }

        public ModelTraining(Module<Tensor, Tensor> model, Optimizer optim, Loss<Tensor, Tensor, Tensor> loss) 
        { 
            _model = model;
            Optimizer = optim;
            Loss = loss;
        }

        public void Fit(Tensor x, Tensor y, int epochs)
        {
            for (int i = 0; i < epochs; i++)
            {
                var output = Loss.forward(_model.forward(x), y);
                // Clear the gradients before doing the back-propagation.
                _model.zero_grad();
                // Do back-progatation, which computes all the gradients.
                output.backward();
                Optimizer.step();
            }
        }
    }
}
