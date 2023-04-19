using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class Baseline : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> layer;

        public Baseline(int inputObservations, int inputFeatures, int outputObservations, int outputFeatures) 
            : base(nameof(Baseline))
        {
            RegisterComponents();
            layer = Linear(inputObservations * inputFeatures, outputObservations * outputFeatures);
        }

        public override Tensor forward(Tensor input) => layer.forward(input);
    }
}