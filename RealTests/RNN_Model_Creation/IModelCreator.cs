using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting
{
    internal interface IModelCreator
    {
        IModule<Tensor, Tensor> CreateModule();
    }
}
