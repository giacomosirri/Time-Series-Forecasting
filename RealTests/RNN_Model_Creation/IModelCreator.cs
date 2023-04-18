using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RNN_Model_Creation
{
    internal interface IModelCreator
    {
        IModule<Tensor, Tensor> CreateModule(IList<Record> records);
    }
}
