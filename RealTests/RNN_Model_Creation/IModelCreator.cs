using static TorchSharp.torch;

namespace RNN_Model_Creation
{
    internal interface IModelCreator
    {
        nn.IModule<Tensor, Tensor> CreateModule(IList<Record> records);
    }
}
