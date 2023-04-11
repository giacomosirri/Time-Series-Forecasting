using System;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch.distributions;
using System.Runtime.Serialization;
using System.Transactions;
using System.Text;
using Python.Runtime;

namespace Mnist
{
    internal class Program
    {
        
        private static void WriteTensor(Tensor tensor) => Console.WriteLine(tensor.ToString(TorchSharp.TensorStringStyle.Default));

        private static void Main(string[] args)
        {
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", "C:\\Users\\sirri\\AppData\\Local\\Programs\\Python\\Python310\\python310.dll");
            var trainImagesPath = Path.GetFullPath("mnist.pkl.gz");
            DatasetLoader dl = new(trainImagesPath);
            dl.LoadDataset();
        }
    }
}