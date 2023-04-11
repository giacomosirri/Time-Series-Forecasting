using System;
using System.IO;
using System.IO.Compression;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.ConstrainedExecution;
using static TorchSharp.torch;
using Python.Runtime;
using System.Globalization;

namespace Mnist
{
    internal class DatasetLoader
    {
        private readonly string _filePath;
        private readonly FileStream _decompressedFile;

        public DatasetLoader(string path)
        {
            PythonEngine.Initialize();
            _filePath = path;
            var newFilePath = Path.Combine(Environment.CurrentDirectory, "mnist.pkl");
            if (!File.Exists(newFilePath))
            {
                _decompressedFile = File.Create(newFilePath);
                using FileStream inputFile = File.OpenRead(_filePath);
                using GZipStream decompressionStream = new(inputFile, CompressionMode.Decompress);
                decompressionStream.CopyTo(_decompressedFile);
            }
            else
            {
                _decompressedFile = File.OpenRead(newFilePath);
            }
        }

        public void LoadDataset()
        {
            using (Py.GIL())
            {
                dynamic pickle = Py.Import("pickle");
                using BinaryReader binaryReader = new(_decompressedFile);
                byte[] byteStream = Encoding.Convert(Encoding.Latin1, Encoding.ASCII, 
                    binaryReader.ReadBytes((int)_decompressedFile.Length));
                dynamic obj = pickle.loads(byteStream);
                Console.WriteLine(obj.ToString(TorchSharp.TensorStringStyle.Default));
            }
        }
    }
}
