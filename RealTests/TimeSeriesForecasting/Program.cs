using System.Data;
using TimeSeriesForecasting.IO;
using static TorchSharp.torch;

namespace TimeSeriesForecasting
{
    internal class Program
    {
        private const string DatasetDir = "C:\\Users\\sirri\\Desktop\\Coding\\Tirocinio\\TorchSharp\\datasets\\";
        private const string ValuesFile = "timeseries-2009-2016-no-datetime.parquet";
        private const string DatesFile  = "timeseries-2009-2016-datetime.parquet";
        private const string LogDir = "C:\\Users\\sirri\\Desktop\\Coding\\Tirocinio\\TorchSharp\\RealTests\\Logs\\";
        private const string LabelFile = "labels-tensor.txt";
        private const string FeatureFile = "features-tensor.txt";

        static void Main(string[] args)
        {
            var records = new ParquetDataLoader(DatasetDir + ValuesFile, DatasetDir + DatesFile).GetRecords();
            var dpp = new DataPreprocessor(records, Tuple.Create(70,20,10), "Normalization");
            DataTable trainingSet = dpp.GetTrainingSet();
            DataTable validationSet = dpp.GetValidationSet();
            DataTable testSet = dpp.GetTestSet();
            var winGen = new WindowGenerator(48, 1, 6, new string[] { "T (degC)" });
            Tuple<Tensor, Tensor> tensors = winGen.GenerateWindows<double>(trainingSet);
            var inputTensor = tensors.Item1;
            var outputTensor = tensors.Item2;
            var logger = new TensorLogger(LogDir + FeatureFile);
            logger.Log(inputTensor, "Features");
        }
    }
}