using System.Data;
using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
{
    internal class Program
    {
        private const string DatasetDir = "C:\\Users\\sirri\\Desktop\\Coding\\Tirocinio\\TorchSharp\\datasets\\";
        private const string ValuesFile = "timeseries-2009-2016-no-datetime.parquet";
        private const string DatesFile  = "timeseries-2009-2016-datetime.parquet";

        static void Main(string[] args)
        {
            var records = new ParquetDataLoader(DatasetDir + ValuesFile, DatasetDir + DatesFile).GetRecords();
            var dpp = new DataPreprocessor(records, Tuple.Create(70,20,10), "Normalization");
            DataTable trainingSet = dpp.GetTrainingSet();
            DataTable validationSet = dpp.GetValidationSet();
            DataTable testSet = dpp.GetTestSet();
            Console.WriteLine(trainingSet.Rows.Count);
            var winGen = new WindowGenerator(48, 1, 6, new string[] { "T (degC)" });
            var tensors = winGen.GenerateWindows<double>(trainingSet);
        }
    }
}