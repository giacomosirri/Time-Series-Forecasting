using System.Data;

namespace RNN_Model_Creation
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
        }
    }
}