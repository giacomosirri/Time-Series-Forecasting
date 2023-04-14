namespace RNN_Model_Creation
{
    internal class Program
    {
        private const string DatasetDir = "C:\\Users\\sirri\\Desktop\\Coding\\Tirocinio\\TorchSharp\\dataset\\";
        private const string ValuesFile = "timeseries-2009-2016-no-datetime.parquet";
        private const string DatesFile  = "timeseries-2009-2016-datetime.parquet";

        static void Main(string[] args)
        {
            var records = new ParquetDataLoader(DatasetDir + ValuesFile, DatasetDir + DatesFile).GetRecords();
            new DataPreprocessor(records);
        }
    }
}