using System.Data;
using static TimeSeriesForecasting.DataPreprocessor;
using TimeSeriesForecasting.IO;
using TimeSeriesForecasting.NeuralNetwork;
using static TorchSharp.torch;

namespace TimeSeriesForecasting
{
    internal class Program
    {
        private static readonly string ValuesFile = Properties.Resources.NumericDatasetParquetFilePath;
        private static readonly string DatesFile = Properties.Resources.TimestampDatasetParquetFilePath;
        private static readonly string LogDir = Properties.Resources.LogDirectoryPath;

        private const string LabelFile = "labels-training-set-timeseries-2009-2016.txt";
        private const string FeatureFile = "features-training-set-timeseries-2009-2016.txt";
        private const string Completion = "  COMPLETE\n";

        static void Main(string[] args)
        {
            DateTime startTime = DateTime.Now;
            Console.WriteLine($"Program is running...    {startTime}\n");

            Console.Write("Loading data from .parquet file...");
            var records = new ParquetDataLoader(ValuesFile, DatesFile).GetRecords();
            Console.WriteLine(Completion);

            Console.Write("Initializing the preprocessor...");
            // Date range: from 01/01/2012 to 31/12/2013
            var dpp = new DataPreprocessor(records, Tuple.Create(70,20,10), NormalizationMethod.STANDARDIZATION, 
                                            Tuple.Create<DateTime?, DateTime?>(new DateTime(2012, 1, 1), new DateTime(2013, 12, 31)));
            Console.WriteLine(Completion);

            Console.Write("Getting the processed training, validation and test sets...");
            DataTable trainingSet = dpp.GetTrainingSet();
            DataTable validationSet = dpp.GetValidationSet();
            DataTable testSet = dpp.GetTestSet();
            Console.WriteLine(Completion);

            var winGen = new WindowGenerator(6, 1, 1, new string[] { "T (degC)" }); 
            Console.Write("Generating windows (batches) of data from the training set...");
            (Tensor inputTensor, Tensor outputTensor) = winGen.GenerateWindows<double>(trainingSet);
            Console.WriteLine(Completion);
#if TEST
            var featureLogger = new TensorLogger(LogDir + FeatureFile);
            Console.Write("Logging training set features on file...");
            featureLogger.Log(inputTensor, "Training set features");
            Console.WriteLine(Completion);

            var labelLogger = new TensorLogger(LogDir + LabelFile);
            Console.Write("Logging training set labels on file...");
            labelLogger.Log(outputTensor, "Training set values to predict: Temperature (°C)");
            Console.WriteLine(Completion);
#endif
            var simpleModel = new Baseline(inputTensor.shape[1], inputTensor.shape[2], 
                                            outputTensor.shape[1], outputTensor.shape[2]);
            IModelTrainer trainer = new ModelTrainer(simpleModel);
            Console.Write("Training the baseline model...");
            trainer.Fit(inputTensor, outputTensor, epochs: 50);
            Console.WriteLine(Completion);
            Console.WriteLine($"MSE: {trainer.CurrentLoss:F4}\n");

            DateTime endTime = DateTime.Now;
            Console.WriteLine($"Program is completed...    {endTime}\n");

            TimeSpan elapsedTime = endTime - startTime;
            Console.WriteLine($"Elapsed time: {elapsedTime}");
        }
    }
}