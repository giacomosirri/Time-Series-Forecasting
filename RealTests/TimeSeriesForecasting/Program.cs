using System.Data;
using TimeSeriesForecasting.IO;
using TimeSeriesForecasting.NeuralNetwork;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using MoreLinq;
using static TimeSeriesForecasting.DataPreprocessor;
using static TorchSharp.torch;

namespace TimeSeriesForecasting
{
    internal class Configuration
    {
        public string[] LabelColumns { get; private set; }
        public string NormalizationMethod { get; private set; }
        public DateTime? FirstValidDate { get; private set; }
        public DateTime? LastValidDate { get; private set; }
        public (int training, int validation, int test) DatasetSplitRatio { get; private set; }
        public int InputWidth { get; private set; }
        public int OutputWidth { get; private set; }
        public int Offset { get; private set; }
        public string ModelName { get; private set; }

        public Configuration() 
        {
            string resourceName = "TimeSeriesForecasting.Properties.configurationSettings.json";
            using var reader = new StreamReader(Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName)!);
            JObject jsonObject = JsonConvert.DeserializeObject<JObject>(reader.ReadToEnd())!;
            LabelColumns = jsonObject["label columns"]!.ToObject<string[]>()!;
            NormalizationMethod = jsonObject["normalization method"]?.Value<string>() ?? "None";
            FirstValidDate = jsonObject["first valid date"]?.Value<DateTime>();
            LastValidDate = jsonObject["last valid date"]?.Value<DateTime>();
            int[] splits = jsonObject["training validation test split"]?.ToObject<int[]>()!;
            if ((splits != null) && (splits[0] + splits[1] + splits[2] == 100))
            {
                DatasetSplitRatio = (splits[0], splits[1], splits[2]);
            }
            else
            {
                DatasetSplitRatio = (70, 20, 10);
            }
            InputWidth = jsonObject["input window width"]!.Value<int>();
            OutputWidth = jsonObject["output window width"]!.Value<int>();
            Offset = jsonObject["offset between input and output"]!.Value<int>();
            ModelName = jsonObject["ANN model"]!.Value<string>()!;
        }
    }

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
            var config = new Configuration();
            DateTime startTime = DateTime.Now;
            Console.WriteLine($"Program is running...    {startTime}\n");

            Console.Write("Loading data from .parquet file...");
            var records = new ParquetDataLoader(ValuesFile, DatesFile).GetRecords();
            Console.WriteLine(Completion);

            Console.Write("Initializing the preprocessor...");
            // Date range: from 01/01/2012 to 31/12/2013
            // TODO: Create a DataPreprocessor builder.
            var dpp = new DataPreprocessor(
                        records,
                        config.DatasetSplitRatio, 
                        (NormalizationMethod)Enum.Parse(typeof(NormalizationMethod), config.NormalizationMethod.ToUpper()), 
                        Tuple.Create(config.FirstValidDate, config.LastValidDate)
            );
            Console.WriteLine(Completion);

            Console.Write("Getting the processed training, validation and test sets...");
            DataTable trainingSet = dpp.GetTrainingSet();
            DataTable validationSet = dpp.GetValidationSet();
            DataTable testSet = dpp.GetTestSet();
            Console.WriteLine(Completion);

            var singleStepWindow = new WindowGenerator(config.InputWidth, config.OutputWidth, config.Offset, config.LabelColumns); 
            Console.Write("Generating windows (batches) of data from the training, validation and test sets...");
            (Tensor trainingInputTensor, Tensor trainingOutputTensor) = singleStepWindow.GenerateWindows<double>(trainingSet);
            (Tensor validationInputTensor, Tensor validationOutputTensor) = singleStepWindow.GenerateWindows<double>(validationSet);
            (Tensor testInputTensor, Tensor testOutputTensor) = singleStepWindow.GenerateWindows<double>(testSet);
            Console.WriteLine(Completion);
#if TEST
            var featureLogger = new TensorLogger(LogDir + FeatureFile);
            Console.Write("Logging training set features on file...");
            featureLogger.Log(trainingInputTensor, "Training set features");
            Console.WriteLine(Completion);

            var labelLogger = new TensorLogger(LogDir + LabelFile);
            Console.Write("Logging training set labels on file...");
            labelLogger.Log(trainingOutputTensor, $"Training set values to predict: {string.Join(", ", config.LabelColumns)}");
            Console.WriteLine(Completion);
#endif
            NetworkModel? model = null;
            if (config.ModelName == "RNN")
            {
                model = new RecurrentNeuralNetwork(trainingInputTensor.size(2), 
                    trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            }
            else if (config.ModelName == "Linear")
            {
                model = new SimpleNeuralNetwork(trainingInputTensor.size(1), trainingInputTensor.size(2),
                    trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            }
            IModelTrainer trainer = new ModelTrainer(model!, LogDir);
            Console.Write("Tuning the hyperparameters of the model on the validation set...");
            trainer.TuneHyperparameters(validationInputTensor, validationOutputTensor);
            Console.WriteLine(Completion);

            Console.Write("Training the model...");
            trainer.Fit(trainingInputTensor, trainingOutputTensor);
            Console.WriteLine(Completion);
            Console.WriteLine($"MSE: {trainer.CurrentLoss:F5}\n");

            Console.Write("Assessing model performance on the test set...");
            IDictionary<AccuracyMetric, double> metrics = trainer.EvaluateAccuracy(trainingInputTensor, trainingOutputTensor);
            Console.WriteLine(Completion);
            metrics.ForEach(metric => Console.WriteLine($"{metric.Key}: {metric.Value:F5}"));
            Console.WriteLine();

            DateTime endTime = DateTime.Now;
            Console.WriteLine($"Program is completed...    {endTime}\n");

            TimeSpan elapsedTime = endTime - startTime;
            Console.WriteLine($"Elapsed time: {elapsedTime}");
        }
    }
}