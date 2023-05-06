using System.Data;
using TimeSeriesForecasting.IO;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using static TimeSeriesForecasting.DataPreprocessor;
using static TorchSharp.torch;

namespace TimeSeriesForecasting
{
    internal class Configuration
    {
        internal string[] LabelColumns { get; private set; }
        internal string NormalizationMethod { get; private set; }
        internal DateTime? FirstValidDate { get; private set; }
        internal DateTime? LastValidDate { get; private set; }
        internal (int training, int validation, int test) DatasetSplitRatio { get; private set; }
        internal int InputWidth { get; private set; }
        internal int OutputWidth { get; private set; }
        internal int Offset { get; private set; }
        internal string ModelName { get; private set; }

        internal Configuration() 
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
        internal static readonly string LogDir = Properties.Resources.LogDirectoryPath;

        internal const string Completion = "  COMPLETE\n"; 
        internal const string PredictionFile = "predictions.txt";
        internal const string ExpectedFile = "expected-values.txt";
        private const string LabelFile = "labels-training-set-timeseries-2009-2016.txt";
        private const string FeatureFile = "features-training-set-timeseries-2009-2016.txt";

        internal static Configuration Configuration { get; } = new Configuration();

        static void Main(string[] args)
        {
            DateTime startTime = DateTime.Now;
            Console.WriteLine($"Program is running...    {startTime}\n");

            Console.Write("Loading data from .parquet file...");
            var records = new ParquetDataLoader(ValuesFile, DatesFile).GetRecords();
            Console.WriteLine(Completion);

            Console.Write("Initializing the preprocessor...");
            NormalizationMethod normalization = (NormalizationMethod)Enum.Parse(typeof(NormalizationMethod),
                                                    Configuration.NormalizationMethod.ToUpper());
            var dpp = new DataPreprocessorBuilder()
                            .Split(Configuration.DatasetSplitRatio)
                            .Normalize(normalization)
                            .AddDateRange((Configuration.FirstValidDate, Configuration.LastValidDate))
                            .Build(records);
            Console.WriteLine(Completion);

            Console.Write("Getting the processed training, validation and test sets...");
            DataTable trainingSet = dpp.GetTrainingSet();
            DataTable validationSet = dpp.GetValidationSet();
            DataTable testSet = dpp.GetTestSet();
            Console.WriteLine(Completion);
            
            Console.Write("Generating windows (batches) of data from the training, validation and test sets...");
            var singleStepWindow = new WindowGenerator(Configuration.InputWidth, 
                Configuration.OutputWidth, Configuration.Offset, Configuration.LabelColumns);
            (Tensor trainingInputTensor, Tensor trainingOutputTensor) = singleStepWindow.GenerateWindows<double>(trainingSet);
            (Tensor validationInputTensor, Tensor validationOutputTensor) = singleStepWindow.GenerateWindows<double>(validationSet);
            (Tensor testInputTensor, Tensor testOutputTensor) = singleStepWindow.GenerateWindows<double>(testSet);
            Console.WriteLine(Completion);

            /*
             * The commented out code below prints training input features and labels values on file and can be used
             * to check that the window generation algorithm is correct.
             */
            /*
            var featureLogger = new TensorLogger(LogDir + FeatureFile);
            Console.Write("Logging training set features on file...");
            featureLogger.Log(trainingInputTensor, "Training set features");
            Console.WriteLine(Completion);

            var labelLogger = new TensorLogger(LogDir + LabelFile);
            Console.Write("Logging training set labels on file...");
            labelLogger.Log(trainingOutputTensor, $"Training set values to predict: {string.Join(", ", config.LabelColumns)}");
            Console.WriteLine(Completion);
            */

            ProgramTrain.Train(trainingInputTensor, trainingOutputTensor,
                validationInputTensor, validationOutputTensor, testInputTensor, testOutputTensor);
            ProgramPredict.Predict(testInputTensor, testOutputTensor, Configuration.InputWidth, (int)trainingInputTensor.size(2), 
                Configuration.OutputWidth, Configuration.LabelColumns.Length);

            DateTime endTime = DateTime.Now;
            Console.WriteLine($"Program is completed...    {endTime}\n");

            TimeSpan elapsedTime = endTime - startTime;
            Console.WriteLine($"Elapsed time: {elapsedTime}");
        }
    }
}