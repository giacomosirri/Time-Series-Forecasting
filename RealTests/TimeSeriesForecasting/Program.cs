using System.Data;
using TimeSeriesForecasting.IO;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using static TimeSeriesForecasting.DataPreprocessor;
using static TorchSharp.torch;
using System.Diagnostics;

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
        private const string LabelFile = "labels-training-set-timeseries-2009-2016.txt";
        private const string FeatureFile = "features-training-set-timeseries-2009-2016.txt";

        internal static Configuration Configuration { get; } = new Configuration();
        internal static string CurrentDirPath { get; private set; } = "";

        /*
         * This is the starting point of program execution. Initially, the program loads data of interest from file,
         * then it preprocesses it and create the necessary data structures to perform the analysis.
         * After that, there is a fork between the code that trains a neural network model and the code that predicts
         * new values using a pre-trained model. 
         * 
         * This is for two main reasons:
         * - Separation of concerns: Training a model and predicting a time series are two very different operations
         * and they are not even necessarily consequential. In fact, one can train a model, save its parameters on
         * file and then use those parameters on a completely different software. Conversely, the prediction can be
         * made using parameters read from a file.
         * - Efficiency: The prediction of new values is a much more frequent operation than the training of a model.
         * 
         * This means that this program can be run either in training mode or in prediction mode.
         * To choose which to run, specify the correct command-line argument. There are three numerics arguments allowed:
         * - 0: Train the model specified in the configurationSettings.json file and save its learnt parameters on file.
         * - 1: Predict new values using the model parameters read from file.
         * Currently, this operation merely tries to predict the actual values of the test set, which are already known. 
         * This is obviously done for debugging purposes, but it will be changed in the final implementation.
         * - 2: Perform operation 0 and 1 subsequently, so that the full model lifecycle is mimicked.
         * Any other command-line argument will cease the program.
         */
        static void Main(string[] args)
        {
            DateTime startTime = DateTime.Now;
            Console.WriteLine($"Program is running...    {startTime}\n");

            int arg = int.Parse(args[0]);
            // If the program is running in prediction mode, the directory where the model is located must be provided.
            if (arg == 1)
            {
                if (args.Length > 1)
                {
                    CurrentDirPath = args[1];
                }
                else
                {
                    Environment.Exit(1);
                }
            }
            else
            {
                /* 
                 * Create a new subdirectory of the log directory specified in the Resources file, with the name
                 * yyyy-mm-dd hh.mm.ss, where the date is the start time of the current execution of the program.
                 */
                CurrentDirPath = $"{LogDir}{(arg == 0 ? "training" : (arg == 1 ? "predictions" : "training + predictions"))} " +
                    $"{startTime.Year}-{startTime.Month:00}-{startTime.Day:00} " +
                    $"{startTime.Hour:00}.{startTime.Minute:00}.{startTime.Second:00}\\";
                Directory.CreateDirectory(CurrentDirPath);
            }

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

            if (arg != 0 && arg != 1 && arg != 2)
            {
                Environment.Exit(1);
            }
            if (arg == 0 || arg == 2)
            {
                ProgramTrain.Train(trainingInputTensor, trainingOutputTensor,
                    validationInputTensor, validationOutputTensor, testInputTensor, testOutputTensor);
            }
            if (arg == 1 || arg == 2)
            {
                ProgramPredict.Predict(testInputTensor, testOutputTensor, Configuration.InputWidth, (int)trainingInputTensor.size(2),
                    Configuration.OutputWidth, Configuration.LabelColumns.Length);
            }

            DateTime endTime = DateTime.Now;
            Console.WriteLine($"Program is completed...    {endTime}\n");

            TimeSpan elapsedTime = endTime - startTime;
            Console.WriteLine($"Elapsed time: {elapsedTime}");
        }

        internal static (bool result, string? message) DrawGraph(string id)
        {
            if (Environment.GetEnvironmentVariable("PATH")!.Contains("Python"))
            {
                string resourceName = "";
                try
                {
                    var assembly = Assembly.GetExecutingAssembly();
                    // Name of the python script.
                    resourceName = assembly.GetManifestResourceNames().Where(fileName => fileName.Contains(id)).Single();
                    using Stream stream = assembly.GetManifestResourceStream(resourceName)!;
                    using var reader = new StreamReader(stream);
                    string script = reader.ReadToEnd();
                }
                catch (Exception ex) when (ex is ArgumentException || ex is ArgumentNullException || 
                                           ex is FileLoadException || ex is FileNotFoundException)
                {
                    return (false, "Could not load python script.");
                }
                var process = new Process();
                process.StartInfo.FileName = "python";
                process.StartInfo.Arguments = $"{resourceName} {CurrentDirPath}";
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.Start();
                string output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();
                return (true, null);
            }
            else return (false, "Python is not installed on your system. Please install it and try again.");
        }
    }
}