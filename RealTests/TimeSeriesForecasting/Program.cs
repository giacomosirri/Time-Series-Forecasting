﻿using System.Data;
using TimeSeriesForecasting.IO;
using System.Reflection;
using Newtonsoft.Json;
using static TimeSeriesForecasting.DataPreprocessor;
using static TorchSharp.torch;
using System.Diagnostics;
using System.CommandLine;

namespace TimeSeriesForecasting
{
    internal enum Mode
    {
        TEST,
        TRAIN,
        PREDICT
    }

    public class Configuration
    {
        public string[] LabelColumns { get; set; } = Array.Empty<string>();
        public string NormalizationMethod { get; set; } = "";
        public DateTime? FirstValidDate { get; set; }
        public DateTime? LastValidDate { get; set; }
        /*
         * This property is used to set the values from the json configuration file,
         * so it basically readonly from the Program's point of view.
         */
        public int[] DatasetSplitRatio { private get; set; } = Array.Empty<int>();
        /*
         * This property is used by the Program to access the splits.
         */
        public (int training, int validation, int test) TrainingValidationTestSplits
        { 
            get => (DatasetSplitRatio[0], DatasetSplitRatio[1], DatasetSplitRatio[2]);
        }
        public int InputWidth { get; set; }
        public int OutputWidth { get; set; }
        public int Offset { get; set; }
        public string ModelName { get; set; } = "";
    }

    internal class Program
    {
        private static readonly (string train, string predict, string all) _availableInputModes = ("train", "predict", "all");
        private static Mode Mode { get; set; }

        private static readonly string ValuesFile = Properties.Resources.NumericDatasetParquetFilePath;
        private static readonly string DatesFile = Properties.Resources.TimestampDatasetParquetFilePath;
        internal static readonly string LogDir = Properties.Resources.LogDirectoryPath;

        internal const string Completion = "  COMPLETE\n";

        internal static Configuration Configuration { get; private set; } = new Configuration();
        internal static string LogDirPath { get; private set; } = "";
        internal static string UserInputDir { get; private set; } = "";

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

            var rootCommand = new RootCommand("App that creates, trains and runs a neural network for time series forecasting.");

            var trainCommand = new Command("train", "Trains the neural network, i.e. changes its parameters according to the provided data, " +
                "but does not test the new trained model.");
            var trainLogOption = new Option<bool>("--log", "if true, some output that aids the understanding of the training process is created.")
            {
                IsRequired = false,
            };
            trainLogOption.SetDefaultValue(true);
            trainLogOption.AddAlias("--l");
            trainCommand.AddOption(trainLogOption);
            var trainArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            trainCommand.AddArgument(trainArgument);

            var testCommand = new Command("train", "Trains the neural network, i.e. changes its parameters according to the provided data, " +
                "but does not test the new trained model.");
            var testLogOption = new Option<bool>("--log", "if true, some output that aids the understanding of the training process is created.")
            {
                IsRequired = false
            };
            testLogOption.AddAlias("--l");
            testLogOption.SetDefaultValue(true);
            testCommand.AddOption(testLogOption); 
            var testArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            testCommand.AddArgument(testArgument);

            var predictCommand = new Command("predict", "Predicts future values using a trained neural network.");
            var predictArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            predictCommand.AddArgument(predictArgument);

            rootCommand.AddCommand(trainCommand);
            rootCommand.AddCommand(testCommand);
            rootCommand.AddCommand(predictCommand);

            trainCommand.SetHandler((string inputDirPath) =>
            {
                try
                {
                    UserInputDir = Path.GetFullPath(inputDirPath);
                    string configFile = Directory.GetFiles(UserInputDir, "config.json").Single();
                    using var reader = new StreamReader(Path.Combine(new string[] { UserInputDir, configFile }));
                    var settings = new JsonSerializerSettings() { DateFormatString = "yyyy-MM-dd" };
                    Configuration = JsonConvert.DeserializeObject<Configuration>(reader.ReadToEnd(), settings)!;
                }
                catch (Exception) { }
            }, trainArgument);

            rootCommand.Invoke(args);

            Console.Write("Loading data from .parquet file...");
            var records = new ParquetDataLoader(ValuesFile, DatesFile).GetRecords();
            Console.WriteLine(Completion);

            Console.Write("Initializing the preprocessor...");
            NormalizationMethod normalization = (NormalizationMethod)Enum.Parse(typeof(NormalizationMethod),
                                                    Configuration.NormalizationMethod.ToUpper());
            var dpp = new DataPreprocessorBuilder()
                            .Split(Configuration.TrainingValidationTestSplits)
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
            var featureLogger = new TensorLogger(LogDir + "training-features.txt");
            Console.Write("Logging training set features on file...");
            featureLogger.Log(trainingInputTensor, "Training set features");
            Console.WriteLine(Completion);

            var labelLogger = new TensorLogger(LogDir + "training-labels.txt");
            Console.Write("Logging training set labels on file...");
            labelLogger.Log(trainingOutputTensor, $"Training set values to predict: {string.Join(", ", config.LabelColumns)}");
            Console.WriteLine(Completion);
            */

            if (Mode == Mode.TRAIN || Mode == Mode.TEST)
            {
                ProgramTrain.Train(trainingInputTensor, trainingOutputTensor,
                    validationInputTensor, validationOutputTensor, testInputTensor, testOutputTensor);
            }
            if (Mode == Mode.PREDICT)
            {
                ProgramPredict.Predict(testInputTensor, testOutputTensor, Configuration.InputWidth, (int)trainingInputTensor.size(2),
                    Configuration.OutputWidth, Configuration.LabelColumns.Length);
            }

            DateTime endTime = DateTime.Now;
            Console.WriteLine($"Program is completed...    {endTime}\n");

            TimeSpan elapsedTime = endTime - startTime;
            Console.WriteLine($"Elapsed time: {elapsedTime}");
        }

        internal static (bool result, string? message) RunPythonScript(string scriptName)
        {
            if ((Environment.GetEnvironmentVariable("PATH") != null) && 
                Environment.GetEnvironmentVariable("PATH")!.Contains("Python"))
            {
                string scriptPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, scriptName);
                // Create and execute a new python process to draw the graph.
                var process = new Process();
                process.StartInfo.FileName = "python";
                process.StartInfo.Arguments = $"{scriptPath} {LogDirPath}";
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                bool res = process.Start();
                if (!res)
                {
                    return (false, "Python script could not be executed.");
                }
                process.WaitForExit();
                return (true, null);
            }
            else return (false, "  Python is not installed on your system. Please install it and try again.\n");
        }
    }
}