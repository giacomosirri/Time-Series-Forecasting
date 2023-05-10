using TimeSeriesForecasting.NeuralNetwork;
using static TorchSharp.torch;
using TimeSeriesForecasting.IO;
using static TimeSeriesForecasting.DataPreprocessor;
using System.Data;
using System.Diagnostics;
using System.Reflection;

namespace TimeSeriesForecasting
{
    /*
     * This class encapsulates all the configuration parameters that pertain only to the training 
     * of the model, but that will not be of interest when predicting future unknown values.
     * Actually, the same normalization method used during training must be applied to input data also
     * when predicting. However, the normalizer is serialized when training and deserialized before
     * predicting, so that there is consistency of scale between training and prediction input data
     * without the need for the user to specify the normalizer as a global configuration parameter.
     */
    internal class TrainingConfiguration
    {
        // The normalization method to apply to the training data.
        public string NormalizationMethod { internal get; set; } = "";

        // The first timestamp in temporal order to be considered during training.
        public DateTime? FirstValidDate { internal get; set; }

        // The first timestamp in temporal order to be considered during training.
        public DateTime? LastValidDate { internal get; set; }

        // This property is used by the JsonConvert object to fetch the split ratios array from the configuration file.
        public int[] DatasetSplitRatios { private get; set; } = Array.Empty<int>();

        // The fraction of data reserved for training, validation and test respectively.
        internal (int training, int validation, int test) DatasetSplitRatio
            => (DatasetSplitRatios[0], DatasetSplitRatios[1], DatasetSplitRatios[2]) = (70, 20, 10);
    }

    internal class ProgramTrain
    {
        private static readonly string ScriptName = "plot_loss_progress.py";
        internal static string UserInputDirectory { get; private set; } = "";

        private static TrainingConfiguration _trainingConfiguration = new TrainingConfiguration();

        internal static void Train(string inputDirectoryPath)
        {
            Console.Write("Loading data from .parquet file...");
            var records = new ParquetDataLoader(ValuesFile, DatesFile).GetRecords();
            Console.WriteLine(Program.Completion);

            Console.Write("Initializing the preprocessor...");
            NormalizationMethod normalization = (NormalizationMethod)Enum.Parse(typeof(NormalizationMethod),
                                                    _trainingConfiguration.NormalizationMethod.ToUpper());
            var dpp = new DataPreprocessorBuilder()
                            .Split(_trainingConfiguration.DatasetSplitRatio)
                            .Normalize(normalization)
                            .AddDateRange((_trainingConfiguration.FirstValidDate, _trainingConfiguration.LastValidDate))
                            .Build(records);
            Console.WriteLine(Program.Completion);

            Console.Write("Getting the processed training, validation and test sets...");
            DataTable trainingSet = dpp.GetTrainingSet();
            DataTable validationSet = dpp.GetValidationSet();
            DataTable testSet = dpp.GetTestSet();
            Console.WriteLine(Program.Completion);

            Console.Write("Generating windows (batches) of data from the training, validation and test sets...");
            var singleStepWindow = new WindowGenerator(Program.GlobalConfiguration.InputWidth,
                Program.GlobalConfiguration.OutputWidth, Program.GlobalConfiguration.Offset, Program.GlobalConfiguration.LabelColumns);
            (Tensor trainingInputTensor, Tensor trainingOutputTensor) = singleStepWindow.GenerateWindows<double>(trainingSet);
            (Tensor validationInputTensor, Tensor validationOutputTensor) = singleStepWindow.GenerateWindows<double>(validationSet);
            (Tensor testInputTensor, Tensor testOutputTensor) = singleStepWindow.GenerateWindows<double>(testSet);
            Console.WriteLine(Program.Completion);

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

            // The network model used is always LSTM.
            NetworkModel nn = new LSTM(trainingInputTensor.size(2), trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            IModelManager model = new ModelManager(nn);

            // Create a README inside the current subdirectory.
            var descriptionLogger = new TupleLogger<string, string>(Program.LogDirPath + "README.md");
            string description = $"\nThis is a LSTM model, trained using Stochatic Gradient Descent " +
                $"on data {(_trainingConfiguration.FirstValidDate.HasValue || _trainingConfiguration.LastValidDate.HasValue ? $"ranging {(_trainingConfiguration.FirstValidDate.HasValue ? $"from {_trainingConfiguration.FirstValidDate?.ToString("yyyy-MM-dd")}" : "")} " + $"{(_trainingConfiguration.LastValidDate.HasValue ? $"to {_trainingConfiguration.LastValidDate?.ToString("yyyy-MM-dd")}" : "")}" : "")} " +
                $"{(_trainingConfiguration.NormalizationMethod == "None" ? "" : $"preprocessed using {_trainingConfiguration.NormalizationMethod}")}. " +
                $"The model tries to predict the next {Program.GlobalConfiguration.OutputWidth} value of the variable(s) " +
                $"{string.Join(", ", Program.GlobalConfiguration.LabelColumns)} {Program.GlobalConfiguration.Offset} hour into the future, " +
                $"using the previous {Program.GlobalConfiguration.InputWidth} hour of data.";
            descriptionLogger.Prepare(("Description", description), null);
            descriptionLogger.Write();

            Console.Write("Training the model...");
            model.Fit(trainingInputTensor, trainingOutputTensor, validationInputTensor, validationOutputTensor);
            var lossLogger = new TupleLogger<int, float>(Program.LogDirPath + "loss_progress.txt");
            Console.WriteLine(Program.Completion);

            Console.Write("Logging the progress of the loss during training on file...");
            lossLogger.Prepare(model.LossProgress.Select((value, index) => (index, value)).ToList(),
                "Loss after n epochs:");
            lossLogger.Write();
            Console.WriteLine(Program.Completion);

            Console.Write("Drawing a graph to show loss progress...");
            (bool res, string? message) = RunPythonScript(ScriptName);
            Console.WriteLine(res ? Program.Completion : message);

            // Train and test commands are differentiated by the following code.
            if (Program.Mode == RunningMode.TEST)
            {
                Console.Write("Assessing model performance on the test set...");
                IDictionary<AccuracyMetric, double> metrics = model.EvaluateAccuracy(testInputTensor, testOutputTensor);
                var metricsLogger = new TupleLogger<string, double>(Program.LogDirPath + "metrics.txt");
                metricsLogger.Prepare(metrics.Select(metric => (metric.Key.ToString(), metric.Value)).ToList(), null);
                metricsLogger.Prepare(("Training time in seconds", model.LastTrainingTime.Seconds), null);
                metricsLogger.Write();
                Console.WriteLine(Program.Completion);

                Console.Write("Saving the model on file...");
                model.Save(Program.LogDirPath);
                Console.WriteLine(Program.Completion);

                Console.Write("Predicting new values...");
                Tensor output = model.Predict(testInputTensor);
                Console.WriteLine(Program.Completion);

                Console.Write("Logging predicted and expected values on file...");
                var predictionLogger = new TensorLogger(Program.LogDirPath + "predictions.txt");
                predictionLogger.Prepare(output.reshape(output.size(0), 1), "Predictions on the test set");
                predictionLogger.Write();
                var expectedLogger = new TensorLogger(Program.LogDirPath + "expected.txt");
                expectedLogger.Prepare(testOutputTensor.reshape(output.size(0), 1), "Expected values of the predicted variable");
                expectedLogger.Write();
                Console.WriteLine(Program.Completion);

                Console.Write("Drawing a graph to compare predicted and expected output...");
                (res, message) = RunPythonScript(ScriptName);
                Console.WriteLine(res ? Program.Completion : message);
            }
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
                // The arguments are the name of the script and the path of the directory where the data is located.
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
