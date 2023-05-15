using TimeSeriesForecasting.NeuralNetwork;
using static TorchSharp.torch;
using TimeSeriesForecasting.IO;
using static TimeSeriesForecasting.DataPreprocessor;
using System.Data;

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
        private int[] _splits = new int[] { 70, 20, 10 };

        // The normalization method to apply to the training data.
        public string NormalizationMethod { internal get; set; } = "";

        // The first timestamp in temporal order to be considered during training.
        public DateTime? FirstValidDate { internal get; set; }

        // The first timestamp in temporal order to be considered during training.
        public DateTime? LastValidDate { internal get; set; }

        // This property is used by the JsonConvert object to fetch the split ratios array from the configuration file.
        public int[] DatasetSplitRatios 
        { 
            private get => _splits;
            set
            {
                // Update the value of _splits only if the sum of the splits is 100.
                if (value[0] + value[1] + value[2] == 100)
                {
                    _splits = value;
                }
            }
        }

        // The fraction of data reserved for training, validation and test respectively.
        internal (int training, int validation, int test) DatasetSplitRatio
            => (DatasetSplitRatios[0], DatasetSplitRatios[1], DatasetSplitRatios[2]);
    }


    internal class ProgramTrain
    {
        private const string ValuesFile = "data-values.parquet";
        private const string DatesFile = "data-dates.parquet";
        private const string TrainingSubdirectory = "training";
        private const string ModelSubdirectory = "model";
        private const string TrainingConfigFile = "training_config.json";

        // The test code is executed only if directly specified in the method call.
        private static bool _test = false;

        internal static void ExecuteTrainCommand(string inputDirectoryAbsolutePath, string outputDirectoryAbsolutePath, bool test)
        {
            _test = test;
            ExecuteTrainCommand(inputDirectoryAbsolutePath, outputDirectoryAbsolutePath);
        }

        internal static void ExecuteTrainCommand(string inputDirectoryAbsolutePath, string outputDirectoryAbsolutePath)
        {
            // Check if all necessary files and subdirectories exist and if they don't, terminate the program.
            if (!IsInputDirectoryValid(inputDirectoryAbsolutePath))
            {
                Console.WriteLine(Program.DirectoryErrorMessage);
                Environment.Exit(1);
            }

            // The existence of these files has already been checked.
            string valuesFileAbsolutePath = Path.Combine(new string[] { inputDirectoryAbsolutePath, ValuesFile });
            string datesFileAbsolutePath = Path.Combine(new string[] { inputDirectoryAbsolutePath, DatesFile });
            string inputTrainingDirectoryAbsolutePath = Path.Combine(new string[] { inputDirectoryAbsolutePath, TrainingSubdirectory });
            string configFileAbsolutePath = Path.Combine(new string[] { inputTrainingDirectoryAbsolutePath, TrainingConfigFile });

            // The model subdirectory may not exist yet. The code below creates this directory only if not already present.
            string modelDirectoryAbsolutePath = Path.Combine(new string[] { outputDirectoryAbsolutePath, ModelSubdirectory });
            Directory.CreateDirectory(modelDirectoryAbsolutePath);
            // Same for the training output subdirectory.
            string outputTrainingDirectoryAbsolutePath = Path.Combine(new string[] { outputDirectoryAbsolutePath, TrainingSubdirectory });
            Directory.CreateDirectory(outputTrainingDirectoryAbsolutePath);

            // Load the training configuration from the config file.
            TrainingConfiguration trainingConfiguration = Program.GetConfiguration<TrainingConfiguration>(configFileAbsolutePath);

            Console.Write("Loading data from .parquet file...");
            var records = new ParquetDataLoader(valuesFileAbsolutePath, datesFileAbsolutePath).GetRecords();
            Console.WriteLine(Program.Completion);

            Console.Write("Initializing the preprocessor...");
            NormalizationMethod normalization = (NormalizationMethod)Enum.Parse(typeof(NormalizationMethod),
                                                    trainingConfiguration.NormalizationMethod.ToUpper());
            var dpp = new DataPreprocessorBuilder()
                            .Split(trainingConfiguration.DatasetSplitRatio)
                            .Normalize(normalization)
                            .AddDateRange((trainingConfiguration.FirstValidDate, trainingConfiguration.LastValidDate))
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

            Console.Write("Training the model...");
            model.Fit(trainingInputTensor, trainingOutputTensor, validationInputTensor, validationOutputTensor);
            Console.WriteLine(Program.Completion);

            Console.Write("Saving the model on file...");
            model.Save(modelDirectoryAbsolutePath);
            Console.WriteLine(Program.Completion);

            if (Program.IsLogEnabled)
            {
                // Create a README inside the current subdirectory.
                var descriptionLogger = new TupleLogger<string, string>(Path.Combine(new string[] { outputTrainingDirectoryAbsolutePath, "README.md" }));
                string description = $"\nThis is a LSTM model trained " +
                    $"on data {(trainingConfiguration.FirstValidDate.HasValue || trainingConfiguration.LastValidDate.HasValue ? $"ranging {(trainingConfiguration.FirstValidDate.HasValue ? $"from {trainingConfiguration.FirstValidDate?.ToString("yyyy-MM-dd")}" : "")} " + $"{(trainingConfiguration.LastValidDate.HasValue ? $"to {trainingConfiguration.LastValidDate?.ToString("yyyy-MM-dd")}" : "")}" : "")} " +
                    $"{(trainingConfiguration.NormalizationMethod == "None" ? "" : $"preprocessed using {trainingConfiguration.NormalizationMethod}")}. " +
                    $"The model tries to predict the next {Program.GlobalConfiguration.OutputWidth} value of the variable(s) " +
                    $"{string.Join(", ", Program.GlobalConfiguration.LabelColumns)} {Program.GlobalConfiguration.Offset} hour into the future, " +
                    $"using the previous {Program.GlobalConfiguration.InputWidth} hour of data.";
                descriptionLogger.Prepare(("Description", description), null);
                descriptionLogger.Write();

                Console.Write("Logging the progress of the loss during training on file...");
                var lossLogger = new TupleLogger<int, float>(Path.Combine(new string[] { outputTrainingDirectoryAbsolutePath, "loss_progress.txt" }));
                lossLogger.Prepare(model.LossProgress.Select((value, index) => (index, value)).ToList(), "Loss after n epochs:");
                lossLogger.Write();
                Console.WriteLine(Program.Completion);

                Console.Write("Drawing a graph to show loss progress...");
                (bool res, string? message) = Program.RunPythonScript("plot_loss_progress.py", outputTrainingDirectoryAbsolutePath);
                Console.WriteLine(res ? Program.Completion : message);
            }

            // Train and test commands are differentiated by the following code.
            if (_test)
            {
                Console.Write("Assessing model performance on the test set...");
                IDictionary<AccuracyMetric, double> metrics = model.EvaluateAccuracy(testInputTensor, testOutputTensor);
                var metricsLogger = new TupleLogger<string, double>(Path.Combine(new string[] {outputTrainingDirectoryAbsolutePath, "metrics.txt"}));
                metricsLogger.Prepare(metrics.Select(metric => (metric.Key.ToString(), metric.Value)).ToList(), null);
                metricsLogger.Prepare(("Training time in seconds", model.LastTrainingTime.Seconds), null);
                metricsLogger.Write();
                Console.WriteLine(Program.Completion);

                Console.Write("Predicting new values...");
                Tensor output = model.Predict(testInputTensor);
                Console.WriteLine(Program.Completion);

                Console.Write("Logging predicted and expected values on file...");
                var predictionLogger = new TensorLogger(Path.Combine(new string[] { outputTrainingDirectoryAbsolutePath, "predictions.txt" }));
                predictionLogger.Prepare(output.reshape(output.size(0), 1), "Predictions on the test set");
                predictionLogger.Write();
                var expectedLogger = new TensorLogger(Path.Combine(new string[] { outputTrainingDirectoryAbsolutePath, "expected.txt" }));
                expectedLogger.Prepare(testOutputTensor.reshape(output.size(0), 1), "Expected values of the predicted variable");
                expectedLogger.Write();
                Console.WriteLine(Program.Completion);

                Console.Write("Drawing a graph to compare predicted and expected output...");
                (bool res, string? message) = Program.RunPythonScript("plot_predicted_vs_expected.py", outputTrainingDirectoryAbsolutePath);
                Console.WriteLine(res ? Program.Completion : message);
            }
        }

        /*
         * This method checks if the given directory path represents a valid directory for the scope of this class.
         */
        private static bool IsInputDirectoryValid(string directoryPath)
        {
            try
            {
                // A file called data-values.parquet must exist.
                _ = Directory.GetFiles(directoryPath, ValuesFile).Single();
                // A file called data-datetimes.parquet must exist.
                _ = Directory.GetFiles(directoryPath, DatesFile).Single();
                // A /training subdirectory must exist.
                string trainingDir = Directory.GetDirectories(directoryPath, TrainingSubdirectory).Single();
                // A file called training_config.json must exist inside the training subdirectory.
                _ = Directory.GetFiles(trainingDir, TrainingConfigFile).Single();
                // If all the necessary files and subdirectories have been found, then return true.
                return true;
            }
            catch (Exception ex) when (ex is ArgumentNullException || ex is InvalidOperationException) 
            {
                // If any necessary file or subdirectory is not found, then return false.
                return false;
            }
        }
    }
}
