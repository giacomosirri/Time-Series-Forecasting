﻿using TimeSeriesForecasting.ANN;
using static TorchSharp.torch;
using TimeSeriesForecasting.IO;
using static TimeSeriesForecasting.DataPreprocessor;
using System.Data;
using MoreLinq;

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
        public string NormalizationMethod { internal get; set; } = "None";

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
        private const string TrainingConfigFile = "training_config.json";
        private const string ValueRangesFile = "value_ranges.json";
        private const string TrainingSubdirectory = "training";

        // The test code is executed only if directly specified in the method call.
        private static bool _test = false;

        internal static void ExecuteTrainCommand(string inputDirectoryAbsolutePath, 
            string outputDirectoryAbsolutePath, TrainingHyperparameters hyperparameters, bool test)
        {
            _test = test;
            ExecuteTrainCommand(inputDirectoryAbsolutePath, outputDirectoryAbsolutePath, hyperparameters);
        }

        internal static void ExecuteTrainCommand(string inputDirectoryAbsolutePath, 
            string outputDirectoryAbsolutePath, TrainingHyperparameters hyperparameters)
        {
            // Check if all necessary files and subdirectories exist and if they don't, terminate the program.
            if (!Program.IsUserDataSubdirectoryValid(inputDirectoryAbsolutePath))
            {
                Program.StopProgram(Program.DirectoryErrorMessage);
            }

            // The existence of the data files inside /data has already been checked.
            string inputDataDirectoryAbsolutePath = Path.Combine(new string[] { inputDirectoryAbsolutePath, Program.DataSubdirectory });
            string valuesFileAbsolutePath = Path.Combine(new string[] { inputDataDirectoryAbsolutePath, Program.ValuesFile });
            string datesFileAbsolutePath = Path.Combine(new string[] { inputDataDirectoryAbsolutePath, Program.DatesFile });

            // The value ranges file is not strictly necessary for the program to work.
            string valueRangesFileAbsolutePath = Path.Combine(new string[] { inputDataDirectoryAbsolutePath, ValueRangesFile });
            if (File.Exists(valueRangesFileAbsolutePath))
            {
                var res = Record.ReadValueRangesFromJsonFile(valueRangesFileAbsolutePath);
                if (!res)
                {
                    Program.StopProgram(Program.DirectoryErrorMessage);
                }
            }

            // The training config file may not exist since all training configuration parameters have a default value.
            string inputTrainingDirectoryAbsolutePath = Path.Combine(new string[] { inputDirectoryAbsolutePath, TrainingSubdirectory });
            Directory.CreateDirectory(inputTrainingDirectoryAbsolutePath);
            string configFileAbsolutePath = Path.Combine(new string[] { inputTrainingDirectoryAbsolutePath, TrainingConfigFile });
            TrainingConfiguration trainingConfiguration;
            if (File.Exists(configFileAbsolutePath))
            {
                // Load the training configuration from the config file.
                trainingConfiguration = Program.GetConfiguration<TrainingConfiguration>(configFileAbsolutePath);
            }
            else
            {
                // Run with the default training configuration.
                trainingConfiguration = new TrainingConfiguration();
            }

            // The model subdirectory may not exist yet. The code below creates this directory only if not already present.
            string modelDirectoryAbsolutePath = Path.Combine(new string[] { outputDirectoryAbsolutePath, Program.ModelSubdirectory });
            Directory.CreateDirectory(modelDirectoryAbsolutePath);
            // Same for the training output subdirectory.
            string outputTrainingDirectoryAbsolutePath = Path.Combine(new string[] { outputDirectoryAbsolutePath, TrainingSubdirectory });
            Directory.CreateDirectory(outputTrainingDirectoryAbsolutePath);

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

            Console.Write("Generating windows of data from the training, validation and test sets...");
            var windowGenerator = new WindowGenerator(Program.GlobalConfiguration.InputWidth,
                Program.GlobalConfiguration.OutputWidth, Program.GlobalConfiguration.Offset, Program.GlobalConfiguration.LabelColumns);
            (Tensor trainingInputTensor, Tensor trainingOutputTensor) = windowGenerator.GenerateWindows<double>(trainingSet);
            (Tensor validationInputTensor, Tensor validationOutputTensor) = windowGenerator.GenerateWindows<double>(validationSet);
            (Tensor testInputTensor, Tensor testOutputTensor) = windowGenerator.GenerateWindows<double>(testSet);
            Console.WriteLine(Program.Completion);

            // The network model used is always LSTM.
            Console.Write("Creating and training the model...");
            NeuralNetwork nn = new LSTM(trainingInputTensor.size(2), trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            INeuralNetworkModel model = NeuralNetworkModel.Compile(nn);
            model.Fit(trainingInputTensor, trainingOutputTensor, validationInputTensor, validationOutputTensor, 
                hyperparameters.Epochs, hyperparameters.BatchSize, hyperparameters.LearningRate);
            Console.WriteLine(Program.Completion);

            Console.Write("Saving the model's parameters and summary on file...");
            model.Save(modelDirectoryAbsolutePath);
            var modelLogger = new StringLogger(Path.Combine(new string[] { modelDirectoryAbsolutePath, "model_summary.txt" }));
            modelLogger.Prepare(model.Summarize(), "A summary of the model layers and learnable parameters");
            modelLogger.Write();
            Console.WriteLine(Program.Completion);

            if (Program.IsLogEnabled)
            {
                // Log training features and labels on file. Allows checking that the window generation algorithm is correct.
                Console.Write("Logging training set features and labels on file...");
                var featureLogger = new TensorLogger(Path.Combine(new string[] { inputTrainingDirectoryAbsolutePath, "training-features.txt" }));
                featureLogger.Prepare(trainingInputTensor, "Training set features");
                featureLogger.Write();
                var labelLogger = new TensorLogger(Path.Combine(new string[] { inputTrainingDirectoryAbsolutePath, "training-labels.txt" }));
                labelLogger.Prepare(trainingOutputTensor,
                    $"Training set values to predict: {string.Join(", ", Program.GlobalConfiguration.LabelColumns)}");
                labelLogger.Write();
                Console.WriteLine(Program.Completion);

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
                Console.Write("Predicting new values...");
                // If the user has specified the batch size then use it, otherwise rely on the default value.
                Tensor predictedOutput = hyperparameters.BatchSize.HasValue ? 
                    model.Predict(testInputTensor, hyperparameters.BatchSize.Value) : model.Predict(testInputTensor);
                Console.WriteLine(Program.Completion);

                Console.Write("Logging predicted and expected values on file...");
                var predictionLogger = new TensorLogger(Path.Combine(new string[] { outputTrainingDirectoryAbsolutePath, "predictions.txt" }));
                predictionLogger.Prepare(predictedOutput, "Predictions on the test set");
                predictionLogger.Write();
                var expectedLogger = new TensorLogger(Path.Combine(new string[] { outputTrainingDirectoryAbsolutePath, "expected.txt" }));
                expectedLogger.Prepare(testOutputTensor, "Expected values of the predicted variable");
                expectedLogger.Write();
                Console.WriteLine(Program.Completion);

                Console.Write("Assessing model performance on the test set...");
                IDictionary<AccuracyMetric, IList<double>> metrics = model.Evaluate(predictedOutput, testOutputTensor);
                var metricsLogger = new TupleLogger<string, double>(Path.Combine(new string[] { outputTrainingDirectoryAbsolutePath, "metrics.txt" }));
                // Log the value of each metric for each feature.
                metrics.ForEach(metric =>
                {
                    metric.Value.ForEach((double feature, int index) =>
                    {
                        metricsLogger.Prepare((metric.Key.ToString() + $" feature {index}", feature), null);
                    });
                });
                metricsLogger.Write();
                Console.WriteLine(Program.Completion);
            }
        }
    }
}
