using TimeSeriesForecasting.NeuralNetwork;
using static TorchSharp.torch;
using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
{
    internal class ProgramTrain
    {
        private static readonly string ScriptName = "plot_loss_progress.py";

        internal static void ParseInputDirectory(string inputDirectory)
        {

        }

        internal static void Train(Tensor trainingInputTensor, Tensor trainingOutputTensor,
            Tensor validationInputTensor, Tensor validationOutputTensor, Tensor testInputTensor, Tensor testOutputTensor)
        {
            NetworkModel nn;
            if (Program.GlobalConfiguration.ModelName == "RNN")
            {
                nn = new RecurrentNeuralNetwork(trainingInputTensor.size(2),
                    trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            }
            else if (Program.GlobalConfiguration.ModelName == "Linear")
            {
                nn = new SimpleNeuralNetwork(trainingInputTensor.size(1), trainingInputTensor.size(2),
                    trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            }
            else if (Program.GlobalConfiguration.ModelName == "LSTM")
            {
                nn = new LSTM(trainingInputTensor.size(2),
                    trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            }
            else
            {
                throw new InvalidDataException("The configuration parameter that contains the name of the model is wrong.");
            }
            IModelManager model = new ModelManager(nn);

            // Create a README inside the current subdirectory.
            var descriptionLogger = new TupleLogger<string, string>(Program.LogDirPath + "README.md");
            string description = $"\nThis is a {Program.GlobalConfiguration.ModelName} model, trained using Stochatic Gradient Descent " +
                $"on data {(Program.GlobalConfiguration.FirstValidDate.HasValue || Program.GlobalConfiguration.LastValidDate.HasValue ? $"ranging {(Program.GlobalConfiguration.FirstValidDate.HasValue ? $"from {Program.GlobalConfiguration.FirstValidDate?.ToString("yyyy-MM-dd")}" : "")} " + $"{(Program.GlobalConfiguration.LastValidDate.HasValue ? $"to {Program.GlobalConfiguration.LastValidDate?.ToString("yyyy-MM-dd")}" : "")}" : "")} " +
                $"{(Program.GlobalConfiguration.NormalizationMethod == "None" ? "" : $"preprocessed using {Program.GlobalConfiguration.NormalizationMethod}")}. " +
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
            (bool res, string? message) = Program.RunPythonScript(ScriptName);
            Console.WriteLine(res ? Program.Completion : message);

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
        }
    }
}
