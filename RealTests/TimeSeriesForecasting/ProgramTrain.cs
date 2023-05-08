using TimeSeriesForecasting.NeuralNetwork;
using static TorchSharp.torch;
using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
{
    internal class ProgramTrain
    {
        internal static void Train(Tensor trainingInputTensor, Tensor trainingOutputTensor,
            Tensor validationInputTensor, Tensor validationOutputTensor, Tensor testInputTensor, Tensor testOutputTensor)
        {
            NetworkModel nn;
            if (Program.Configuration.ModelName == "RNN")
            {
                nn = new RecurrentNeuralNetwork(trainingInputTensor.size(2),
                    trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            }
            else if (Program.Configuration.ModelName == "Linear")
            {
                nn = new SimpleNeuralNetwork(trainingInputTensor.size(1), trainingInputTensor.size(2),
                    trainingOutputTensor.size(1), trainingOutputTensor.size(2));
            }
            else
            {
                throw new InvalidDataException("The configuration parameter that contains the name of the model is wrong.");
            }
            IModelManager model = new ModelManager(nn);

            // Create a README inside the current subdirectory.
            var descriptionLogger = new TupleLogger<string, string>(Program.CurrentDirPath + "README.md");
            string description = $"\nThis is a {Program.Configuration.ModelName} model, trained using Stochatic Gradient Descent " +
                $"on data {(Program.Configuration.FirstValidDate.HasValue || Program.Configuration.LastValidDate.HasValue ? $"ranging {(Program.Configuration.FirstValidDate.HasValue ? $"from {Program.Configuration.FirstValidDate?.ToString("yyyy-MM-dd")}" : "")} " + $"{(Program.Configuration.LastValidDate.HasValue ? $"to {Program.Configuration.LastValidDate?.ToString("yyyy-MM-dd")}" : "")}" : "")} " +
                $"{(Program.Configuration.NormalizationMethod == "None" ? "" : $"preprocessed using {Program.Configuration.NormalizationMethod}")}. " +
                $"The model tries to predict the next {Program.Configuration.OutputWidth} value of the variable(s) " +
                $"{string.Join(", ", Program.Configuration.LabelColumns)} {Program.Configuration.Offset} hour into the future, " +
                $"using the previous {Program.Configuration.InputWidth} hour of data.";
            descriptionLogger.Prepare(("Description", description), null);
            descriptionLogger.Write();

            Console.Write("Training the model...");
            model.Fit(trainingInputTensor, trainingOutputTensor, validationInputTensor, validationOutputTensor);
            var lossLogger = new TupleLogger<int, float>(Program.CurrentDirPath + "loss-progress.txt");
            Console.WriteLine(Program.Completion);

            Console.Write("Logging the progress of the loss during training on file...");
            lossLogger.Prepare(model.LossProgress.Select((value, index) => (index, value)).ToList(),
                "Loss after n epochs:");
            lossLogger.Write();
            Console.WriteLine(Program.Completion);

            Console.Write("Drawing a graph to show loss progress...");
            (bool res, string? message) = Program.DrawGraph("loss");
            Console.WriteLine(res ? Program.Completion : message);

            Console.Write("Assessing model performance on the test set...");
            IDictionary<AccuracyMetric, double> metrics = model.EvaluateAccuracy(testInputTensor, testOutputTensor);
            var metricsLogger = new TupleLogger<string, double>(Program.CurrentDirPath + "metrics.txt");
            metricsLogger.Prepare(metrics.Select(metric => (metric.Key.ToString(), metric.Value)).ToList(), null);
            metricsLogger.Prepare(("Training time in seconds", model.LastTrainingTime.Seconds), null);
            metricsLogger.Write();
            Console.WriteLine(Program.Completion);

            Console.Write("Saving the model on file...");
            model.Save(Program.CurrentDirPath);
            Console.WriteLine(Program.Completion);
        }
    }
}
