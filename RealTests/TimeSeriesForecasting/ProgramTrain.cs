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

            Console.Write("Training the model...");
            model.Fit(trainingInputTensor, trainingOutputTensor, validationInputTensor, validationOutputTensor);
            var lossLogger = new TupleLogger<int>(Program.CurrentDirPath + "loss.txt");
            Console.WriteLine(Program.Completion);

            Console.Write("Logging the progress of the loss during training on file...");
            lossLogger.Prepare(model.LossProgress.Select((value, index) => (index, value)).ToList(), 
                "Loss progress - Loss after n epochs:");
            lossLogger.Write();
            Console.WriteLine(Program.Completion);

            Console.Write("Assessing model performance on the test set...");
            IDictionary<AccuracyMetric, double> metrics = model.EvaluateAccuracy(testInputTensor, testOutputTensor);
            var metricsLogger = new TupleLogger<string>(Program.CurrentDirPath + "metrics.txt");
            metricsLogger.Prepare(metrics.Select(metric => (metric.Key.ToString(), (float)metric.Value)).ToList(), null);
            metricsLogger.Write();
            Console.WriteLine(Program.Completion);

            Console.Write("Saving the model on file...");
            model.Save(Program.CurrentDirPath);
            Console.WriteLine(Program.Completion);
        }
    }
}
