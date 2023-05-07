using TimeSeriesForecasting.NeuralNetwork;
using MoreLinq;
using static TorchSharp.torch;
using ICSharpCode.SharpZipLib;
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
            var lossLogger = new LossLogger(Program.CurrentDirPath + "loss.txt");
            lossLogger.Prepare(model.LossProgress.Select((value, index) => (index, value)).ToList(), "Loss progress");
            Console.WriteLine(Program.Completion);

            Console.Write("Assessing model performance on the test set...");
            IDictionary<AccuracyMetric, double> metrics = model.EvaluateAccuracy(testInputTensor, testOutputTensor);
            Console.WriteLine(Program.Completion);
            metrics.ForEach(metric => Console.WriteLine($"{metric.Key}: {metric.Value:F5}"));
            Console.WriteLine();

            Console.Write("Saving the model on file...");
            model.Save(Program.CurrentDirPath);
            Console.WriteLine(Program.Completion);
        }
    }
}
