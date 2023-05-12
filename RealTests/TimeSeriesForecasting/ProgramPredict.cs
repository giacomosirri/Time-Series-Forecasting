using TimeSeriesForecasting.IO;
using TimeSeriesForecasting.NeuralNetwork;
using static TorchSharp.torch;

namespace TimeSeriesForecasting
{
    internal class ProgramPredict
    {
        private static readonly string ScriptName = "plot_predicted_vs_expected.py";

        /*
         * This method does not actually predict new values right now. In fact, it is more of a test of
         * the quality of the model, as the expected output is known.
         */
        internal static void Predict(Tensor inputTensor, Tensor expectedOutput, int inputTimeSteps, int inputFeatures, 
            int outputTimeSteps, int outputFeatures)
        {
            Console.Write("Loading the model from file...");
            NetworkModel nn = new LSTM(inputFeatures, outputTimeSteps, outputFeatures, Program.LogDirPath + $"LSTM.model.bin");
            IModelManager model = new ModelManager(nn);
            Console.WriteLine(Program.Completion);

            Console.Write("Predicting new values...");
            Tensor output = model.Predict(inputTensor);
            Console.WriteLine(Program.Completion);

            Console.Write("Logging predicted and expected values on file...");
            var predictionLogger = new TensorLogger(Program.LogDirPath + "predictions.txt");
            predictionLogger.Prepare(output.reshape(output.size(0), 1), "Predictions on the test set");
            predictionLogger.Write(); 
            var expectedLogger = new TensorLogger(Program.LogDirPath + "expected.txt");
            expectedLogger.Prepare(expectedOutput.reshape(output.size(0), 1), "Expected values of the predicted variable");
            expectedLogger.Write();
            Console.WriteLine(Program.Completion);

            Console.Write("Drawing a graph to compare predicted and expected output...");
            (bool res, string? message) = Program.RunPythonScript(ScriptName);
            Console.WriteLine(res ? Program.Completion : message);
        }
    }
}
