using TimeSeriesForecasting.IO;
using TimeSeriesForecasting.NeuralNetwork;
using static TorchSharp.torch;

namespace TimeSeriesForecasting
{
    internal class ProgramPredict
    {
        /*
         * This method does not actually predict new values right now. In fact, it is more of a test of
         * the quality of the model, as the expected output is known.
         */
        internal static void Predict(Tensor inputTensor, Tensor expectedOutput, int inputTimeSteps, int inputFeatures, 
            int outputTimeSteps, int outputFeatures)
        {
            Console.Write("Loading the model from file...");
            NetworkModel nn;
            if (Program.Configuration.ModelName == "RNN")
            {
                nn = new RecurrentNeuralNetwork(inputFeatures, 
                    outputTimeSteps, outputFeatures, Program.CurrentDirPath + $"RNN.model.bin");
            }
            else if (Program.Configuration.ModelName == "Linear")
            {
                nn = new SimpleNeuralNetwork(inputTimeSteps, inputFeatures,
                    outputTimeSteps, outputFeatures, Program.CurrentDirPath + $"Linear.model.bin");
            }
            else
            {
                throw new InvalidDataException("The configuration parameter that contains the name of the model is wrong.");
            }
            IModelManager model = new ModelManager(nn);
            Console.WriteLine(Program.Completion);

            Console.Write("Predicting new values...");
            Tensor output = model.Predict(inputTensor);
            Console.WriteLine(Program.Completion);

            Console.Write("Logging predicted and expected values on file...");
            var predictionLogger = new TensorLogger(Program.CurrentDirPath + "predictions.txt");
            predictionLogger.Prepare(output.reshape(output.size(0), 1), "Predictions on the test set");
            predictionLogger.Write(); 
            var expectedLogger = new TensorLogger(Program.CurrentDirPath + "expected.txt");
            expectedLogger.Prepare(expectedOutput.reshape(output.size(0), 1), "Expected values of the predicted variable");
            expectedLogger.Write();
            Console.WriteLine(Program.Completion);
        }
    }
}
