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
            int outputTimeSteps, int outputFeatures, string modelDir)
        {
            Console.Write("Loading the model from file...");
            NetworkModel nn;
            if (Program.Configuration.ModelName == "RNN")
            {
                nn = new RecurrentNeuralNetwork(inputFeatures, 
                    outputTimeSteps, outputFeatures, modelDir + $"RNN.model.bin");
            }
            else if (Program.Configuration.ModelName == "Linear")
            {
                nn = new SimpleNeuralNetwork(inputTimeSteps, inputFeatures,
                    outputTimeSteps, outputFeatures, modelDir + $"Linear.model.bin");
            }
            else
            {
                throw new InvalidDataException("The configuration parameter that contains the name of the model is wrong.");
            }
            IModelManager model = new ModelManager(nn);
            Console.WriteLine(Program.Completion);

            Console.Write("Predicting new values...");
            Tensor y = model.Predict(inputTensor);
            Console.WriteLine(Program.Completion);

            Console.Write("Logging predicted and expected values on file...");
            Tensor output = y;
            var predictionLogger = new TensorLogger(Program.CurrentDirPath + Program.PredictionFile);
            predictionLogger.Prepare(output.reshape(y.size(0), 1), "predictions on the test set");
            predictionLogger.Write(); 
            Tensor expected = expectedOutput;
            var expectedLogger = new TensorLogger(Program.CurrentDirPath + Program.ExpectedFile);
            expectedLogger.Prepare(expected.reshape(y.size(0), 1), "expected values");
            expectedLogger.Write();
            Console.WriteLine(Program.Completion);
        }
    }
}
