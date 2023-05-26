using TimeSeriesForecasting.IO;
using System.Data;
using static TorchSharp.torch;
using TimeSeriesForecasting.ANN;

namespace TimeSeriesForecasting
{
    internal class ProgramPredict
    {
        private const string PredictionSubdirectory = "prediction";

        internal static void ExecutePredictCommand(string inputDirectoryAbsolutePath, 
            string outputDirectoryAbsolutePath, int? batchSize)
        {
            // Check that the directory contains both a valid /data subdirectory and the seralized model file.
            string modelAbsolutePath = Path.Combine(new string[] { inputDirectoryAbsolutePath,
                    Program.ModelSubdirectory, "LSTM.model.bin"});
            if (!Program.IsUserDataSubdirectoryValid(inputDirectoryAbsolutePath) || !File.Exists(modelAbsolutePath))
            {
                Program.StopProgram(Program.DirectoryErrorMessage);
            }

            // The existence of the data files inside /data has already been checked.
            string inputDataDirectoryAbsolutePath = Path.Combine(new string[] { inputDirectoryAbsolutePath, Program.DataSubdirectory });
            string valuesFileAbsolutePath = Path.Combine(new string[] { inputDataDirectoryAbsolutePath, Program.ValuesFile });
            string datesFileAbsolutePath = Path.Combine(new string[] { inputDataDirectoryAbsolutePath, Program.DatesFile });

            // Create directory for prediction output, if it does not exist yet.
            string predictionDirectoryAbsolutePath = Path.Combine(new string[] { outputDirectoryAbsolutePath, PredictionSubdirectory });

            Console.Write("Loading data from .parquet file...");
            var records = new ParquetDataLoader(valuesFileAbsolutePath, datesFileAbsolutePath).GetRecords();
            Console.WriteLine(Program.Completion);

            Console.Write("Initializing the preprocessor...");
            // All data is considered part of the test subset when predicting. A specific method may be more appropriate.
            var dpp = new DataPreprocessorBuilder()
                            .Split((0, 0, 100))
                            .Build(records);
            Console.WriteLine(Program.Completion);

            Console.Write("Getting the processed input set...");
            DataTable inputSet = dpp.GetTestSet();
            Console.WriteLine(Program.Completion);

            Console.Write("Generating windows of data from the input set...");
            var windowGenerator = new WindowGenerator(Program.GlobalConfiguration.InputWidth,
                Program.GlobalConfiguration.OutputWidth, Program.GlobalConfiguration.Offset, Program.GlobalConfiguration.LabelColumns);
            (Tensor inputTensor, _) = windowGenerator.GenerateWindows<Tensor>(inputSet);
            Console.WriteLine(Program.Completion);

            Console.Write("Loading the model from file...");
            NeuralNetwork nn = new LSTM(inputTensor.size(2), Program.GlobalConfiguration.OutputWidth, 
                Program.GlobalConfiguration.LabelColumns.LongLength);
            INeuralNetworkModel model = NeuralNetworkModel.Compile(nn);
            Console.WriteLine(Program.Completion);

            Console.Write("Predict future values and log them on file...");
            // If the user has specified the batch size then use it, otherwise rely on the default value.
            Tensor predictedOutput = batchSize.HasValue ? model.Predict(inputTensor, batchSize.Value) : model.Predict(inputTensor);
            var logger = new TensorLogger(Path.Combine(new string[] { predictionDirectoryAbsolutePath, "predictions.txt" }));
            logger.Prepare(predictedOutput, "Values predicted by the model");
            logger.Write();
            Console.WriteLine(Program.Completion);
        }
    }
}
