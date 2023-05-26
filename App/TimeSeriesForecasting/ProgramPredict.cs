using TimeSeriesForecasting.IO;
using System.Data;
using static TorchSharp.torch;
using TimeSeriesForecasting.ANN;

namespace TimeSeriesForecasting
{
    internal class ProgramPredict
    {
        internal static void ExecutePredictCommand(string inputDirectoryAbsolutePath, 
            string outputDirectoryAbsolutePath, int? batchSize)
        {
            // Check if all necessary files and subdirectories exist and if they don't, terminate the program.
            if (!Program.IsUserDirectoryValidAsInput(inputDirectoryAbsolutePath))
            {
                Program.StopProgram(Program.DirectoryErrorMessage);
            }

            // The existence of the data files inside /data has already been checked.
            string inputDataDirectoryAbsolutePath = Path.Combine(new string[] { inputDirectoryAbsolutePath, Program.DataSubdirectory });
            string valuesFileAbsolutePath = Path.Combine(new string[] { inputDataDirectoryAbsolutePath, Program.ValuesFile });
            string datesFileAbsolutePath = Path.Combine(new string[] { inputDataDirectoryAbsolutePath, Program.DatesFile });

            Console.Write("Loading data from .parquet file...");
            var records = new ParquetDataLoader(valuesFileAbsolutePath, datesFileAbsolutePath).GetRecords();
            Console.WriteLine(Program.Completion);

            Console.Write("Initializing the preprocessor...");
            // All the 
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

            Console.Write("Loading the model from file...");
            NeuralNetwork nn = new LSTM(inputTensor.size(2), Program.GlobalConfiguration.OutputWidth, 
                Program.GlobalConfiguration.LabelColumns.LongLength);
            INeuralNetworkModel model = NeuralNetworkModel.Compile(nn);
        }
    }
}
