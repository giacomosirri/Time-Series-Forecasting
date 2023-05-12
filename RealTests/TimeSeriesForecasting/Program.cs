using Newtonsoft.Json;
using System.CommandLine;

namespace TimeSeriesForecasting
{
    // The available running modes for this program.
    internal enum RunningMode
    {
        TEST,
        TRAIN,
        PREDICT
    }

    /*
     * This class encapsulates all the global configuration parameters.
     * These parameters are needed for every running mode, as the input to both the training 
     * and the prediction is a parquet table, which has one index column and one or more 
     * label columns and which must be split into windows of data usually called "time series".
     */
    public class GlobalConfiguration
    {
        // The name of the columns that contain the values to predict.
        public string[] LabelColumns { internal get; set; } = Array.Empty<string>();

        // The name of the column that contains the primary keys.
        public string IndexColumn { internal get; set; } = string.Empty;

        // The number of time steps in the input to the model.
        public int InputWidth { internal get; set; }

        // The number of time steps in the output that the model must produce.
        public int OutputWidth { internal get; set; }

        // The distance in time steps between the input and the output.
        public int Offset { internal get; set; }
    }

    internal class Program
    {
        internal static RunningMode Mode { get; set; }
        internal static bool IsLogEnabled { get; private set; }

        // To be removed.
        private static readonly string ValuesFile = Properties.Resources.NumericDatasetParquetFilePath;
        private static readonly string DatesFile = Properties.Resources.TimestampDatasetParquetFilePath;
        internal static readonly string LogDir = Properties.Resources.LogDirectoryPath;

        internal const string Completion = "  COMPLETE\n";

        internal static GlobalConfiguration GlobalConfiguration { get; private set; } = new GlobalConfiguration();
        internal static string LogDirPath { get; private set; } = "";

        /*
         * This is the starting point of program execution. Initially, the program loads data of interest from file,
         * then it preprocesses it and create the necessary data structures to perform the analysis.
         * After that, there is a fork between the code that trains a neural network model and the code that predicts
         * new values using a pre-trained model. 
         * 
         * This is for two main reasons:
         * - Separation of concerns: Training a model and predicting a time series are two very different operations
         * and they are not even necessarily consequential. In fact, one can train a model, save its parameters on
         * file and then use those parameters on a completely different software. Conversely, the prediction can be
         * made using parameters read from a file.
         * - Efficiency: The prediction of new values is a much more frequent operation than the training of a model.
         * 
         * This means that this program can be run either in training mode or in prediction mode.
         * To choose which to run, specify the correct command-line argument. There are three numerics arguments allowed:
         * - 0: Train the model specified in the configurationSettings.json file and save its learnt parameters on file.
         * - 1: Predict new values using the model parameters read from file.
         * Currently, this operation merely tries to predict the actual values of the test set, which are already known. 
         * This is obviously done for debugging purposes, but it will be changed in the final implementation.
         * - 2: Perform operation 0 and 1 subsequently, so that the full model lifecycle is mimicked.
         * Any other command-line argument will cease the program.
         */
        static void Main(string[] args)
        {
            DateTime startTime = DateTime.Now;
            Console.WriteLine($"Program is running...    {startTime}\n");

            // Create command line.
            var rootCommand = new RootCommand("App that creates, trains and runs a neural network for time series forecasting.");

            /*
             * Create command "train", add its options and arguments and set its behavior.
             */
            var trainCommand = new Command("train", "Trains the neural network, i.e. changes its parameters according to the provided data, " +
                "but does not test the new trained model.");
            var trainLogOption = new Option<bool>("--log", "if true, some output that aids the understanding of the training process is created.")
            {
                IsRequired = false,
            };
            trainLogOption.SetDefaultValue(true);
            trainLogOption.AddAlias("--l");
            trainCommand.AddOption(trainLogOption);
            var trainArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            trainCommand.AddArgument(trainArgument);
            trainCommand.SetHandler((bool log, string inputDirectoryPath) =>
            {
                // Sets the running mode and the log enabling based on the command and its log option.
                Mode = RunningMode.TRAIN;
                IsLogEnabled = log;
                try
                {                
                    // Calculate the asbolute path of the input directory.
                    string userInputDir = Path.GetFullPath(inputDirectoryPath);
                    // Then try to read the config file inside the directory. If not present, an exception is thrown.
                    CreateGlobalConfiguration(userInputDir);
                    // Finally execute the train command.
                    ProgramTrain.ExecuteTrainCommand(userInputDir);
                }
                catch (Exception ex) when (ex is IOException)
                {
                    Console.WriteLine("An unexpected error happened when reading a directory or file. The program is stopped.");
                    Environment.Exit(1);
                }
                catch (Exception)
                {
                    Console.WriteLine("The given path is not a valid directory, so the program cannot run.");
                    Environment.Exit(1);
                }
            }, trainLogOption, trainArgument);

            /*
             * Create command "test", add its options and arguments and set its behavior.
             */
            var testCommand = new Command("train", "Trains the neural network, i.e. changes its parameters according to the provided data, " +
                "but does not test the new trained model.");
            var testLogOption = new Option<bool>("--log", "if true, some output that aids the understanding of the training process is created.")
            {
                IsRequired = false
            };
            testLogOption.AddAlias("--l");
            testLogOption.SetDefaultValue(true);
            testCommand.AddOption(testLogOption); 
            var testArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            testCommand.AddArgument(testArgument);
            testCommand.SetHandler((bool log, string inputDirectoryPath) =>
            {
                // Sets the running mode and the log enabling based on the command and its log option.
                Mode = RunningMode.TEST;
                IsLogEnabled = log;
                try
                {
                    // Calculate the asbolute path of the input directory.
                    string userInputDir = Path.GetFullPath(inputDirectoryPath);
                    // Then try to read the config file inside the directory. If not present, an exception is thrown.
                    CreateGlobalConfiguration(userInputDir);
                    ProgramTrain.ExecuteTrainCommand(userInputDir);
                }
                catch (Exception ex) when (ex is IOException)
                {
                    Console.WriteLine("An unexpected error happened when reading a directory or file. The program is stopped.");
                    Environment.Exit(1);
                }
                catch (Exception)
                {
                    Console.WriteLine("The given path is not a valid directory, so the program cannot run.");
                    Environment.Exit(1);
                }
            }, trainLogOption, trainArgument);

            var predictCommand = new Command("predict", "Predicts future values using a trained neural network.");
            var predictArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            predictCommand.AddArgument(predictArgument);

            rootCommand.AddCommand(trainCommand);
            rootCommand.AddCommand(testCommand);
            rootCommand.AddCommand(predictCommand);
            rootCommand.Invoke(args);

            DateTime endTime = DateTime.Now;
            Console.WriteLine($"Program is completed...    {endTime}\n");

            TimeSpan elapsedTime = endTime - startTime;
            Console.WriteLine($"Elapsed time: {elapsedTime}");
        }

        /*
         * This method creates and returns a global configuration object from the configuration file
         * found inside the user directory. A InvalidArgumentException is raised if no such file is present.
         */
        private static GlobalConfiguration CreateGlobalConfiguration(string inputDirectoryPath)
        {
            try
            {
                // Try to find the file "global_config.json" inside the directory provided by the user.
                string globalConfigFile = Directory.GetFiles(inputDirectoryPath, "global_config.json").Single();
                using var reader = new StreamReader(globalConfigFile);
                var settings = new JsonSerializerSettings() { DateFormatString = "yyyy-MM-dd" };
                return JsonConvert.DeserializeObject<GlobalConfiguration>(reader.ReadToEnd(), settings)!;
            }
            // Distinguish between an unpredictable I/O problem and a wrong user input. 
            catch (Exception ex) when (ex is IOException)
            {
                throw new IOException("Something went wrong when reading the directory or the config file.");
            }
            catch (Exception)
            {
                throw new ArgumentException("The given path is not a directory or does not contain a file named 'global_config.json'.");
            }
        }
    }
}