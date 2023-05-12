using Newtonsoft.Json;
using System.CommandLine;
using System.Diagnostics;
using System.Reflection;

namespace TimeSeriesForecasting
{
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
        internal const string Completion = "  COMPLETE\n";
        private const string IOErrorMessage = "An error occurred when reading a directory or file. The program has been stopped.";
        private const string DirectoryErrorMessage = "The directory you provided is not valid. The program has been stopped.";

        internal static bool IsLogEnabled { get; private set; }

        /*
         * 
         */
        internal static GlobalConfiguration GlobalConfiguration { get; private set; } = new GlobalConfiguration();

        static void Main(string[] args)
        {
            DateTime startTime = DateTime.Now;
            Console.WriteLine($"Program is running...    {startTime}\n");

            // Create command line.
            var rootCommand = new RootCommand("App that creates, trains and runs a neural network for time series forecasting.");

            // Create command "train".
            var trainCommand = new Command("train", "Trains the neural network, i.e. changes its parameters according to the provided data, " +
                                                    "but does not test the new trained model on the test set.");
            var trainLogOption = new Option<bool>(
                name: "--log",
                description: "if true, outputs some data and charts useful to understand the result of the training process."
            )
            {
                IsRequired = false
            };
            trainLogOption.SetDefaultValue(true);
            trainLogOption.AddAlias("--l");
            trainCommand.AddOption(trainLogOption);
            var trainArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            trainCommand.AddArgument(trainArgument);
            trainCommand.SetHandler((bool log, string inputDirectoryPath) =>
            {
                IsLogEnabled = log;
                try
                {                
                    // Calculate the asbolute path of the input directory.
                    string userInputDir = Path.GetFullPath(inputDirectoryPath);
                    // Then try to read the config file inside the directory. If not present, an exception is thrown.
                    GlobalConfiguration = GetGlobalConfiguration(userInputDir);
                    // Finally execute the train command.
                    ProgramTrain.ExecuteTrainCommand(userInputDir);
                }
                catch (Exception ex) when (ex is IOException)
                {
                    Console.WriteLine(IOErrorMessage);
                    Environment.Exit(1);
                }
                catch (Exception)
                {
                    Console.WriteLine(DirectoryErrorMessage);
                    Environment.Exit(1);
                }
            }, trainLogOption, trainArgument);

            // Create command 'test'.
            var testCommand = new Command("train", "Trains the neural network and tests it on the test set, " +
                                                   "providing output useful to understand the quality of the model.");
            var testArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            testCommand.AddArgument(testArgument);
            testCommand.SetHandler((string inputDirectoryPath) =>
            {
                // In test mode the log is always enabled, because testing a model and getting no output would not make any sense.
                IsLogEnabled = true;
                try
                {
                    // Calculate the asbolute path of the input directory.
                    string userInputDir = Path.GetFullPath(inputDirectoryPath);
                    // Then try to read the config file inside the directory. If not present, an exception is thrown.
                    GlobalConfiguration = GetGlobalConfiguration(userInputDir);
                    ProgramTest.ExecuteTestCommand(userInputDir);
                }
                catch (Exception ex) when (ex is IOException)
                {
                    Console.WriteLine(IOErrorMessage);
                    Environment.Exit(1);
                }
                catch (Exception)
                {
                    Console.WriteLine(DirectoryErrorMessage);
                    Environment.Exit(1);
                }
            }, testArgument);

            // Create command "predict".
            var predictCommand = new Command("predict", "Predicts future values using a trained neural network.");
            var predictArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            predictCommand.AddArgument(predictArgument);
            predictCommand.SetHandler((string inputDirectoryPath) =>
            {
                IsLogEnabled = true;
                try
                {
                    // Calculate the asbolute path of the input directory.
                    string userInputDir = Path.GetFullPath(inputDirectoryPath);
                    // Then try to read the config file inside the directory. If not present, an exception is thrown.
                    GlobalConfiguration = GetGlobalConfiguration(userInputDir);
                    ProgramPredict.ExecutePredictCommand(userInputDir);
                }
                catch (Exception ex) when (ex is IOException)
                {
                    Console.WriteLine(IOErrorMessage);
                    Environment.Exit(1);
                }
                catch (Exception)
                {
                    Console.WriteLine(DirectoryErrorMessage);
                    Environment.Exit(1);
                }
            }, testArgument);

            // Add the new commands to the root command.
            rootCommand.AddCommand(trainCommand);
            rootCommand.AddCommand(testCommand);
            rootCommand.AddCommand(predictCommand);
            // Process the command line and invoke the command selected by the user.
            rootCommand.Invoke(args);

            DateTime endTime = DateTime.Now;
            Console.WriteLine($"Program is completed...    {endTime}\n");

            TimeSpan elapsedTime = endTime - startTime;
            Console.WriteLine($"Elapsed time: {elapsedTime}");
        }

        /*
         * This method creates and returns a global configuration object deserializing the configuration file
         * found inside the user directory. An ArgumentException is thrown if no such file is present.
         */
        private static GlobalConfiguration GetGlobalConfiguration(string inputDirectoryPath)
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

        /*
         * This method attemps to draw a graph using python's matplotlib library.
         * The input is the name of a python script that is included in the source files of this project.
         */
        internal static (bool result, string? message) RunPythonScript(string scriptName, string outputDirectoryPath)
        {
            if ((Environment.GetEnvironmentVariable("PATH") != null) &&
                Environment.GetEnvironmentVariable("PATH")!.Contains("Python"))
            {
                string scriptPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, scriptName);
                // Create and execute a new python process to draw the graph.
                var process = new Process();
                process.StartInfo.FileName = "python";
                // The arguments are the name of the script and the path of the directory where the data is located.
                process.StartInfo.Arguments = $"{scriptPath} {outputDirectoryPath}";
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                bool res = process.Start();
                if (!res)
                {
                    return (false, "Python script could not be executed.");
                }
                process.WaitForExit();
                return (true, null);
            }
            else return (false, "  Python is not installed on your system. Please install it and try again.\n");
        }
    }
}