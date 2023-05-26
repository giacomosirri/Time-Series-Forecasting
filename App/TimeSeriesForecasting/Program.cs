using Newtonsoft.Json;
using System.CommandLine;
using System.Diagnostics;
using System.Reflection;
using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
{
    /*
     * This class encapsulates all the global configuration parameters.
     * These parameters are needed for every running mode, as the input to both the training 
     * and the prediction is a parquet table, which has one index column and one or more 
     * label columns and which must be split into windows of data (usually called "time series").
     */
    public class GlobalConfiguration
    {
        // The name of the columns that contain the values to predict.
        public string[] LabelColumns { internal get; set; } = Array.Empty<string>();

        // The name of the primary key (or index).
        public string IndexColumn { set => Record.Index = value; }

        // The number of time steps in the input to the model.
        public int InputWidth { internal get; set; }

        // The number of time steps in the output that the model must produce.
        public int OutputWidth { internal get; set; }

        // The distance in time steps between the input and the output.
        public int Offset { internal get; set; }
    }


    internal class TrainingHyperparameters
    {
        internal int? BatchSize { get; private set; }

        internal int? Epochs { get; private set; }

        internal double? LearningRate { get; private set; }

        internal TrainingHyperparameters(int? batchSize, int? epochs, double? learningRate)
        {
            BatchSize = batchSize;
            Epochs = epochs;
            LearningRate = learningRate;
        }
    }


    internal class Program
    {
        internal static readonly string ValuesFile = "data-values.parquet";
        internal static readonly string DatesFile = "data-dates.parquet";
        internal static readonly string DataSubdirectory = "data";
        internal static readonly string ModelSubdirectory = "model";

        internal const string Completion = "  COMPLETE\n";
        internal const string DirectoryErrorMessage = "The directory you provided is not valid. The program has been stopped.";
        private const string IOErrorMessage = "An error occurred when reading a directory or file. The program has been stopped.";
        private const string GlobalConfigFile = "global_config.json";

        internal static bool IsLogEnabled { get; private set; }

        internal static GlobalConfiguration GlobalConfiguration { get; private set; } = new GlobalConfiguration();

        static void Main(string[] args)
        {
            DateTime startTime = DateTime.Now;
            Console.WriteLine($"Program is running...    {startTime}\n");


            // Create command line.
            var rootCommand = new RootCommand("App that creates, trains and runs a neural network for time series forecasting.");

            var outputOption = new Option<string>(
                name: "--output",
                description: "internal means that the output is placed inside the same directory you provided as input, while " +
                "external means that the app creates a new directory and places all the output there."
            )
            {
                IsRequired = false
            };
            outputOption.FromAmong("internal", "external");
            outputOption.SetDefaultValue("internal");
            outputOption.AddAlias("--o");

            rootCommand.AddGlobalOption(outputOption);


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
            trainLogOption.SetDefaultValue(false);
            trainLogOption.AddAlias("--l");

            var trainBatchSizeOption = new Option<int?>(
                name: "--batch_size",
                description: "Specifies the batch size."
            )
            {
                IsRequired = false
            };
            trainBatchSizeOption.SetDefaultValue(null);

            var trainEpochsOption = new Option<int?>(
                name: "--epochs",
                description: "Specifies the number of epochs."
            )
            {
                IsRequired = false
            };
            trainEpochsOption.SetDefaultValue(null);

            var trainLearningRateOption = new Option<double?>(
                name: "--learning_rate",
                description: "Specifies the learning rate."
            )
            {
                IsRequired = false
            };
            trainLearningRateOption.SetDefaultValue(null);

            trainCommand.AddOption(trainLogOption);
            trainCommand.AddOption(trainBatchSizeOption);
            trainCommand.AddOption(trainEpochsOption);
            trainCommand.AddOption(trainLearningRateOption);

            var trainArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            trainCommand.AddArgument(trainArgument);

            trainCommand.SetHandler((bool log, int? batchSize, int? epochs, double? learningRate, string output, string inputDirectoryPath) =>
            {
                IsLogEnabled = log;
                GlobalConfiguration = GetConfigurationOrExit(inputDirectoryPath);
                var hyperparameters = new TrainingHyperparameters(batchSize, epochs, learningRate);
                string outputDirectoryPath = GetOutputDirectory(output, inputDirectoryPath);
                ProgramTrain.ExecuteTrainCommand(Path.GetFullPath(inputDirectoryPath), outputDirectoryPath, hyperparameters);
            }, trainLogOption, trainBatchSizeOption, trainEpochsOption, trainLearningRateOption, outputOption, trainArgument);


            // Create command "test".
            var testCommand = new Command("test", "Trains the neural network and tests it on the test set, " +
                                                  "providing output useful to understand the quality of the model.");

            var testBatchSizeOption = new Option<int?>(
                name: "--batch_size",
                description: "Specifies the batch size."
            )
            {
                IsRequired = false
            };
            testBatchSizeOption.SetDefaultValue(null);

            var testEpochsOption = new Option<int?>(
                name: "--epochs",
                description: "Specifies the number of epochs."
            )
            {
                IsRequired = false
            };
            testEpochsOption.SetDefaultValue(null);

            var testLearningRateOption = new Option<double?>(
                name: "--learning_rate",
                description: "Specifies the learning rate."
            )
            {
                IsRequired = false
            };
            testLearningRateOption.SetDefaultValue(null);

            testCommand.AddOption(testBatchSizeOption);
            testCommand.AddOption(testEpochsOption);
            testCommand.AddOption(testLearningRateOption);

            var testArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            testCommand.AddArgument(testArgument);

            testCommand.SetHandler((string output, int? batchSize, int? epochs, double? learningRate, string inputDirectoryPath) =>
            {
                // In test mode the log is always enabled, because testing a model and not getting any output would make no sense.
                IsLogEnabled = true;
                GlobalConfiguration = GetConfigurationOrExit(inputDirectoryPath);
                var hyperparameters = new TrainingHyperparameters(batchSize, epochs, learningRate);
                string outputDirectoryPath = GetOutputDirectory(output, inputDirectoryPath);
                ProgramTest.ExecuteTestCommand(Path.GetFullPath(inputDirectoryPath), outputDirectoryPath, hyperparameters);
            }, outputOption, testBatchSizeOption, testEpochsOption, testLearningRateOption, testArgument);


            // Create command "predict".
            var predictCommand = new Command("predict", "Predicts future values using a trained neural network.");

            var predictBatchSizeOption = new Option<int?>(
                name: "--batch_size",
                description: "Specifies the batch size."
            )
            {
                IsRequired = false
            };
            predictBatchSizeOption.SetDefaultValue(null);

            predictCommand.AddOption(predictBatchSizeOption);

            var predictArgument = new Argument<string>("input", "The relative or absolute path of the input directory.");
            predictCommand.AddArgument(predictArgument);

            predictCommand.SetHandler((string output, int? batchSize, string inputDirectoryPath) =>
            {
                IsLogEnabled = true;
                GlobalConfiguration = GetConfigurationOrExit(inputDirectoryPath);
                string outputDirectoryPath = GetOutputDirectory(output, inputDirectoryPath);
                ProgramPredict.ExecutePredictCommand(Path.GetFullPath(inputDirectoryPath), outputDirectoryPath, batchSize);
            }, outputOption, predictBatchSizeOption, testArgument);


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
         * This method checks if the given directory path contains a valid /data subdirectory.
         */
        internal static bool IsUserDataSubdirectoryValid(string directoryPath)
        {
            try
            {
                // A /data subdirectory must exist.
                string dataDir = Directory.GetDirectories(directoryPath, DataSubdirectory).Single();
                // A file called data-values.parquet must exist inside /data.
                _ = Directory.GetFiles(dataDir, ValuesFile).Single();
                // A file called data-datetimes.parquet must exist inside /data.
                _ = Directory.GetFiles(dataDir, DatesFile).Single();
                // If all the necessary files and subdirectories have been found, then return true.
                return true;
            }
            catch (Exception ex) when (ex is ArgumentNullException || ex is InvalidOperationException)
            {
                // If any necessary file or subdirectory is not found, then return false.
                return false;
            }
        }

        /*
         * This method attemps to retrieve a global configuration object from the input directory.
         * If it fails, the program is stopped.
         */
        private static GlobalConfiguration GetConfigurationOrExit(string inputDirectoryPath)
        {
            try
            {
                // Try to find the file "global_config.json" inside the input directory.
                string globalConfigFile = Directory.GetFiles(Path.GetFullPath(inputDirectoryPath), GlobalConfigFile).Single();
                // Then return the global configuration object obtained through the deserialization of the configuration file.
                return GetConfiguration<GlobalConfiguration>(globalConfigFile);
            }
            catch (Exception ex) when (ex is IOException)
            {
                StopProgram(IOErrorMessage);
                return new GlobalConfiguration();
            }
            // If either the directory provided by the user does not exist or is inaccessible or its path is too long,
            // or if there is no json configuration file inside it, then the program is stopped.
            catch (Exception)
            {
                StopProgram(DirectoryErrorMessage);
                return new GlobalConfiguration();
            }
        }

        /*
         * This method creates and returns a configuration object deserializing the configuration file
         * provided as an argument. An ArgumentException is thrown if no such file is present.
         */
        internal static T GetConfiguration<T>(string configFile)
        {
            try
            {
                using var reader = new StreamReader(configFile);
                var settings = new JsonSerializerSettings() { DateFormatString = "yyyy-MM-dd" };
                return JsonConvert.DeserializeObject<T>(reader.ReadToEnd(), settings)!;
            }
            // Distinguish between an unpredictable I/O problem and a wrong user input. 
            catch (Exception ex) when (ex is IOException)
            {
                throw new IOException("Something went wrong when reading the directory or the config file.");
            }
            catch (Exception)
            {
                throw new ArgumentException("The given path is not a directory or does not contain the requested file.");
            }
        }

        /*
         * This method prints the given error message to the standard output, then terminates this process.
         */
        internal static void StopProgram(string errorMessage)
        {
            Console.WriteLine(errorMessage);
            Environment.Exit(1);
        }

        /*
         * This method returns the absolute path of the output directory based on the value specified by the user 
         * for the option "output". This method also ensures that the requested output directory exists after its completion.
         */
        private static string GetOutputDirectory(string optionValue, string inputDirectory)
        {
            if (optionValue == "external")
            {
                string parentDir = Directory.GetParent(inputDirectory)!.FullName;
                int i = 0;
                while (true)
                {
                    string newDir = Path.Combine(new string[] { parentDir, $"Output{(i==0 ? "" : $"({i})")}" });
                    i++;
                    if (!Directory.Exists(newDir))
                    {
                        return Directory.CreateDirectory(Path.Combine(new string[] { parentDir, newDir })).FullName;
                    }
                }
            }
            else return inputDirectory;
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