namespace TimeSeriesForecasting
{
    internal class ProgramTest
    {
        /*
         * The test command is just the train with some additional operations performed on the same dataset,
         * so it makes sense to defer the test operations to the same method that trains the model.
         * Avoiding this would cause the duplication of all the preliminary code that prepares the dataset for
         * training and testing, as the latter still needs the input and expected output to be retrieved from 
         * the dataset provided by the user, using the same split ratios too.
         */
        internal static void ExecuteTestCommand(string inputDirectoryAbsolutePath,
            string outputDirectoryAbsolutePath, TrainingHyperparameters hyperparameters)
        {
            ProgramTrain.ExecuteTrainCommand(inputDirectoryAbsolutePath, outputDirectoryAbsolutePath, hyperparameters, true);
        }
    }
}
