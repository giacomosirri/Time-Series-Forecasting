using System.Data;
using TimeSeriesForecasting.IO;
using static TorchSharp.torch;

namespace TimeSeriesForecasting
{
    /// <summary>
    /// This class provides functionalities to get a ready-to-use <see cref="Tuple"/>
    /// of <see cref="Tensor"/>s that can be fed as the input of the training phase of
    /// a TorchSharp neural network.
    /// </summary>
    public class WindowGenerator : IWindowGenerator
    {
        /// <summary>
        /// The number of time steps in each window.
        /// </summary>
        public int WindowSize { get => InputWidth + Offset + OutputWidth; }

        /// <summary>
        /// The number of time steps in each label, i.e. a value that the deep learning model has to predict.
        /// For single-step models, this number is equal to 1. For multi-step models, this number is greater than 1.
        /// </summary>
        public int OutputWidth { get; set; }

        /// <summary>
        /// The number of time steps in each input of the model.
        /// </summary>
        public int InputWidth { get; set; }

        /// <summary>
        /// The distance between the input's last time step and the label's first.
        /// Note that the distance is measured in terms of time steps, not time.
        /// If time steps are sampled every 10 minutes, this number is then equal to 
        /// the time distance between inputs and labels multiplied by 6.
        /// </summary>
        public int Offset { get; set; }

        /// <summary>
        /// The name of the columns that make up the labels.
        /// </summary>
        public string[] LabelColumns { get; set; }

        /// <summary>
        /// Creates an instance of window generator.
        /// </summary>
        /// <param name="inputWidth"><see cref="InputWidth"/> value.</param>
        /// <param name="outputWidth"><see cref="OutputWidth"/> value.</param>
        /// <param name="offset"><see cref="Offset"/> value.</param>
        /// <param name="labelColumns"><see cref="LabelColumns"/> value.</param>
        /// <exception cref="ArgumentException">If either input width, label width 
        /// or offset are non-positive numbers.</exception>
        public WindowGenerator(int inputWidth, int outputWidth, int offset, string[] labelColumns)
        {
            if (inputWidth > 0 && outputWidth > 0 && offset > 0)
            {
                InputWidth = inputWidth;
                OutputWidth = outputWidth;
                Offset = offset;
                LabelColumns = labelColumns;
            }
            else throw new ArgumentException("Input width, label width and offset must all be greater than 0.");
        }

        /// <summary>
        /// Generates a new window.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the table. Both the labels and the features
        /// must be of the same type.</typeparam>
        /// <param name="table">The <see cref="DataTable"/> that must be split into windows.</param>
        /// <returns>A <see cref="Tuple"/> of <see cref="Tensor"/>s. The first item in the tuple is the features
        /// tensor, while the second is the labels tensor.</returns>
        /// <exception cref="ArgumentException"></exception>
        public Tuple<Tensor, Tensor> GenerateWindows<T>(DataTable table)
        {
            var features = new List<T[][]>();
            var labels = new List<T[][]>();
            for (int startIndex = 0; startIndex + WindowSize < table.Rows.Count; startIndex++)
            {
                T[][] feature = table
                                .AsEnumerable()
                                .Skip(startIndex)
                                .Take(InputWidth)
                                .Select(dr => dr.ItemArray
                                    // Features are all the columns apart from the index of the table.
                                    .Where((_, i) => Record.Index != table.Columns[i].ColumnName)
                                    .Select(item => (T)item!)
                                    .ToArray())
                                .ToArray();
                T[][] label = table
                                .AsEnumerable()
                                .Skip(startIndex + InputWidth + Offset - 1)
                                .Take(OutputWidth)
                                .Select(dr => dr.ItemArray
                                    // Labels are all the LabelColumns columns.
                                    .Where((_, i) => LabelColumns.Contains(table.Columns[i].ColumnName))
                                    .Select(item => (T)item!)
                                    .ToArray())
                                .ToArray();
                features.Add(feature);
                labels.Add(label);
            }
            int inputSamples = features.Count;
            int inputTimeSteps = features[0].GetLength(0);
            int inputFeatures = features[0][0].GetLength(0);
            int outputSamples = labels.Count;
            int outputTimeSteps = labels[0].GetLength(0);
            int outputLabels = labels[0][0].GetLength(0);
            T[] flattenedFeatures = features.SelectMany(x => x.SelectMany(y => y)).ToArray();
            T[] flattenedLabels = labels.SelectMany(x => x.SelectMany(y => y)).ToArray();
            Tensor featuresTensor = from_array(flattenedFeatures).reshape(inputSamples, inputTimeSteps, inputFeatures).to_type(float32);
            Tensor labelsTensor = from_array(flattenedLabels).reshape(outputSamples, outputTimeSteps, outputLabels).to_type(float32);
            return Tuple.Create(featuresTensor, labelsTensor);
        }
    }
}
