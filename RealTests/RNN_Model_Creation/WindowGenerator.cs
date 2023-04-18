using System.Data;
using static TorchSharp.torch;

namespace TimeSeriesForecasting
{
    /// <summary>
    /// This class provides functionalities to get a ready-to-use <see cref="Tuple"/>
    /// of <see cref="Tensor"/>s that can be fed as the input of the training phase of
    /// a deep learning TorchSharp pipeline.
    /// </summary>
    internal class WindowGenerator : IWindowGenerator
    {
        private static readonly string errorMessage = "Missing value in a table cell. Please provide a complete table.";

        /// <summary>
        /// The number of observations in each window.
        /// </summary>
        public int WindowSize { get => LabelWidth + InputWidth + Offset; }

        /// <summary>
        /// The number of observations in each label, i.e. a value that the deep learning model has to predict.
        /// For single-output models, this number is equal to 1. For multi-output models, this number is greater than 1.
        /// </summary>
        public int LabelWidth { get; set; }

        /// <summary>
        /// The number of observations in each input of the model.
        /// </summary>
        public int InputWidth { get; set; }

        /// <summary>
        /// The distance between the input's last observation and the label's first.
        /// Note that the distance is measured in terms of observation, not of time.
        /// If observations are sampled every 10 minutes, this number is then equal to 
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
        /// <param name="labelWidth"><see cref="LabelWidth"/> value.</param>
        /// <param name="offset"><see cref="Offset"/> value.</param>
        /// <param name="labelColumns"><see cref="LabelColumns"/> value.</param>
        /// <exception cref="ArgumentException">If either input width, label width 
        /// or offset are non-positive numbers.</exception>
        public WindowGenerator(int inputWidth, int labelWidth, int offset, string[] labelColumns)
        {
            if (inputWidth > 0 && labelWidth > 0 && offset > 0)
            {
                InputWidth = inputWidth;
                LabelWidth = labelWidth;
                Offset = offset;
                LabelColumns = labelColumns;
            }
            else throw new ArgumentException("Input width, label width and offset must all be greater than 0.");

        }

        /// <summary>
        /// Generates a new window.
        /// </summary>
        /// <typeparam name="T"></typeparam>
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
                                    .Where((_, i) => !LabelColumns.Contains(table.Columns[i].ColumnName))
                                    .Select(item => item != null ? (T)item : throw new ArgumentException(errorMessage))
                                    .ToArray())
                                .ToArray();
                T[][] label = table
                                .AsEnumerable()
                                .Skip(startIndex + Offset)
                                .Take(LabelWidth)
                                .Select(dr => dr.ItemArray
                                    .Where((_, i) => LabelColumns.Contains(table.Columns[i].ColumnName))
                                    .Select(item => item != null ? (T)item : throw new ArgumentException(errorMessage))
                                    .ToArray())
                                .ToArray();
                features.Add(feature);
                labels.Add(label);
            }
            return Tuple.Create(from_array(features.ToArray()), from_array(labels.ToArray()));
        }

        /// <summary>
        /// Clones this window generator object. The returned object has the same parameters of this one,
        /// so that it can be used to operate on related data (e.g. to split the training, validation and test
        /// sets using the same parameters).
        /// </summary>
        /// <returns>A new <see cref="WindowGenerator"/> with the same parameters as the one this method is called on.</returns>
        public WindowGenerator Clone()
        {
            return new WindowGenerator(InputWidth, LabelWidth, Offset, LabelColumns);
        }
    }
}
