﻿using System.Data;
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
        private static readonly string s_errorMessage = "Missing value in a table cell. Please provide a complete table.";
        private static readonly string[] s_indexColumns = { "Date Time" };

        /// <summary>
        /// The number of time steps in each window.
        /// </summary>
        public int WindowSize { get => InputWidth + Offset + OutputWidth; }

        /// <summary>
        /// The number of time steps in each label, i.e. a value that the deep learning model has to predict.
        /// For single-output models, this number is equal to 1. For multi-output models, this number is greater than 1.
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
                                    .Where((_, i) => !s_indexColumns.Contains(table.Columns[i].ColumnName))
                                    .Select(item => item != null ? (T)item : throw new ArgumentException(s_errorMessage))
                                    .ToArray())
                                .ToArray();
                T[][] label = table
                                .AsEnumerable()
                                .Skip(startIndex + InputWidth + Offset)
                                .Take(OutputWidth)
                                .Select(dr => dr.ItemArray
                                    .Where((_, i) => LabelColumns.Contains(table.Columns[i].ColumnName))
                                    .Select(item => item != null ? (T)item : throw new ArgumentException(s_errorMessage))
                                    .ToArray())
                                .ToArray();
                features.Add(feature);
                labels.Add(label);
            }
            T[][][] featuresArr = features.ToArray();
            int featuresBatchSize = featuresArr.GetLength(0);
            int inputTimeSteps = featuresArr[0].GetLength(0);
            int featuresSize = featuresArr[0][0].GetLength(0);
            T[][][] labelsArr = labels.ToArray();
            int labelsBatchSize = labelsArr.GetLength(0);
            int outputTimeSteps = labelsArr[0].GetLength(0);
            int labelsSize = labelsArr[0][0].GetLength(0);
            T[] flattenedFeatures = featuresArr.SelectMany(x => x.SelectMany(y => y)).ToArray();
            T[] flattenedLabels = labels.SelectMany(x => x.SelectMany(y => y)).ToArray();
            Tensor featuresTensor = from_array(flattenedFeatures)
                                    .reshape(featuresBatchSize, inputTimeSteps, featuresSize)
                                    .to_type(float32);
            Tensor labelsTensor = from_array(flattenedLabels)
                                    .reshape(labelsBatchSize, outputTimeSteps, labelsSize)
                                    .to_type(float32);
            return Tuple.Create(featuresTensor, labelsTensor);
        }

        /// <summary>
        /// Clones this window generator object. The returned object has the same parameters of this one,
        /// so that it can be used to operate on related data (e.g. to split the training, validation and test
        /// sets using the same parameters).
        /// </summary>
        /// <returns>A new <see cref="WindowGenerator"/> with the same parameters as the one this method is called on.</returns>
        public WindowGenerator Clone()
        {
            return new WindowGenerator(InputWidth, OutputWidth, Offset, LabelColumns);
        }
    }
}
