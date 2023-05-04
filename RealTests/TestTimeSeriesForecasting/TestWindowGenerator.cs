using System.Data;
using TimeSeriesForecasting;
using static TorchSharp.torch;

namespace TestTimeSeriesForecasting
{
    [Collection("Preprocessor collection")]
    public class TestWindowGenerator
    {
        private const int InputWidth = 10;
        private const int LabelWidth = 1;
        private const int Offset = 5;
        
        private readonly string[] _labelColumns = new string[] { "D" };
        private readonly PreprocessorFixture _fixture;
        private readonly DataTable _set;
        private readonly Tensor _input;
        private readonly Tensor _output;
        private readonly int _batches;
        private readonly int _features;
        private readonly int _labels;

        public TestWindowGenerator(PreprocessorFixture fixture)
        {
            _fixture = fixture;
            _set = fixture.Preprocessor.GetTrainingSet();
            IWindowGenerator winGen = new WindowGenerator(InputWidth, LabelWidth, Offset, _labelColumns);
            (_input, _output) = winGen.GenerateWindows<double>(_set);
            // Number of batches = Rows - LabelWidth - Offset - InputWidth
            _batches = _set.Rows.Count - LabelWidth - Offset - InputWidth;            
            // Number of features per time step = Number of total features - 1 (timestamp column)
            _features = _set.Rows[0].ItemArray.Length - 1;
            // Number of labels per time step = Number of LabelColumns
            _labels = _labelColumns.Length;
        }

        [Fact]
        public void TestWindowSize()
        {
            Assert.Equal(_batches, _input.size(0));
            Assert.Equal(InputWidth, _input.size(1));
            Assert.Equal(_features, _input.size(2));
            Assert.Equal(_batches, _output.size(0));
            Assert.Equal(LabelWidth, _output.size(1));
            Assert.Equal(_labels, _output.size(2));
        }

        [Fact]
        public void TestInputElements()
        {
            var rnd = new Random(5678);
            // Compares 500 values taken randomly from the dataset to see if the data has been split correctly into windows.
            for (int i = 0; i < 500; i++)
            {
                int batch = rnd.Next(0, _batches);
                int row = rnd.Next(0, InputWidth);
                int col = rnd.Next(0, _features);
                // col+1 because the _set DataTable contains the Date Time column while the input tensor does not.
                double expected = (double)_set.Rows[batch + row].ItemArray[col+1]!;
                double actual = _input[batch, row, col].item<float>();
                Assert.True(Math.Abs(expected - actual) < _fixture.Tolerance);
            }
        }

        [Fact]
        public void TestOutputElements()
        {
            var rnd = new Random(3456);
            // Compares 500 values taken randomly from the dataset to see if the data has been split correctly into windows.
            for (int i = 0; i < 500; i++)
            {
                int batch = rnd.Next(0, _batches);
                int row = rnd.Next(0, LabelWidth);
                int col = rnd.Next(0, _labels);
                // Indexing of the row to find a label column, since the output is obviously made of label column values only.
                double expected = (double)_set.Rows[batch + row + InputWidth + Offset][_labelColumns[col]];
                double actual = _output[batch, row, col].item<float>();
                Assert.True(Math.Abs(expected - actual) < _fixture.Tolerance);
            }
        }
    }
}
