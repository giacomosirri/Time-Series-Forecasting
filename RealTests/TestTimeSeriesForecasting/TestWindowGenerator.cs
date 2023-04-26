using System.Data;
using System.Transactions;
using TimeSeriesForecasting;
using TimeSeriesForecasting.IO;
using static TorchSharp.torch;

namespace TestTimeSeriesForecasting
{
    [Collection("Preprocessor collection")]
    public class TestWindowGenerator
    {
        private const int InputWidth = 10;
        private const int LabelWidth = 1;
        private const int Offset = 5;
        private readonly string[] LabelColumns = new string[] { "D" };

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
            IWindowGenerator winGen = new WindowGenerator(InputWidth, LabelWidth, Offset, LabelColumns);
            (_input, _output) = winGen.GenerateWindows<double>(_set);
            // Number of batches = Rows - LabelWidth - Offset - InputWidth
            _batches = _set.Rows.Count - LabelWidth - Offset - InputWidth;            
            // Number of features per observation = Number of total features - Number of label columns - 1 (timestamp column)
            _features = _set.Rows[0].ItemArray.Length - LabelColumns.Length - 1;
            // Number of labels per observation = Number of LabelColumns
            _labels = LabelColumns.Length;
        }

        [Fact]
        public void TestWindowSize()
        {
            Assert.Equal(_batches, _input.shape[0]);
            Assert.Equal(InputWidth, _input.shape[1]);
            Assert.Equal(_features, _input.shape[2]);
            Assert.Equal(_batches, _output.shape[0]);
            Assert.Equal(LabelWidth, _output.shape[1]);
            Assert.Equal(_labels, _output.shape[2]);
        }

        [Fact]
        public void TestElements()
        {
            int batch = 0; int row = 0; int col = 0;
            Assert.True(Math.Abs((double)_set.Rows[GetRow(batch, row)].ItemArray[GetColumn(col)]! - 
                _input[batch, row, col].item<float>()) < _fixture.Tolerance);
            batch = 20; row = 7; col = 1;
            double expected = (double)_set.Rows[GetRow(batch, row)].ItemArray[GetColumn(col)]!;
            double actual = _input[batch, row, col].item<float>();
            Assert.True(Math.Abs(expected - actual) < _fixture.Tolerance);
        }

        private static int GetRow(int batch, int row) => batch + row;

        private static int GetColumn(int column)
        {
            return column == 3 ? 5 : column + 1;
        }
    }
}
