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
        private readonly string[] LabelColumns = new string[] { "D" };

        private DataTable _set;
        private Tuple<Tensor, Tensor> _tensors;

        public TestWindowGenerator(PreprocessorFixture fixture)
        {
            _set = fixture.Preprocessor.GetTrainingSet();
            IWindowGenerator winGen = new WindowGenerator(InputWidth, LabelWidth, Offset, LabelColumns);
            _tensors = winGen.GenerateWindows<double>(_set);
        }

        [Fact]
        public void TestWindowSize()
        {
            Tensor input = _tensors.Item1;
            // Number of batches = Rows - LabelWidth - Offset - InputWidth
            Assert.Equal(_set.Rows.Count - LabelWidth - Offset - InputWidth, input.shape[0]);
            // Number of observations in input = InputWidth
            Assert.Equal(InputWidth, input.shape[1]);
            // Number of features per observation = Number of total features - Number of label columns - 1 (timestamp column)
            Assert.Equal(_set.Rows[0].ItemArray.Length - LabelColumns.Length - 1, input.shape[2]);
            Tensor output = _tensors.Item2;
            // Number of batches in input and output tensors must be equal
            Assert.Equal(input.shape[0], output.shape[0]);
            // Number of observations in output = LabelWidth
            Assert.Equal(LabelWidth, output.shape[1]);
            // Number of labels per observation = Number of LabelColumns
            Assert.Equal(LabelColumns.Length, output.shape[2]);
        }

        [Fact]
        public void TestElements()
        {
        }
    }
}
