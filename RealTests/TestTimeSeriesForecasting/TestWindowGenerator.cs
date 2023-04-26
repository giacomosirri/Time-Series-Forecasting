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

        private readonly PreprocessorFixture _fixture;

        public TestWindowGenerator(PreprocessorFixture fixture) => _fixture = fixture;

        [Fact]
        public void TestWindowGeneration()
        {
            DataTable trainSet = _fixture.Preprocessor.GetTrainingSet();
            IWindowGenerator winGen = new WindowGenerator(InputWidth, LabelWidth, Offset, LabelColumns);
            Tuple<Tensor, Tensor> tensors = winGen.GenerateWindows<double>(trainSet);
            Assert.Equal(trainSet.Rows.Count - LabelWidth - Offset - InputWidth, tensors.Item1.shape[0]);
            Assert.Equal(InputWidth, tensors.Item1.shape[1]);
            Assert.Equal(trainSet.Rows[0].ItemArray.Length - LabelColumns.Length - 1, tensors.Item1.shape[2]);
        }
    }
}
