using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using ScottPlot;

namespace NeoCortexApi.Experiments
{
    [TestClass]
    public class SpatialPoolerInputReconstruction
    {
        /// <summary>
        /// Runs the input reconstruction experiment by initializing necessary components,
        /// training the Spatial Pooler, and performing reconstruction using KNN and HTM classifiers.
        /// It also evaluates the reconstruction accuracy and plots the results for comparison.
        /// </summary>
        [TestMethod]
        [TestCategory("Experiment")]
        public void RunExperiment()
        {
            Console.WriteLine("Running Spatial Pooler Input Reconstruction Experiment...");
            double max = 5;
            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int inputBits = 200;
            int numColumns = 1024;

            HtmConfig cfg = new(new[] { inputBits }, new[] { numColumns })
            {
                CellsPerColumn = 10,
                MaxBoost = maxBoost,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = minOctOverlapCycles,
                GlobalInhibition = true,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * inputBits),
                LocalAreaDensity = -1,
                ActivationThreshold = 10,
                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 10,
            };

            Dictionary<string, object> settings = new()
            {
                { "W", 21 },
                { "N", inputBits },
                { "Radius", -1.0 },
                { "MinVal", 0.0 },
                { "MaxVal", max },
                { "Periodic", false },
                { "Name", "scalar" },
                { "ClipInput", false }
            };

            EncoderBase encoder = new ScalarEncoder(settings);
            List<double> inputValues = Enumerable.Range(0, (int)max).Select(i => (double)i).ToList();

            var sp = TrainSpatialPooler(cfg, encoder, inputValues);
            RunReconstructionExperiment(sp, encoder, inputValues);
        }

        private static SpatialPooler TrainSpatialPooler(HtmConfig cfg, EncoderBase encoder, List<double> inputValues)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;
            int numStableCycles = 0;

            HomeostaticPlasticityController hpa = new(mem, inputValues.Count * 40,
                (isStable, _, _, _) =>
                {
                    isInStableState = isStable;
                    Console.WriteLine(isStable ? "STABLE STATE REACHED" : "INSTABLE STATE");
                });

            SpatialPooler sp = new(hpa);
            sp.Init(mem,
                new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, Column>(1) });

            CortexLayer<object, object> cortexLayer = new("L1");
            cortexLayer.HtmModules.Add("encoder", encoder);
            cortexLayer.HtmModules.Add("sp", sp);

            int maxSPLearningCycles = 1000;
            Stopwatch stopwatch = Stopwatch.StartNew();

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"Cycle {cycle:D4} Stability: {isInStableState}");

                foreach (var input in inputValues)
                {
                    var sdr = cortexLayer.Compute(input, true);
                    var activeCols = sdr as int[] ?? Array.Empty<int>();
                    string sdrString = string.Join(", ", activeCols.Take(20));
                    Console.WriteLine(
                        $"[cycle={cycle:D4}, i={input:F1}, cols={activeCols.Length}] SDR: {sdrString}, ...");
                }

                if (isInStableState) numStableCycles++;
                if (numStableCycles > 5) break;
            }

            stopwatch.Stop();
            Console.WriteLine($"Spatial Pooler Training Time: {stopwatch.ElapsedMilliseconds} ms");
            return sp;
        }

        private static void RunReconstructionExperiment(SpatialPooler sp, EncoderBase encoder, List<double> inputValues)
        {
            KNeighborsClassifier<string, string> knnClassifier = new();
            HtmClassifier<string, string> htmClassifier = new();

            knnClassifier.ClearState();
            htmClassifier.ClearState();

            Dictionary<double, Cell[]> cellList = new();
            Stopwatch stopwatch = Stopwatch.StartNew();

            foreach (var input in inputValues)
            {
                var inpSdr = encoder.Encode(input);
                var actCols = sp.Compute(inpSdr, false);
                var cellArray = actCols.Select(idx => new Cell { Index = idx }).ToArray();
                cellList[input] = cellArray;
                knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cellArray);
                htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cellArray);
            }

            stopwatch.Stop();
            Console.WriteLine("\nClassifier Training Complete");
            Console.WriteLine($"Classifier Training Time: {stopwatch.ElapsedMilliseconds} ms");

            List<double> knnPredictions = new();
            List<double> htmPredictions = new();
            List<double> knnSimilarities = new();
            List<double> htmSimilarities = new();

            foreach (var input in inputValues)
            {
                Console.WriteLine($"\nInput: {input:F1}");

                var knnPrediction = knnClassifier.GetPredictedInputValues(cellList[input])[0];
                var htmPrediction = htmClassifier.GetPredictedInputValues(cellList[input])[0];

                knnPredictions.Add(Double.Parse(knnPrediction.PredictedInput));
                htmPredictions.Add(Double.Parse(htmPrediction.PredictedInput));

                var knnSimilarity = CalculateCosineSimilarity(new List<double> { input },
                    new List<double> { Double.Parse(knnPrediction.PredictedInput) });
                var htmSimilarity = CalculateCosineSimilarity(new List<double> { input },
                    new List<double> { Double.Parse(htmPrediction.PredictedInput) });

                knnSimilarities.Add(knnSimilarity);
                htmSimilarities.Add(htmSimilarity);

                Console.WriteLine(
                    $"KNN - Reconstructed: {knnPrediction.PredictedInput}, Similarity: {knnSimilarity:P2}");
                Console.WriteLine(
                    $"HTM - Reconstructed: {htmPrediction.PredictedInput}, Similarity: {htmSimilarity:P2}");
            }

            PlotReconstructionResults(inputValues, knnPredictions, htmPredictions);
            PlotSimilarityResults(inputValues, knnSimilarities, htmSimilarities);
        }

        private static void PlotReconstructionResults(List<double> inputs, List<double> knnPredictions,
            List<double> htmPredictions)
        {
            var plot = new Plot();
            plot.Add.Scatter(inputs.ToArray(), knnPredictions.ToArray()).LegendText = "KNN Predictions";
            plot.Add.Scatter(inputs.ToArray(), htmPredictions.ToArray()).LegendText = "HTM Predictions";
            plot.Title("Reconstruction Predictions");
            plot.XLabel("Input Values");
            plot.YLabel("Predictions");
            plot.Axes.AutoScale();
            SavePlot(plot, "ReconstructionPlot.png");
        }

        private static void PlotSimilarityResults(List<double> inputs, List<double> knnSimilarities,
            List<double> htmSimilarities)
        {
            var plot = new Plot();
            plot.Add.Scatter(inputs.ToArray(), knnSimilarities.ToArray()).LegendText = "KNN Similarity";
            plot.Add.Scatter(inputs.ToArray(), htmSimilarities.ToArray()).LegendText = "HTM Similarity";
            plot.Title("Similarity Comparison");
            plot.XLabel("Input Values");
            plot.YLabel("Similarity");
            plot.Axes.AutoScale();
            SavePlot(plot, "SimilarityPlot.png");
        }

        private static void SavePlot(Plot plot, string fileName)
        {
            string savePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), fileName);
            plot.Save(savePath, 600, 600);
            Console.WriteLine($"\nPlot saved at: {savePath}");
        }

        /// <summary>
        /// Calculates the cosine similarity between two vectors represented as lists of doubles.
        /// The cosine similarity measures the cosine of the angle between the two vectors.
        /// </summary>
        private static double CalculateCosineSimilarity(List<double> vectorA, List<double> vectorB)
        {
            double dotProduct = vectorA.Zip(vectorB, (a, b) => a * b).Sum();
            double magnitudeA = Math.Sqrt(vectorA.Sum(a => a * a));
            double magnitudeB = Math.Sqrt(vectorB.Sum(b => b * b));
            return dotProduct / (magnitudeA * magnitudeB);
        }
    }

    [TestClass]
public class SpatialPoolerInputReconstructionTest
{
    [TestMethod]
    public void TestReconstructionAccuracy()
    {
        var experiment = new SpatialPoolerInputReconstruction();
        experiment.RunExperiment();
        // Further assertions and checks can be added based on the output of the experiment
    }
}
}
