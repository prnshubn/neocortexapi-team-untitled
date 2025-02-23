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
using NeoCortexApi.Experiments;
using NeoCortexApi.Network;
using NeoCortexApi.Utility;
using ScottPlot;

namespace NeoCortexApi.Experiments
{
    /// <summary>
    /// Demonstrates input reconstruction using Scalar Encoder, Spatial Pooler, and Classifiers (KNN & HTM).
    /// This experiment showcases the process of encoding scalar inputs, training classifiers, and evaluating 
    /// the similarity of reconstructed inputs using both the KNN and HTM classifiers. It also includes 
    /// a learning phase for the Spatial Pooler, which helps in creating stable representations of input patterns.
    /// </summary>
    [TestClass]
    public class SpatialPoolerInputReconstructionExperiment
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
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstructionExperiment)}");

            // Max value for input
            double max = 20;

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

            // Scalar Encoder settings
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

            // Train the Spatial Pooler
            var sp = TrainSpatialPooler(cfg, encoder, inputValues);

            // Perform Reconstruction Experiment
            RunReconstructionExperiment(sp, encoder, inputValues);
        }

        /// <summary>
        /// Trains the Spatial Pooler by initializing its components, running a learning phase, 
        /// and iterating through a predefined number of cycles to achieve stable representation 
        /// of the input patterns. It logs the training cycle details and measures the training time.
        /// </summary>
        /// <param name="cfg"></param>
        /// <param name="encoder"></param>
        /// <param name="inputs"></param>
        /// <returns></returns>
        private static SpatialPooler TrainSpatialPooler(HtmConfig cfg, EncoderBase encoder, List<double> inputs)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;
            int numStableCycles = 0;

            HomeostaticPlasticityController hpa = new(mem, inputs.Count * 40,
                (isStable, numPatterns, actColAvg, seenInputs) =>
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

            // Max iterations (cycles) for the SP learning process
            int maxSPLearningCycles = 1000;

            // Will hold the SDR of every input
            Dictionary<double, int[]> prevActiveCols = new();

            // Will hold the similarity of SDKk and SDRk-1 from every input
            Dictionary<double, double> prevSimilarity = new();

            // Initialize start similarity to zero.
            foreach (var input in inputs)
            {
                prevSimilarity.Add(input, 0.0);
                prevActiveCols.Add(input, new int[0]);
            }

            Stopwatch stopwatch = Stopwatch.StartNew();

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"Cycle {cycle:D4} Stability: {isInStableState}");

                // This trains the layer on input pattern
                foreach (var input in inputs)
                {
                    // Learn the input pattern
                    // Output lyrOut is the output of the last module in the layer
                    var lyrOut = cortexLayer.Compute((object)input, true) as int[];

                    // This is a general way to get the SpatialPooler result from the layer
                    var activeColumns = cortexLayer.GetResult("sp") as int[];

                    var actCols = activeColumns.OrderBy(c => c).ToArray();

                    double similarity = MathHelpers.CalcArraySimilarity(activeColumns, prevActiveCols[input]);

                    Console.WriteLine(
                        $"[cycle={cycle.ToString("D4")}, i={input}, cols=:{actCols.Length} s={similarity}] SDR: {Helpers.StringifyVector(actCols)}");

                    prevActiveCols[input] = activeColumns;
                    prevSimilarity[input] = similarity;
                }

                if (isInStableState)
                {
                    numStableCycles++;
                }

                if (numStableCycles > 5)
                {
                    break;
                }
            }

            stopwatch.Stop();
            Console.WriteLine($"\nSpatial Pooler Training Time: {stopwatch.ElapsedMilliseconds} ms");
            return sp;
        }

        /// <summary>
        /// Runs the reconstruction experiment by training KNN and HTM classifiers using input values,
        /// making predictions for each input, and comparing the reconstructed inputs' similarity 
        /// to the original inputs. The reconstruction results are displayed in the console, and a plot is generated.
        /// </summary>
        /// <param name="sp"></param>
        /// <param name="encoder"></param>
        /// <param name="inputValues"></param>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        private static void RunReconstructionExperiment(SpatialPooler sp, EncoderBase encoder, List<double> inputValues)
        {
            if (sp == null) throw new ArgumentNullException(nameof(sp));
            if (encoder == null) throw new ArgumentNullException(nameof(encoder));
            if (inputValues == null || !inputValues.Any())
                throw new ArgumentException("Input values cannot be null or empty", nameof(inputValues));

            double min = inputValues.Min();
            double max = inputValues.Max();

            // As we are dividing the inout set into two parts for training and testing,
            // there could a bias the classifiers toward lower values and make the test set
            // unrepresentative of the full range. Hence, we are shuffling the list.
            Random random = new();

            // Shuffle the input List
            inputValues = inputValues.OrderBy(_ => random.Next()).ToList();

            // Split data into training (80%) and testing (20%)
            var splitIdx = (int)(inputValues.Count * 0.8);
            var trainDataSet = inputValues.Take(splitIdx).ToList();
            var testDataSet = inputValues.Skip(splitIdx).ToList();

            KNeighborsClassifier<string, string> knnClassifier = new();
            HtmClassifier<string, string> htmClassifier = new();

            // Clear the models from all the stored sequences
            knnClassifier.ClearState();
            htmClassifier.ClearState();

            Stopwatch stopwatch = Stopwatch.StartNew();

            // Train classifiers on TRAINING DATA
            foreach (var trainData in trainDataSet)
            {
                // Generate SDR for TRAINING DATA using the trained SP
                var sdr = encoder.Encode(trainData);
                var actCols = sp.Compute(sdr, false);

                // Converting the int[] to Cell[] because we need Cell[] format for learning
                var cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();

                knnClassifier.Learn(trainData.ToString("F2", CultureInfo.InvariantCulture), cells);
                htmClassifier.Learn(trainData.ToString("F2", CultureInfo.InvariantCulture), cells);
            }

            stopwatch.Stop();
            Console.WriteLine("\nClassifier Training Complete");
            Console.WriteLine($"Classifier Training Time: {stopwatch.ElapsedMilliseconds} ms");

            List<double> knnPredictions = new();
            List<double> htmPredictions = new();
            List<double> knnSimilarities = new();
            List<double> htmSimilarities = new();

            // Test on TEST DATA
            foreach (var testData in testDataSet)
            {
                Console.WriteLine($"\nInput: {testData.ToString("F", CultureInfo.InvariantCulture)}");

                // Generate SDR for TEST DATA using the trained SP
                var testSdr = encoder.Encode(testData);
                var testActCols = sp.Compute(testSdr, false);

                // Converting the int[] to Cell[] because we need Cell[] format for reconstruction
                var testCells = testActCols.Select(idx => new Cell { Index = idx }).ToArray();

                // Get predictions using the test SDR
                var knnPrediction = knnClassifier.GetPredictedInputValues(testCells)[0];
                var htmPrediction = htmClassifier.GetPredictedInputValues(testCells)[0];

                // This is done because HTM provides Similarity value between 0 - 100, but we want between 0 - 1
                var htmNormalizedSimilarity = htmPrediction.Similarity / 100;

                Console.WriteLine($"KNN - Reconstructed Input: {knnPrediction.PredictedInput}");
                Console.WriteLine($"KNN - Internal Similarity: {knnPrediction.Similarity.ToString("P", CultureInfo.InvariantCulture)}");
                Console.WriteLine($"KNN - Percentage Similarity: {CalculatePercentageSimilarity(testData, double.Parse(knnPrediction.PredictedInput, CultureInfo.InvariantCulture), min, max)}");
                Console.WriteLine($"HTM - Reconstructed Input: {htmPrediction.PredictedInput}");
                Console.WriteLine($"HTM - Internal Similarity: {htmNormalizedSimilarity.ToString("P", CultureInfo.InvariantCulture)}");
                Console.WriteLine($"HTM - Percentage Similarity: {CalculatePercentageSimilarity(testData, double.Parse(htmPrediction.PredictedInput, CultureInfo.InvariantCulture), min, max)}");

                var knnSimilarity = knnPrediction.Similarity;
                var htmSimilarity = htmPrediction.Similarity;

                // Add per-input comparison
                string betterClassifier = knnSimilarity > htmSimilarity ? "KNN" : "HTM";
                Console.WriteLine($"{betterClassifier} performed better for this input");

                // Storing the prediction for visualization
                knnPredictions.Add(Double.Parse(knnPrediction.PredictedInput));
                htmPredictions.Add(Double.Parse(htmPrediction.PredictedInput));
                knnSimilarities.Add(knnSimilarity);
                htmSimilarities.Add(htmSimilarity);
            }

            PlotReconstructionResults(testDataSet, knnPredictions, htmPredictions);
            PlotSimilarityResults(testDataSet, knnSimilarities, htmSimilarities);

            // Analyze the results
            AnalyzeResults(testDataSet, knnPredictions, htmPredictions, knnSimilarities, htmSimilarities);
        }

        /// <summary>
        /// Plots the reconstruction results by creating a scatter plot comparing the original input values 
        /// with the reconstructed predictions from both KNN and HTM classifiers.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="knnPredictions"></param>
        /// <param name="htmPredictions"></param>
        private static void PlotReconstructionResults(List<double> inputs, List<double> knnPredictions,
            List<double> htmPredictions)
        {
            var plot = new Plot();
            plot.Add.Scatter(inputs.ToArray(), knnPredictions.ToArray()).LegendText = "KNN Predictions";
            plot.Add.Scatter(inputs.ToArray(), htmPredictions.ToArray()).LegendText = "HTM Predictions";
            plot.Title("Reconstruction Predictions");
            plot.XLabel("Input Values");
            plot.YLabel("Predictions");
            plot.Axes.SetLimits(0, 20, 0, 20); // Set axes limits
            SavePlot(plot, "ReconstructionPlot.png");
        }

        /// <summary>
        /// Plots the similarity results by creating a scatter plot comparing similarities
        /// of reconstructed inputs with original inputs from both KNN and HTM classifiers.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="knnSimilarities"></param>
        /// <param name="htmSimilarities"></param>
        private static void PlotSimilarityResults(List<double> inputs, List<double> knnSimilarities, List<double> htmSimilarities)
        {
            var plot = new Plot();
            plot.Add.Scatter(inputs.ToArray(), knnSimilarities.Select(s => s * 100).ToArray()).LegendText = "KNN Similarity";
            plot.Add.Scatter(inputs.ToArray(), htmSimilarities.Select(s => s * 100).ToArray()).LegendText = "HTM Similarity";
            plot.Title("Similarity Comparison");
            plot.XLabel("Input Values");
            plot.YLabel("Similarity (%)");
            plot.Axes.SetLimits(0, 20, 0, 20); // Fixed X-axis to match reconstruction plot
            SavePlot(plot, "SimilarityPlot.png");
        }


        /// <summary>
        /// Saves the generated plot to the desktop in a cross-platform compatible way.
        /// The plot is saved as "ScalarInputReconstructionPlot.png" with specified dimensions.
        /// </summary>
        /// <param name="plot"></param>
        /// <param name="fileName"></param>
        private static void SavePlot(Plot plot, string fileName)
        {
            string savePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), fileName);
            plot.Save(savePath, 600, 600);
            Console.WriteLine($"\nPlot saved at: {savePath}");
        }

        /// <summary>
        /// Analyzes and discusses the results of the reconstruction experiment.
        /// </summary>
        private static void AnalyzeResults(List<double> inputs, List<double> knnPredictions, List<double> htmPredictions, List<double> knnSimilarities, List<double> htmSimilarities)
        {
            // Calculate Mean Absolute Error (MAE)
            double knnMAE = inputs.Zip(knnPredictions, (a, p) => Math.Abs(a - p)).Average();
            double htmMAE = inputs.Zip(htmPredictions, (a, p) => Math.Abs(a - p)).Average();

            // Calculate average similarity (convert to percentage)
            double knnAvgSimilarity = knnSimilarities.Average() * 100;
            double htmAvgSimilarity = htmSimilarities.Average();

            Console.WriteLine("\nResults Analysis:");
            Console.WriteLine($"Average KNN Similarity: {knnAvgSimilarity:F2}%");
            Console.WriteLine($"Average HTM Similarity: {htmAvgSimilarity:F2}%");
            Console.WriteLine($"KNN Mean Absolute Error: {knnMAE:F2}");
            Console.WriteLine($"HTM Mean Absolute Error: {htmMAE:F2}");

            // Enhanced comparison
            bool htmBetter = htmAvgSimilarity > knnAvgSimilarity;
            Console.WriteLine(htmBetter ?
                "HTM performed better than KNN in reconstructing inputs." :
                "KNN performed better than HTM in reconstructing inputs.");
        }

        /// <summary>
        /// Calculates the Absolute Percentage Similarity between the Original Input
        /// and the Reconstructed Input.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        private static string CalculatePercentageSimilarity(double value1, double value2, double min, double max)
        {
            double range = max - min;

            double difference = Math.Abs(value1 - value2);
            double similarity = (1 - (difference / range)) * 100;

            // Ensure similarity is not negative.
            similarity = Math.Max(0, similarity);

            return similarity.ToString("F2", CultureInfo.InvariantCulture) + "%";
        } }

            [TestClass]
            public class SpatialPoolerInputReconstructionTest
        {
            [TestMethod]
        public void TestReconstructionAccuracy()
            {
                var experiment = new SpatialPoolerInputReconstructionExperiment();
                experiment.RunExperiment();
                // Further assertions and checks can be added based on the output of the experiment
            }
        }

        [TestClass]
        public class SpatialPoolerTrainingTest
        {
            [TestMethod]
            public void TestTrainingTime()
            {
                var experiment = new SpatialPoolerInputReconstructionExperiment();
                experiment.RunExperiment();
                // Test that the training time is within expected bounds
            }
        }

        [TestClass]
        public class SimilarityComparisonTest
        {
            [TestMethod]
        public void TestSimilarityResults()
            {
                var experiment = new SpatialPoolerInputReconstructionExperiment();
                experiment.RunExperiment();
                // Test that the similarity results are within expected bounds
            }
        }
    }
    


