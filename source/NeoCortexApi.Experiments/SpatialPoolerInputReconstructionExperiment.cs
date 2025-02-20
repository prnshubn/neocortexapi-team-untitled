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
            double max = 50;

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
        private static void RunReconstructionExperiment(SpatialPooler sp, EncoderBase encoder, List<double> inputValues)
        {
            Random random = new();

            // Shuffle the input List - randomizing the order
            inputValues = inputValues.OrderBy(_ => random.Next()).ToList();
            
            // Split data into training (80%) and testing (20%)
            var splitIdx = (int)(inputValues.Count * 0.8);
            var trainData = inputValues.Take(splitIdx).ToList();
            var testData = inputValues.Skip(splitIdx).ToList();
            
            KNeighborsClassifier<string, string> knnClassifier = new();
            HtmClassifier<string, string> htmClassifier = new();

            // Clear the models from all the stored sequences
            knnClassifier.ClearState();
            htmClassifier.ClearState();

            Stopwatch stopwatch = Stopwatch.StartNew();

            // Train classifiers on TRAINING DATA
            foreach (var input in trainData)
            {
                // Generate SDR for TRAINING DATA using the trained SP
                var sdr = encoder.Encode(input);
                var actCols = sp.Compute(sdr, false);

                // Converting the int[] to Cell[] because we need Cell[] format for learning
                var cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();

                knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);
                htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);
            }

            stopwatch.Stop();
            Console.WriteLine("\nClassifier Training Complete");
            Console.WriteLine($"Classifier Training Time: {stopwatch.ElapsedMilliseconds} ms");

            List<double> knnPredictions = new();
            List<double> htmPredictions = new();
            List<double> knnSimilarities = new();
            List<double> htmSimilarities = new();

            // Test on TEST DATA
            foreach (var input in testData)
            {
                Console.WriteLine($"\nInput: {input.ToString("F", CultureInfo.InvariantCulture)}");

                // Generate SDR for TEST DATA using the trained SP
                var testSdr = encoder.Encode(input);
                var testActCols = sp.Compute(testSdr, false);
                
                // Converting the int[] to Cell[] because we need Cell[] format for reconstruction
                var testCells = testActCols.Select(idx => new Cell { Index = idx }).ToArray();

                // Get predictions using the test SDR
                var knnPrediction = knnClassifier.GetPredictedInputValues(testCells)[0];
                var htmPrediction = htmClassifier.GetPredictedInputValues(testCells)[0];
                
                // This is done because HTM provides Similarity value between 0 - 100, but we want between 0 - 1
                var htmNormalizedSimilarity = htmPrediction.Similarity / 100;

                Console.WriteLine(
                    $"KNN - Reconstructed: {knnPrediction.PredictedInput}, Similarity: {knnPrediction.Similarity.ToString("P", CultureInfo.InvariantCulture)}");
                Console.WriteLine(
                    $"HTM - Reconstructed: {htmPrediction.PredictedInput}, Similarity: {htmNormalizedSimilarity.ToString("P", CultureInfo.InvariantCulture)}");

                var knnSimilarity = CalculateCosineSimilarity(new List<double> { input },
                    new List<double> { Double.Parse(knnPrediction.PredictedInput) });
                var htmSimilarity = CalculateCosineSimilarity(new List<double> { input },
                    new List<double> { Double.Parse(htmPrediction.PredictedInput) });

                // Storing the prediction for visualization
                knnPredictions.Add(Double.Parse(knnPrediction.PredictedInput));
                htmPredictions.Add(Double.Parse(htmPrediction.PredictedInput));
                knnSimilarities.Add(knnSimilarity);
                htmSimilarities.Add(htmSimilarity);
            }

            PlotReconstructionResults(testData, knnPredictions, htmPredictions);
            PlotSimilarityResults(testData, knnSimilarities, htmSimilarities);
        }

        /// <summary>
        /// Plots the reconstruction results by creating a scatter plot comparing the original input values 
        /// with the reconstructed predictions from both KNN and HTM classifiers.
        /// </summary>
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

        /// <summary>
        /// Plots the similarity results by creating a scatter plot comparing similarities
        /// of reconstructed inputs with original inputs from both KNN and HTM classifiers.
        /// </summary>
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
        
        /// <summary>
        /// Calculates the cosine similarity between two vectors represented as lists of doubles.
        /// The cosine similarity measures the cosine of the angle between the two vectors.
        /// </summary>
        private static double CalculateCosineSimilarity(List<double> vectorA, List<double> vectorB)
        {
            double dotProduct = vectorA.Zip(vectorB, (a, b) => a * b).Sum();
            double magnitudeA = Math.Sqrt(vectorA.Sum(a => a * a));
            double magnitudeB = Math.Sqrt(vectorB.Sum(b => b * b));
    
            // Handle zero magnitude edge cases
            if (magnitudeA == 0 || magnitudeB == 0)
            {
                // If either vector has zero magnitude, cosine similarity is undefined, we return 0.0
                return 0.0;
            }

            return dotProduct / (magnitudeA * magnitudeB);
        }

        /// <summary>
        /// Saves the generated plot to the desktop in a cross-platform compatible way.
        /// The plot is saved as "ScalarInputReconstructionPlot.png" with specified dimensions.
        /// </summary>
        private static void SavePlot(Plot plot, string fileName)
        {
            string savePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), fileName);
            plot.Save(savePath, 600, 600);
            Console.WriteLine($"\nPlot saved at: {savePath}");
        }
    }
}