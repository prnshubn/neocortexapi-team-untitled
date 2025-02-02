using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using NeoCortexApi.Utility;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using ScottPlot;

namespace NeoCortexApi.Experiments
{
    /// <summary>
    /// Demonstrates input reconstruction using Scalar Encoder, Spatial Pooler, and Classifiers (KNN & HTM).
    /// This experiment showcases the process of encoding scalar inputs, training classifiers, and evaluating 
    /// the performance of reconstructed inputs using both the KNN and HTM classifiers. It also includes 
    /// a learning phase for the Spatial Pooler, which helps in creating stable representations of input patterns.
    /// </summary>
    [TestClass]
    public class SpatialPoolerInputReconstruction
    {
        /// <summary>
        /// Setup method for the input reconstruction experiment.
        /// Initializes the Scalar Encoder, Spatial Pooler, and classifiers (KNN & HTM) for training and evaluation. 
        /// The method also sets up configuration parameters for the Spatial Pooler and Scalar Encoder, and triggers
        /// the training and evaluation processes using a set of input values with complex patterns.
        /// </summary>
        [TestMethod]
        [TestCategory("Experiment")]
        public void Setup()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstruction)} with Noise and Complex Patterns");

            double max = 5;
            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int inputBits = 200;
            int numColumns = 1024;

            // HTM configuration
            HtmConfig cfg = new (new int[] { inputBits }, new int[] { numColumns })
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

            // Instantiate encoder and input values
            EncoderBase encoder = new ScalarEncoder(settings);
            List<double> inputValues = new();
            for (double i = 0; i < max; i++)
            {
                inputValues.Add(i);
            }

            // Train the Spatial Pooler
            var sp = SpatialPoolerTraining(cfg, encoder, inputValues);
            
            // Use the trained Spatial Pooler to Reconstruct Inputs
            ReconstructionExperiment(sp, encoder, inputValues);
        }

        /// <summary>
        /// Trains the Spatial Pooler using the provided configuration and input values.
        /// The Spatial Pooler learns stable representations of the input patterns over multiple learning cycles.
        /// <param name="cfg"></param>
        /// <param name="encoder"></param>
        /// <param name="inputValues"></param>
        /// </summary>
        private static SpatialPooler SpatialPoolerTraining(HtmConfig cfg, EncoderBase encoder, List<double> inputValues)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;

            HomeostaticPlasticityController hpa = new(mem, inputValues.Count * 40,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    if (!isStable)
                    {
                        Debug.WriteLine($"INSTABLE STATE");
                        isInStableState = false;
                    }
                    else
                    {
                        Debug.WriteLine($"STABLE STATE");
                        isInStableState = true;
                    }
                });

            SpatialPooler sp = new(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, Column>(1) });

            CortexLayer<object, object> cortexLayer = new CortexLayer<object, object>("L1");
            cortexLayer.HtmModules.Add("encoder", encoder);
            cortexLayer.HtmModules.Add("sp", sp);

            double[] inputs = inputValues.ToArray();
            Dictionary<double, int[]> prevActiveCols = new();
            Dictionary<double, double> prevSimilarity = new();

            foreach (var input in inputs)
            {
                prevSimilarity.Add(input, 0.0);
                prevActiveCols.Add(input, new int[0]);
            }

            int maxSPLearningCycles = 1000;
            int numStableCycles = 0;
            Stopwatch stopwatch = Stopwatch.StartNew();

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"Cycle ** {cycle} ** Stability: {isInStableState}");

                foreach (var input in inputs)
                {
                    var lyrOut = cortexLayer.Compute((object)input, true) as int[];
                    var activeColumns = cortexLayer.GetResult("sp") as int[];
                    var actCols = activeColumns.OrderBy(c => c).ToArray();
                    double similarity = MathHelpers.CalcArraySimilarity(activeColumns, prevActiveCols[input]);

                    Console.WriteLine($"[cycle={cycle:D4}, i={input}, cols={actCols.Length}, s={similarity:F2}] SDR: {Helpers.StringifyVector(actCols)}");

                    prevActiveCols[input] = activeColumns;
                    prevSimilarity[input] = similarity;
                }

                if (isInStableState) numStableCycles++;
                if (numStableCycles > 5) break;
            }

            stopwatch.Stop();
            Console.WriteLine($"\nSpatial Pooler Training Time: {stopwatch.ElapsedMilliseconds} ms");
            return sp;
        }

        /// <summary>
        /// Evaluates the performance of the KNN and HTM classifiers after training the Spatial Pooler.
        /// The experiment predicts reconstructed inputs using both classifiers and calculates the similarity and 
        /// reconstruction errors. The results are displayed for each input.
        /// <param name="sp"></param>
        /// <param name="encoder"></param>
        /// <param name="inputValues"></param>
        /// </summary>
        private static void ReconstructionExperiment(SpatialPooler sp, EncoderBase encoder, List<double> inputValues)
        {
            // Initialize classifiers
            KNeighborsClassifier<string, string> knnClassifier = new();
            HtmClassifier<string, string> htmClassifier = new();

            // Clears the model from all the stored sequences
            knnClassifier.ClearState();
            htmClassifier.ClearState();

            Dictionary<double, Cell[]> cellList = new();
            Stopwatch stopwatch = Stopwatch.StartNew();

            // Train classifiers
            foreach (var input in inputValues)
            {
                var inpSdr = encoder.Encode(input);
                var actCols = sp.Compute(inpSdr, false);
                var cellArray = actCols.Select(idx => new Cell { Index = idx }).ToArray();
                cellList.Add(input, cellArray);

                knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cellArray);
                htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cellArray);
            }
            
            stopwatch.Stop();
            Console.WriteLine("\nTraining the classifier is complete");
            Console.WriteLine($"\nClassifier Training Time: {stopwatch.ElapsedMilliseconds} ms");
        
            // Shuffle the list to randomize the order
            Random random = new();
            // inputValues = inputValues.OrderBy(_ => random.Next()).ToList();
            
            // After training, display reconstruction performance and prepare data for plotting
            List<double> knnPredictions = new();
            List<double> htmPredictions = new();
            
            foreach (var input in inputValues)
            {
                Console.WriteLine($"\nInput: {input:F1}");
                
                // KNN Classifier Reconstruction
                Console.WriteLine("KNN Classifier");
                var knnPrediction = knnClassifier.GetPredictedInputValues(cellList[input])[0];
                var normalizedSimilarity = knnPrediction.Similarity * 100;
                Console.WriteLine($"Reconstructed Input: {knnPrediction.PredictedInput}, Similarity: {normalizedSimilarity.ToString("F2", CultureInfo.InvariantCulture)}%");
                knnPredictions.Add(Double.Parse(knnPrediction.PredictedInput));

                // HTM Classifier Reconstruction
                Console.WriteLine("HTM Classifier");
                var htmPrediction = htmClassifier.GetPredictedInputValues(cellList[input])[0];
                Console.WriteLine($"Reconstructed Input: {htmPrediction.PredictedInput}, Similarity: {htmPrediction.Similarity.ToString("F2", CultureInfo.InvariantCulture)}%");
                htmPredictions.Add(Double.Parse(htmPrediction.PredictedInput));
            }
            
            // Plot the results using ScottPlot
            PlotAndDisplayGraph(inputValues, knnPredictions, htmPredictions);
        }
        
        private static void PlotAndDisplayGraph(
            List<double> inputs,
            List<double> knnPredictions,
            List<double> htmPredictions)
        {
            var plot = new Plot();

            double[] x = inputs.ToArray();
            double[] yKnn = knnPredictions.ToArray();
            double[] yHtm = htmPredictions.ToArray();

            // Add scatter plots
            var knnScatter = plot.Add.Scatter(x, yKnn);
            knnScatter.Label = "KNN Predictions";
            knnScatter.Color = Colors.Blue;

            var htmScatter = plot.Add.Scatter(x, yHtm);
            htmScatter.Label = "HTM Predictions";
            htmScatter.Color = Colors.Orange;

            // Customize plot
            plot.Title("Prediction Comparison");
            plot.XLabel("Input Values");
            plot.YLabel("Predictions");
            plot.Axes.AutoScale();

            // macOS-specific path handling
            string savePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "ReconstructionPlot.png");
            plot.Save(savePath, 600, 600);
            Console.WriteLine($"Plot saved at: {savePath}");

            // macOS file opening command
            Process.Start("open", savePath);
        }
    }
}