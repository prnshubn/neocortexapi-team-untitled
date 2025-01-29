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

namespace NeoCortexApi.Experiments
{
    /// <summary>
    /// Demonstrates input reconstruction using Scalar Encoder, Spatial Pooler, and Classifiers (KNN & HTM).
    /// </summary>
    [TestClass]
    public class SpatialPoolerInputReconstruction
    {
        private static List<string> classifierLogs = new List<string>();

        [TestMethod]
        [TestCategory("Experiment")]
        public void SetupWithNoiseAndComplexPatterns()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstruction)} with Noise and Complex Patterns");

            double max = 20;
            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int inputBits = 200;
            int numColumns = 1024;

            HtmConfig cfg = new HtmConfig(new int[] { inputBits }, new int[] { numColumns })
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

            // Create complex input values with noise
            List<double> inputValues = GenerateNoisyInputValues(max);

            var sp = SpatialPoolerTraining(cfg, encoder, inputValues);
            ReconstructionExperiment(sp, encoder, inputValues);
        }

        /// <summary>
        /// Generates noisy input values by adding random noise to a linear sequence of numbers.
        /// </summary>
        /// <param name="max">The maximum value for the input range.</param>
        /// <returns>A list of noisy input values.</returns>
        private static List<double> GenerateNoisyInputValues(double max)
        {
            List<double> inputValues = new List<double>();
            Random rand = new Random();

            // Generate a mix of linear and random noisy values
            for (int i = 0; i < max; i++)
            {
                double noise = rand.NextDouble() * 2 - 1; // noise between -1 and 1
                double value = i + noise; // add noise to a simple sequence
                inputValues.Add(Math.Max(0, Math.Min(value, max))); // ensure it's within bounds
            }

            // Optionally, add more complex patterns here (e.g., non-linear sequences, etc.)
            return inputValues;
        }

        /// <summary>
        /// Trains the Spatial Pooler with the provided configuration, encoder, and noisy input values.
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

            CortexLayer<object, object> cortexLayer = new("L1");
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

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"Cycle  ** {cycle} ** Stability: {isInStableState}");

                foreach (var input in inputs)
                {
                    double similarity;
                    var lyrOut = cortexLayer.Compute((object)input, true) as int[];
                    var activeColumns = cortexLayer.GetResult("sp") as int[];
                    var actCols = activeColumns.OrderBy(c => c).ToArray();

                    similarity = MathHelpers.CalcArraySimilarity(activeColumns, prevActiveCols[input]);
                    Console.WriteLine($"[cycle={cycle.ToString("D4")}, i={input}, cols=:{actCols.Length} s={similarity}] SDR: {Helpers.StringifyVector(actCols)}");

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

            return sp;
        }

        /// <summary>
        /// Runs the reconstruction experiment for both KNN and HTM classifiers, and calculates reconstruction errors.
        /// </summary>
        private static void ReconstructionExperiment(SpatialPooler sp, EncoderBase encoder, List<double> inputValues)
        {
            // Initialize KNN and HTM classifiers
            KNeighborsClassifier<string, string> knnClassifier = new();
            HtmClassifier<string, string> htmClassifier = new();
            knnClassifier.ClearState();
            htmClassifier.ClearState();

            // Process each input value
            foreach (var input in inputValues)
            {
                var inpSdr = encoder.Encode(input);
                var actCols = sp.Compute(inpSdr, false);
                
                Cell[] cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();
                knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);
                htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);

                Console.WriteLine($"\nInput: {input}");

                // KNN Classifier Prediction
                Console.WriteLine("KNN Classifier");
                var knnPredictions = knnClassifier.GetPredictedInputValues(cells);
                foreach (var result in knnPredictions)
                {
                    Console.WriteLine($"Predicted Input: {result.PredictedInput}, Similarity: {result.Similarity}");
                    // Calculate Reconstruction Error for KNN
                    CalculateReconstructionError(input, knnPredictions);
                }

                // HTM Classifier Prediction
                Console.WriteLine("HTM Classifier");
                var htmPredictions = htmClassifier.GetPredictedInputValues(cells);
                foreach (var result in htmPredictions)
                {
                    Console.WriteLine($"Predicted Input: {result.PredictedInput}, Similarity: {result.Similarity}");
                    // Calculate Reconstruction Error for HTM
                    CalculateReconstructionError(input, htmPredictions);
                }
            }
            // Display classifier performance logs
            DisplayClassifierLogs();
              
        }

        /// <summary>
        /// Calculates and prints the reconstruction error based on the predicted input.
        /// </summary>
        private static void CalculateReconstructionError(double originalInput, IEnumerable<NeoCortexApi.Classifiers.ClassifierResult<string>> predictions)
        {
            foreach (var prediction in predictions)
            {
                // Extract the predicted input value from the classifier result
                double predictedInput = Convert.ToDouble(prediction.PredictedInput);
                double error = Math.Abs(originalInput - predictedInput); // Calculate absolute error
                Console.WriteLine($"Reconstruction Error: {error} for predicted input: {predictedInput}");
            }
        }
        /// <summary>
        /// Displays the classifier logs after each experiment.
        /// </summary>
        private static void DisplayClassifierLogs()
        {
            Console.WriteLine("\nClassifier Performance Logs:");
            foreach (var log in classifierLogs)
            {
                Console.WriteLine(log);
            }
        }
        
    }
}
