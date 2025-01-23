using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using NeoCortexApi.Utility;

namespace NeoCortexApi.Experiments
{
    /// <summary>
    /// Demonstrates input reconstruction using Scalar Encoder, Spatial Pooler, and HTM Classifiers.
    /// </summary>
    [TestClass]
    public class SpatialPoolerInputReconstructionExperiment
    {
        [TestMethod]
        [TestCategory("Experiment")]
        public void Setup()
        {
            Console.WriteLine($"Starting Experiment: {nameof(SpatialPoolerInputReconstructionExperiment)}");

            double maxInputValue = 5; // Maximum value for inputs.

            // Spatial Pooler configuration parameters.
            double minOverlapDutyCycle = 1.0;
            double maxBoostFactor = 5.0;
            int inputBits = 200; // Number of input bits in the encoder.
            int numColumns = 1024; // Number of mini-columns in the Spatial Pooler.

            // Initialize configuration for the Spatial Pooler.
            HtmConfig config = new HtmConfig(new int[] { inputBits }, new int[] { numColumns })
            {
                CellsPerColumn = 10,
                MaxBoost = maxBoostFactor,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = minOverlapDutyCycle,

                GlobalInhibition = false,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * inputBits),
                LocalAreaDensity = -1,
                ActivationThreshold = 10,

                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 10,
            };

            // Encoder settings for the scalar values.
            var encoderSettings = new Dictionary<string, object>
            {
                { "W", 21 },
                { "N", inputBits },
                { "Radius", -1.0 },
                { "MinVal", 0.0 },
                { "MaxVal", maxInputValue },
                { "Periodic", false },
                { "Name", "scalar" },
                { "ClipInput", false }
            };
            EncoderBase encoder = new ScalarEncoder(encoderSettings);

            // Generate sequential input values for the experiment.
            List<double> inputValues = Enumerable.Range(0, (int)maxInputValue).Select(i => (double)i).ToList();

            // Run the input reconstruction experiment.
            InputReconstruction(config, encoder, inputValues);
        }

        /// <summary>
        /// Implements the input reconstruction experiment using Spatial Pooler and HTM classifiers.
        /// </summary>
        private static void InputReconstruction(HtmConfig config, EncoderBase encoder, List<double> inputValues)
        {
            // Initialize connections and plasticity controller.
            var connections = new Connections(config);
            bool isStableState = false;

            var plasticityController = new HomeostaticPlasticityController(
                connections,
                inputValues.Count * 40,
                (isStable, _, _, _) =>
                {
                    Console.WriteLine(isStable ? "STABLE STATE" : "INSTABLE STATE");
                    isStableState = isStable;
                });

            // Initialize Spatial Pooler.
            SpatialPooler spatialPooler = new(plasticityController);
            spatialPooler.Init(connections, new DistributedMemory
            {
                ColumnDictionary = new InMemoryDistributedDictionary<int, Column>(1)
            });

            // Create a Cortex Layer and add modules.
            CortexLayer<object, object> cortexLayer = new("L1");
            cortexLayer.HtmModules.Add("encoder", encoder);
            cortexLayer.HtmModules.Add("sp", spatialPooler);

            // Initialize classifiers.
            var knnClassifier = new KNeighborsClassifier<string, string>();
            var htmClassifier = new HtmClassifier<string, string>();

            // Track previous active columns and similarities.
            Dictionary<double, int[]> previousActiveColumns = inputValues.ToDictionary(input => input, _ => Array.Empty<int>());
            Dictionary<double, double> previousSimilarity = inputValues.ToDictionary(input => input, _ => 0.0);

            const int maxCycles = 1000; // Maximum Spatial Pooler training cycles.
            int stableCycleCount = 0;

            // Training phase: Learn input patterns.
            for (int cycle = 0; cycle < maxCycles; cycle++)
            {
                Console.WriteLine($"\nCycle: {cycle:D4} | Stability: {isStableState}");

                foreach (var input in inputValues)
                {
                    // Compute Spatial Pooler output for the current input.
                    var activeColumns = cortexLayer.Compute(input, true) as int[];
                    var sortedActiveColumns = activeColumns.OrderBy(c => c).ToArray();

                    // Calculate similarity with the previous input's active columns.
                    double similarity = MathHelpers.CalcArraySimilarity(activeColumns, previousActiveColumns[input]);

                    Console.WriteLine($"[Cycle={cycle:D4}, Input={input}, ActiveCols={sortedActiveColumns.Length}, Similarity={similarity:F2}] SDR: {Helpers.StringifyVector(sortedActiveColumns)}");

                    // Update tracking dictionaries.
                    previousActiveColumns[input] = activeColumns;
                    previousSimilarity[input] = similarity;

                    // Train the classifiers with the current input and active columns.
                    var cells = sortedActiveColumns.Select(idx => new Cell { Index = idx }).ToArray();
                    knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);
                    htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);
                }

                if (isStableState)
                {
                    stableCycleCount++;
                    if (stableCycleCount > 5)
                        break; // Exit if stability is achieved for sufficient cycles.
                }
            }

            Console.WriteLine("\nClassifier training complete.");

            // Testing phase: Reconstruct inputs and evaluate similarity percentages.
            Console.WriteLine("\nReconstructing inputs and calculating similarity percentages...");
            foreach (var input in inputValues)
            {
                var activeColumns = cortexLayer.Compute(input, false) as int[];
                var cells = activeColumns.Select(idx => new Cell { Index = idx }).ToArray();

                Console.WriteLine($"\nInput: {input}");

                // Evaluate KNN Classifier predictions.
                Console.WriteLine("KNN Classifier:");
                foreach (var prediction in knnClassifier.GetPredictedInputValues(cells))
                {
                    double similarityPercentage = Math.Min(100, prediction.Similarity * 100);
                    Console.WriteLine($"Predicted Input: {prediction.PredictedInput}, Similarity: {similarityPercentage:F2}%");
                }

                // Evaluate HTM Classifier predictions.
                Console.WriteLine("HTM Classifier:");
                foreach (var prediction in htmClassifier.GetPredictedInputValues(cells))
                {
                    double similarityPercentage = Math.Min(100, prediction.Similarity * 100);
                    Console.WriteLine($"Predicted Input: {prediction.PredictedInput}, Similarity: {similarityPercentage:F2}%");
                }
            }
        }
    }
}