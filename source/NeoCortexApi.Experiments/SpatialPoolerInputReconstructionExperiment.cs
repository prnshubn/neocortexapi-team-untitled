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
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstructionExperiment)}");

            double max = 5;

            // Used as a boosting parameter to ensure homeostatic plasticity effect.
            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;

            // Use 200 bits to represent an input vector (pattern).
            int inputBits = 200;

            // Build a slice of the cortex with the given number of mini-columns.
            int numColumns = 1024;

            // Configuration parameters for the experiment.
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

            // Typical encoder parameters.
            Dictionary<string, object> settings = new Dictionary<string, object>()
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

            // Create input values from 0 to 4.
            List<double> inputValues = new();
            for (int i = 0; i < (int)max; i++)
            {
                inputValues.Add(i);
            }

            InputReconstruction(cfg, encoder, inputValues);
        }

        /// <summary>
        /// Implements the experiment.
        /// </summary>
        /// <param name="cfg"></param>
        /// <param name="encoder"></param>
        /// <param name="inputValues"></param>
        private static void InputReconstruction(HtmConfig cfg, EncoderBase encoder, List<double> inputValues)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, inputValues.Count * 40,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    isInStableState = isStable;
                    Console.WriteLine(isStable ? "STABLE STATE" : "INSTABLE STATE");
                });

            SpatialPooler sp = new(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, Column>(1) });

            CortexLayer<object, object> cortexLayer = new("L1");
            cortexLayer.HtmModules.Add("encoder", encoder);
            cortexLayer.HtmModules.Add("sp", sp);

            double[] inputs = inputValues.ToArray();

            Dictionary<double, int[]> prevActiveCols = new();
            foreach (var input in inputs)
            {
                prevActiveCols.Add(input, new int[0]);
            }

            int maxSPLearningCycles = 1000;
            int numStableCycles = 0;

            var knnClassifier = new KNeighborsClassifier<string, string>();
            var htmClassifier = new HtmClassifier<string, string>();

            // Spatial Pooler Learning Phase
            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"\nCycle  ** {cycle} ** Stability: {isInStableState}");

                foreach (var input in inputs)
                {
                    var lyrOut = cortexLayer.Compute(input, true) as int[];
                    var activeColumns = cortexLayer.GetResult("sp") as int[];
                    var actCols = activeColumns.OrderBy(c => c).ToArray();

                    double similarity = MathHelpers.CalcArraySimilarity(activeColumns, prevActiveCols[input]);
                    Console.WriteLine($"[cycle={cycle:D4}, i={input}, cols=:20 s={similarity:F2}] SDR: {Helpers.StringifyVector(actCols)}");

                    prevActiveCols[input] = activeColumns;

                    Cell[] cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();
                    knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);
                    htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);
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

            Console.WriteLine("\nClassifier training complete.");
            Console.WriteLine("\nReconstructing inputs and calculating similarity percentages...");

            // Input Reconstruction Phase
            foreach (var input in inputValues)
            {
                var sdr = cortexLayer.Compute(input, false) as int[];
                var activeColumns = cortexLayer.GetResult("sp") as int[];
                Cell[] cells = activeColumns.Select(idx => new Cell { Index = idx }).ToArray();

                Console.WriteLine($"\nInput: {input}");

                // KNN Classifier Predictions
                Console.WriteLine("KNN Classifier:");
                var knnPredictions = knnClassifier.GetPredictedInputValues(cells)
                    .Where(res => res.Similarity > 0.0)
                    .OrderByDescending(res => res.Similarity);

                foreach (var result in knnPredictions)
                {
                    double normalizedSimilarity = Math.Min(100, result.Similarity * 100);
                    Console.WriteLine($"Predicted Input: {result.PredictedInput}, Similarity: {normalizedSimilarity:F2}%");
                }

                // HTM Classifier Predictions
                Console.WriteLine("HTM Classifier:");
                var htmPredictions = htmClassifier.GetPredictedInputValues(cells)
                    .Where(res => res.Similarity > 0.0)
                    .OrderByDescending(res => res.Similarity);

                foreach (var result in htmPredictions)
                {
                    double normalizedSimilarity = Math.Min(100, result.Similarity * 100);
                    Console.WriteLine($"Predicted Input: {result.PredictedInput}, Similarity: {normalizedSimilarity:F2}%");
                }
            }
        }
    }
}
