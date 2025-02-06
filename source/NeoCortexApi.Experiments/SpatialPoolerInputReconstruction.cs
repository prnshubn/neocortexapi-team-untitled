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
    /// <summary>
    /// Demonstrates input reconstruction using Scalar Encoder, Spatial Pooler, and Classifiers (KNN & HTM).
    /// This experiment encodes scalar inputs, trains classifiers, and evaluates input reconstruction performance.
    /// </summary>
    [TestClass]
    public class SpatialPoolerInputReconstruction
    {
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

            HtmConfig cfg = new(new int[] { inputBits }, new int[] { numColumns })
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

        private static SpatialPooler TrainSpatialPooler(HtmConfig cfg, EncoderBase encoder, List<double> inputValues)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;
            int numStableCycles = 0;

            HomeostaticPlasticityController hpa = new(mem, inputValues.Count * 40,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    isInStableState = isStable;
                    Console.WriteLine(isStable ? "STABLE STATE REACHED" : "INSTABLE STATE");
                });

            SpatialPooler sp = new(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, Column>(1) });

            CortexLayer<object, object> cortexLayer = new CortexLayer<object, object>("L1");
            cortexLayer.HtmModules.Add("encoder", encoder);
            cortexLayer.HtmModules.Add("sp", sp);

            int maxSPLearningCycles = 1000;
            Stopwatch stopwatch = Stopwatch.StartNew();
            
            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"Cycle {cycle:D4} Stability: {isInStableState}");
                
                foreach (var input in inputValues)
                {
                    var sdr = cortexLayer.Compute((object)input, true);
                    LogSDROutput(cycle, input, sdr);
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

            // Train classifiers
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
            
            foreach (var input in inputValues)
            {
                Console.WriteLine($"\nInput: {input:F1}");

                var knnPrediction = knnClassifier.GetPredictedInputValues(cellList[input])[0];
                var htmPrediction = htmClassifier.GetPredictedInputValues(cellList[input])[0];

                knnPredictions.Add(Double.Parse(knnPrediction.PredictedInput));
                htmPredictions.Add(Double.Parse(htmPrediction.PredictedInput));

                Console.WriteLine($"KNN - Reconstructed: {knnPrediction.PredictedInput}, Similarity: {knnPrediction.Similarity:P2}");
                Console.WriteLine($"HTM - Reconstructed: {htmPrediction.PredictedInput}, Similarity: {htmPrediction.Similarity:P2}");
            }

            PlotResults(inputValues, knnPredictions, htmPredictions);
        }

        private static void LogSDROutput(int cycle, double input, object sdr)
        {
            var activeCols = sdr as int[] ?? Array.Empty<int>();
            string sdrString = string.Join(", ", activeCols.Take(20)); // Displaying only first 20 values
            Console.WriteLine($"[cycle={cycle:D4}, i={input:F1}, cols={activeCols.Length}] SDR: {sdrString}, ...");
        }

        private static void PlotResults(List<double> inputs, List<double> knnPredictions, List<double> htmPredictions)
        {
            var plot = new Plot();
            plot.Add.Scatter(inputs.ToArray(), knnPredictions.ToArray()).Label = "KNN Predictions";
            plot.Add.Scatter(inputs.ToArray(), htmPredictions.ToArray()).Label = "HTM Predictions";
            plot.Title("Prediction Comparison");
            plot.XLabel("Input Values");
            plot.YLabel("Predictions");
            plot.Axes.AutoScale();

            string savePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "ReconstructionPlot.png");
            plot.Save(savePath, 600, 600);
            Console.WriteLine($"Plot saved at: {savePath}");
        }
    }
}
