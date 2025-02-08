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
    /// the performance of reconstructed inputs using both the KNN and HTM classifiers. It also includes 
    /// a learning phase for the Spatial Pooler, which helps in creating stable representations of input patterns.
    /// </summary>
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
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstruction)}");
            
            // Max value for input
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
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, Column>(1) });

            CortexLayer<object, object> cortexLayer = new ("L1");
            cortexLayer.HtmModules.Add("encoder", encoder);
            cortexLayer.HtmModules.Add("sp", sp);

            // Max iterations (cycles) for the SP learning process
            int maxSPLearningCycles = 1000;
            
            // Will hold the SDR of every input
            Dictionary<double, int[]> prevActiveCols = new ();

            // Will hold the similarity of SDKk and SDRk-1 fro every input
            Dictionary<double, double> prevSimilarity = new ();
            
            // Initiaize start similarity to zero.
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

                    Console.WriteLine($"[cycle={cycle.ToString("D4")}, i={input}, cols=:{actCols.Length} s={similarity}] SDR: {Helpers.StringifyVector(actCols)}");

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
        /// Runs the reconstruction experiment by training KNN and HTM classifiers using input values,
        /// making predictions for each input, and comparing the reconstructed inputs' similarity 
        /// to the original inputs. The reconstruction results are displayed in the console, and a plot is generated.
        /// </summary>
        private static void RunReconstructionExperiment(SpatialPooler sp, EncoderBase encoder, List<double> inputs)
        {
            KNeighborsClassifier<string, string> knnClassifier = new();
            HtmClassifier<string, string> htmClassifier = new();

            // Clear the models from all the stored sequences
            knnClassifier.ClearState();
            htmClassifier.ClearState();

            Dictionary<double, Cell[]> cellList = new();
            
            Stopwatch stopwatch = Stopwatch.StartNew();

            // Train classifiers
            foreach (var input in inputs)
            {
                var inpSdr = encoder.Encode(input);
                var actCols = sp.Compute(inpSdr, false);
                
                // Converting the int[] to Cell[] because the Learn method that as input
                var cellArray = actCols.Select(idx => new Cell { Index = idx }).ToArray();
                
                knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cellArray);
                htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cellArray);
                
                cellList[input] = cellArray;
            }
            
            stopwatch.Stop();
            Console.WriteLine("\nClassifier Training Complete");
            Console.WriteLine($"Classifier Training Time: {stopwatch.ElapsedMilliseconds} ms");
        
            List<double> knnPredictions = new();
            List<double> htmPredictions = new();
            
            Random random = new ();
            
            // Shuffling the input List - randomizing the order
            inputs = inputs.OrderBy(_ => random.Next()).ToList();
            
            foreach (var input in inputs)
            {
                Console.WriteLine($"\nInput: {input.ToString("F", CultureInfo.InvariantCulture)}");

                var knnPrediction = knnClassifier.GetPredictedInputValues(cellList[input])[0];
                var htmPrediction = htmClassifier.GetPredictedInputValues(cellList[input])[0];
                
                // This is done because HTM provides Similarity value between 0 - 100, but we want between 0 - 1
                var htmNormaLizedSimilarity = htmPrediction.Similarity / 100;
                
                Console.WriteLine($"KNN - Reconstructed: {knnPrediction.PredictedInput}, Similarity: {knnPrediction.Similarity.ToString("P", CultureInfo.InvariantCulture)}");
                Console.WriteLine($"HTM - Reconstructed: {htmPrediction.PredictedInput}, Similarity: {htmNormaLizedSimilarity.ToString("P", CultureInfo.InvariantCulture)}");
                
                // Storing the prediction for visualization
                knnPredictions.Add(Double.Parse(knnPrediction.PredictedInput));
                htmPredictions.Add(Double.Parse(htmPrediction.PredictedInput));
            }
            
            PlotResults(inputs, knnPredictions, htmPredictions);
        }

        /// <summary>
        /// Plots the reconstruction results by creating a scatter plot comparing the original input values 
        /// with the reconstructed predictions from both KNN and HTM classifiers. The plot is saved to the desktop.
        /// </summary>
        private static void PlotResults(List<double> inputs, List<double> knnPredictions, List<double> htmPredictions)
        {
            var plot = new Plot();
            plot.Add.Scatter(inputs.ToArray(), knnPredictions.ToArray()).Label = "KNN Predictions";
            plot.Add.Scatter(inputs.ToArray(), htmPredictions.ToArray()).Label = "HTM Predictions";
            plot.Title("Prediction Comparison");
            plot.XLabel("Input Values");
            plot.YLabel("Predictions");
            plot.Axes.AutoScale();

            // Method to save the plot, cross-platform compatible
            SavePlot(plot);
        }

        /// <summary>
        /// Saves the generated plot to the desktop in a cross-platform compatible way.
        /// The plot is saved as "ScalarInputReconstructionPlot.png" with specified dimensions.
        /// </summary>
        private static void SavePlot(Plot plot)
        {
            string savePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "ScalarInputReconstructionPlot.png");
            plot.Save(savePath, 600, 600);
            Console.WriteLine($"\nPlot saved at: {savePath}");
        }

        /// <summary>
        /// Cosine Similarity calculation between two vectors.
        /// </summary>
        private static double CalculateCosineSimilarity(List<double> vectorA, List<double> vectorB)
        {
            double dotProduct = vectorA.Zip(vectorB, (a, b) => a * b).Sum();
            double magnitudeA = Math.Sqrt(vectorA.Sum(a => a * a));
            double magnitudeB = Math.Sqrt(vectorB.Sum(b => b * b));
            return dotProduct / (magnitudeA * magnitudeB);
        }
    }
}
