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

            double max = 20;
            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int inputBits = 200;
            int numColumns = 1024;

            // HTM configuration
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
            for (int i = 0; i < (int)max; i++)
            {
                inputValues.Add(i * 0.1); // Using 1-digit decimal inputs
            }

            // Train the Spatial Pooler and perform the reconstruction experiment
            var sp = SpatialPoolerTraining(cfg, encoder, inputValues);
            ReconstructionExperiment(sp, encoder, inputValues);
        }

        /// <summary>
        /// Trains the Spatial Pooler using the provided configuration and input values.
        /// The Spatial Pooler learns stable representations of the input patterns over multiple learning cycles.
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

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"Cycle  ** {cycle} ** Stability: {isInStableState}");

                foreach (var input in inputs)
                {
                    var lyrOut = cortexLayer.Compute((object)input, true) as int[];
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

            return sp;
        }

        /// <summary>
        /// Evaluates the performance of the KNN and HTM classifiers after training the Spatial Pooler.
        /// The experiment predicts reconstructed inputs using both classifiers and calculates the similarity and 
        /// reconstruction errors. The results are displayed for each input.
        /// </summary>
        private static void ReconstructionExperiment(SpatialPooler sp, EncoderBase encoder, List<double> inputValues)
        {
            // Initialize classifiers.
            KNeighborsClassifier<string, string> knnClassifier = new();
            HtmClassifier<string, string> htmClassifier = new();

            knnClassifier.ClearState();
            htmClassifier.ClearState();

            List<Cell[]> cellList = new();

            // Train classifiers
            foreach (var input in inputValues)
            {
                var inpSdr = encoder.Encode(input);
                var actCols = sp.Compute(inpSdr, false);
                var cellArray = actCols.Select(idx => new Cell { Index = idx }).ToArray();
                cellList.Add(cellArray);

                knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cellArray);
                htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cellArray);
            }

            Console.WriteLine("Training the classifier is complete\n");

            // After training, display reconstruction performance and prepare data for plotting
            List<double> knnPredictions = new();
            List<double> htmPredictions = new();
            List<double> similarities = new();

            foreach (var input in inputValues)
            {
                Console.WriteLine($"\nInput: {input:F1}"); // Format input to 1 decimal place

                // KNN Classifier Prediction
                var knnPredictionsList = knnClassifier.GetPredictedInputValues(cellList[(int)(input * 10)]); // Adjusted index for 1 decimal point input
                var knnBestPrediction = knnPredictionsList.OrderBy(p => p.Similarity).First(); // Get the best prediction
                double predictedKnnInput = double.Parse(knnBestPrediction.PredictedInput);
                double knnError = Math.Abs(input - predictedKnnInput); // Calculate the error as the absolute difference
                Console.WriteLine($"KNN Classifier: Reconstructed Input: {predictedKnnInput:F1}, Similarity: {knnBestPrediction.Similarity:F2}, Reconstruction Error: {knnError:F2}");

                // HTM Classifier Prediction
                var htmPredictionsList = htmClassifier.GetPredictedInputValues(cellList[(int)(input * 10)]); // Adjusted index for 1 decimal point input
                var htmBestPrediction = htmPredictionsList.OrderBy(p => p.Similarity).First(); // Get the best prediction
                double predictedHtmInput = double.Parse(htmBestPrediction.PredictedInput);
                double htmError = Math.Abs(input - predictedHtmInput); // Calculate the error as the absolute difference
                Console.WriteLine($"HTM Classifier: Reconstructed Input: {predictedHtmInput:F1}, Similarity: {htmBestPrediction.Similarity:F2}, Reconstruction Error: {htmError:F2}");

                // Store data for plotting
                knnPredictions.Add(predictedKnnInput);
                htmPredictions.Add(predictedHtmInput);
                similarities.Add(knnError);  // You could also store the HTM error if preferred
            }

            // Now, plot the results using ScottPlot
            PlotAndDisplayGraph(inputValues, knnPredictions, htmPredictions, similarities);
        }

        private static void PlotAndDisplayGraph(List<double> inputs, List<double> knnPredictions, List<double> htmPredictions, List<double> similarities)
        {
            // Create a new plot (no need to pass width/height in constructor)
            var plt = new Plot();

            // Add scatter plots for KNN and HTM predictions
            plt.AddScatter(inputs.ToArray(), knnPredictions.ToArray(), label: "KNN Predictions");
            plt.AddScatter(inputs.ToArray(), htmPredictions.ToArray(), label: "HTM Predictions");

            // Optionally, add the reconstruction error as a scatter plot as well
            plt.AddScatter(inputs.ToArray(), similarities.ToArray(), label: "Reconstruction Error");

            // Customize the plot with labels and title
            plt.XLabel("Input Values");
            plt.YLabel("Predictions / Error");
            plt.Legend();

            // Save the plot as an image file with specified size
            plt.SaveFig("ReconstructionPlot.png", 600, 400); // Save the plot as an image file
            Console.WriteLine("Plot saved to 'ReconstructionPlot.png'.");
        }



    }
}