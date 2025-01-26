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
        
        [TestMethod]
        [TestCategory("Experiment")]
        public void Setup()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstruction)}");

            double max = 20;

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
            Dictionary<string, object> settings = new ()
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

            var sp = SpatialPoolerTraining(cfg, encoder, inputValues);
            ReconstructionExperiment(sp, encoder, inputValues);
        }

        /// <summary>
        /// Training the SP.
        /// </summary>
        /// <param name="cfg"></param>
        /// <param name="encoder"></param>
        /// <param name="inputValues"></param>
        private static SpatialPooler SpatialPoolerTraining(HtmConfig cfg, EncoderBase encoder, List<double> inputValues)
        {
            // Creates the htm memory.
            var mem = new Connections(cfg);

            bool isInStableState = false;

            // HPC extends the default Spatial Pooler algorithm.
            // The purpose of HPC is to set the SP in the new-born stage at the beginning of the learning process.
            // In this stage the boosting is very active, but the SP behaves instable. After this stage is over
            // (defined by the second argument) the HPC is controlling the learning process of the SP.
            // Once the SDR generated for every input gets stable, the HPC will fire event that notifies your code
            // that SP is stable now.
            HomeostaticPlasticityController hpa = new (mem, inputValues.Count * 40,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    // Event should only be fired when entering the stable state.
                    // Ideal SP should never enter unstable state after stable state.
                    if (!isStable)
                    {
                        Debug.WriteLine($"INSTABLE STATE");
                        // This should usually not happen.
                        isInStableState = false;
                    }
                    else
                    {
                        Debug.WriteLine($"STABLE STATE");
                        // Here you can perform any action if required.
                        isInStableState = true;
                    }
                });

            SpatialPooler sp = new (hpa);

            sp.Init(mem,
                new DistributedMemory()
                    { ColumnDictionary = new InMemoryDistributedDictionary<int, Column>(1) });

            // It creates the instance of the neo-cortex layer.
            // Algorithm will be performed inside of that layer.
            CortexLayer<object, object> cortexLayer = new CortexLayer<object, object>("L1");

            // Add encoder as the very first module. This model is connected to the sensory input cells
            // that receive the input. Encoder will receive the input and forward the encoded signal
            // to the next module.
            cortexLayer.HtmModules.Add("encoder", encoder);

            // The next module in the layer is Spatial Pooler. This module will receive the output of the encoder
            cortexLayer.HtmModules.Add("sp", sp);

            double[] inputs = inputValues.ToArray();

            // Will hold the SDR of every input.
            Dictionary<double, int[]> prevActiveCols = new ();

            // Will hold the similarity of SDKk and SDRk-1 fro every input.
            Dictionary<double, double> prevSimilarity = new ();

            // Initialize start similarity to zero.
            foreach (var input in inputs)
            {
                prevSimilarity.Add(input, 0.0);
                prevActiveCols.Add(input, new int[0]);
            }

            // Learning process will take 1000 iterations (cycles)
            int maxSPLearningCycles = 1000;

            int numStableCycles = 0;
            
            

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"Cycle  ** {cycle} ** Stability: {isInStableState}");

                // This trains the layer on input pattern.
                foreach (var input in inputs)
                {
                    double similarity;

                    // Learn the input pattern.
                    // Output lyrOut is the output of the last module in the layer.
                    var lyrOut = cortexLayer.Compute((object)input, true) as int[];

                    // This is a general way to get the SpatialPooler result from the layer.
                    var activeColumns = cortexLayer.GetResult("sp") as int[];

                    var actCols = activeColumns.OrderBy(c => c).ToArray();

                    similarity = MathHelpers.CalcArraySimilarity(activeColumns, prevActiveCols[input]);

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
        
        private static void ReconstructionExperiment(SpatialPooler sp, EncoderBase encoder, 
            List<double> inputValues)
        {
            // Initialize classifiers.
            KNeighborsClassifier<string, string> knnClassifier = new ();
            HtmClassifier<string, string> htmClassifier = new ();
            
            // Clear all learned patterns in the classifier.
            knnClassifier.ClearState();
            htmClassifier.ClearState();

            // Loop through each input value in the list of input values.
            foreach (var input in inputValues)
            {
                // Encode the current input value using the provided encoder, resulting in an SDR.
                var inpSdr = encoder.Encode(input);

                // Compute the active columns in the spatial pooler for the given input SDR, without learning.
                var actCols = sp.Compute(inpSdr, false);
                
                Cell[] cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();
                
                // Training the classifiers
                knnClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);
                htmClassifier.Learn(input.ToString("F2", CultureInfo.InvariantCulture), cells);

                Console.WriteLine($"\nInput: {input}");

                // KNN Classifier Prediction
                Console.WriteLine("KNN Classifier");
                var knnPredictions = knnClassifier.GetPredictedInputValues(cells);

                foreach (var result in knnPredictions)
                {
                    Console.WriteLine($"Predicted Input: {result.PredictedInput}, Similarity: {result.Similarity}");
                }

                // HTM Classifier Prediction
                Console.WriteLine("HTM Classifier");
                var htmPredictions = htmClassifier.GetPredictedInputValues(cells);

                foreach (var result in htmPredictions)
                {
                    Console.WriteLine($"Predicted Input: {result.PredictedInput}, Similarity: {result.Similarity}");
                }
            }
        }
        
    }
}