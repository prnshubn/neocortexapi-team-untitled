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
using ScottPlot.Plottables;

namespace NeoCortexApi.Experiments
{
    /// <summary>
    ///     <see href="https://github.com/prnshubn/neocortexapi-team-untitled">Project Link</see><br />
    ///     <see href="https://github.com/prnshubn/neocortexapi-team-untitled/tree/master/source/Team_Untitled_Files/Documentation">Documentation</see>
    ///     <br />
    ///     Demonstrates input reconstruction using Scalar Encoder, Spatial Pooler, and Classifiers (KNN and HTM).
    ///     This experiment showcases the process of encoding scalar inputs, training classifiers, and evaluating
    ///     the similarity of reconstructed inputs using both the KNN and HTM classifiers. It also includes
    ///     a learning phase for the Spatial Pooler, which helps in creating stable representations of input patterns.
    /// </summary>
    /// <br />
    /// <para>
    ///     University: Frankfurt University of Applied Sciences<br />
    ///     Degree: Master of Engineering in Information Technology<br />
    ///     Year: 2024-2025<br />
    ///     Team: Untitled<br />
    ///     Contributors:
    ///     <see href="https://github.com/prnshubn">Priyanshu Bandyopadhyay</see>,
    ///     <see href="https://github.com/Hanumanthumanoj01">Manoj Hanumanthu</see>,
    ///     <see href="https://github.com/Akshay-Gudekar">Akshay Gudekar</see>
    /// </para>
    /// <para>
    ///     Test cases are present in <b>SpatialPoolerInputReconstructionExperimentTests</b>
    /// </para>
    /// <para>
    ///     To run the experiment please use the "run" method in the "Run" Class.
    /// </para>
    
    [TestClass]
    public class ExperimentRunner
    {
        
        [TestMethod]
        public void run()
        {
            SpatialPoolerInputReconstructionExperiment experiment = new();
            
            // Please provide value greater than 10
            experiment.ReconstructionExperiment(50);
        }
    }
    
    public class SpatialPoolerInputReconstructionExperiment
    {
        // Private field to track output file path
        private string outputFilePath;
        
        // Properties to store results
        public Dictionary<double, (double KnnReconstructedInput,
                double HtmReconstructedInput,
                double KnnInternalSimilarity,
                double HtmInternalSimilarity,
                double KnnPercentageSimilarity,
                double HtmPercentageSimilarity)>
            Results { get; } = new();

        /// <summary>
        ///     Runs the input reconstruction experiment by initializing necessary components,
        ///     training the Spatial Pooler, and performing reconstruction using KNN and HTM classifiers.
        ///     It also evaluates the reconstruction accuracy and plots the results for comparison.
        /// </summary>
        /// <param name="max">Starting from 1 the maximum input you want to try reconstructing</param>
        /// <param name="seedValue">Needed later in reconstruction step to check if experiment is providing desired result</param>
        public void ReconstructionExperiment(double max, int seedValue=0)
        {
            if (max < 10) throw new ArgumentException("max must be 10 or greater", nameof(max));
            
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstructionExperiment)}"); 
            
            // Initialize output file
            InitializeOutputFile();

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
                StimulusThreshold = 10
            };

            // Scalar Encoder settings
            Dictionary<string, object> settings = new()
            {
                { "W", 21 },
                { "N", inputBits },
                { "Radius", -1.0 },
                { "MinVal", 1.0 },
                { "MaxVal", max },
                { "Periodic", false },
                { "Name", "scalar" },
                { "ClipInput", false }
            };
            
            EncoderBase encoder = new ScalarEncoder(settings);
            
            // Generating a list of scalar inputs from 1 to max
            List<double> inputValues = Enumerable.Range(1, (int)max).Select(i => (double)i).ToList();

            // Train the Spatial Pooler
            SpatialPooler sp = TrainSpatialPooler(cfg, encoder, inputValues);

            // Perform Reconstruction Experiment
            CompareClassifiers(sp, encoder, inputValues, seedValue);
        }

        /// <summary>
        ///     Train the Spatial Pooler by initializing its components, running a learning phase,
        ///     and iterating through a predefined number of cycles to achieve stable representation
        ///     of the input patterns. Log the training cycle details and measures the training time.
        /// </summary>
        /// <param name="cfg">The configuration for HTM</param>
        /// <param name="encoder">Encoder to use for converting inout to SDR</param>
        /// <param name="inputs">List of inputs to be checked for reconstruction</param>
        /// <returns>The trained version of the SP</returns>
        private SpatialPooler TrainSpatialPooler(HtmConfig cfg, EncoderBase encoder, List<double> inputs)
        {
            Connections mem = new(cfg);
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
                new DistributedMemory { ColumnDictionary = new InMemoryDistributedDictionary<int, Column>(1) });

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
            foreach (double input in inputs)
            {
                prevSimilarity.Add(input, 0.0);
                prevActiveCols.Add(input, new int[0]);
            }

            Stopwatch stopwatch = Stopwatch.StartNew();

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                string cycleInfo = $"Cycle {cycle:D4} Stability: {isInStableState}";
                Debug.WriteLine(cycleInfo);
                
                // Write last stable state to file
                if (isInStableState)
                {
                    UpdateOutputFile(cycleInfo);
                }
                Debug.WriteLine($"Cycle {cycle:D4} Stability: {isInStableState}");

                // This trains the layer on input pattern
                foreach (double input in inputs)
                {
                    // Learn the input pattern
                    // Output lyrOut is the output of the last module in the layer
                    int[] lyrOut = cortexLayer.Compute(input, true) as int[];

                    // This is a general way to get the SpatialPooler result from the layer
                    int[] activeColumns = cortexLayer.GetResult("sp") as int[];

                    int[] actCols = activeColumns.OrderBy(c => c).ToArray();

                    double similarity = MathHelpers.CalcArraySimilarity(activeColumns, prevActiveCols[input]);

                    Debug.WriteLine(
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
            string trainingTime = $"\nSpatial Pooler Training Time: {stopwatch.ElapsedMilliseconds} ms";
            Console.WriteLine(trainingTime);
            UpdateOutputFile(trainingTime);
            return sp;
        }

        /// <summary>
        ///     Runs the reconstruction experiment by training KNN and HTM classifiers using input values,
        ///     making predictions for each input, and comparing the reconstructed inputs' similarity
        ///     to the original inputs. The reconstruction results are displayed in the console, and a plot is generated.
        /// </summary>
        /// <param name="sp"></param>
        /// <param name="encoder"></param>
        /// <param name="inputValues"></param>
        /// <param name="seedValue"></param>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        private void CompareClassifiers(SpatialPooler sp, EncoderBase encoder, List<double> inputValues,
            int seedValue)
        {
            if (sp == null)
            {
                throw new ArgumentNullException(nameof(sp));
            }

            if (encoder == null)
            {
                throw new ArgumentNullException(nameof(encoder));
            }

            if (inputValues == null || !inputValues.Any())
            {
                throw new ArgumentException("Input values cannot be null or empty", nameof(inputValues));
            }

            // As we are dividing the input set into two parts for training and testing,
            // there could a bias the classifiers toward lower values and make the test set
            // unrepresentative of the full range. Hence, we need to shuffle the list.
            Random random;
            if (seedValue == 0)
            {
                random = new Random();
            }
            else
            {
                random = new Random(seedValue);
            }

            // Shuffle the input List
            inputValues = inputValues.OrderBy(_ => random.Next()).ToList();

            // Split data into training (80%) and testing (20%)
            int splitIdx = (int)(inputValues.Count * 0.8);
            List<double> trainDataSet = inputValues.Take(splitIdx).ToList();
            List<double> testDataSet = inputValues.Skip(splitIdx).ToList();

            KNeighborsClassifier<string, string> knnClassifier = new();
            HtmClassifier<string, string> htmClassifier = new();

            // Clear the models from all the stored sequences
            knnClassifier.ClearState();
            htmClassifier.ClearState();

            Stopwatch stopwatch = Stopwatch.StartNew();

            // Train classifiers on TRAINING DATA
            foreach (double trainData in trainDataSet)
            {
                // Generate SDR for TRAINING DATA using the trained SP
                int[] sdr = encoder.Encode(trainData);
                int[] actCols = sp.Compute(sdr, false);

                // Converting the int[] to Cell[] because we need Cell[] format for learning
                Cell[] cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();

                knnClassifier.Learn(trainData.ToString("F2", CultureInfo.InvariantCulture), cells);
                htmClassifier.Learn(trainData.ToString("F2", CultureInfo.InvariantCulture), cells);
            }

            stopwatch.Stop();

            string classifierTime = $"Classifier Training Time: {stopwatch.ElapsedMilliseconds} ms";
            
            Console.WriteLine("\nClassifier Training Complete");
            Console.WriteLine(classifierTime);
            
            // Write classifier training info to file
            UpdateOutputFile("Classifier Training Complete");
            UpdateOutputFile(classifierTime);
            
            
            // Run the Reconstruction on test data - the data which was not used to train the classifiers
            this.ReconstructInput(testDataSet, encoder, sp, knnClassifier, htmClassifier, inputValues.Max(), "Test");

            // Run the Reconstruction on training data - the data which was used to train the classifiers
            this.ReconstructInput(trainDataSet, encoder, sp, knnClassifier, htmClassifier, inputValues.Max(), "Train");
        }

        /// <summary>
        ///     Runs the reconstruction part of the experiment by generating SDRs for the dataset,
        ///     making predictions using KNN and HTM classifiers, and comparing the reconstructed inputs'
        ///     similarity to the original inputs. The results are stored and plotted for visualization.
        /// </summary>
        /// <param name="dataset">The dataset to be used for reconstruction</param>
        /// <param name="encoder">The encoder used for converting input to SDR</param>
        /// <param name="sp">The trained Spatial Pooler</param>
        /// <param name="knnClassifier">The KNN classifier used for reconstruction</param>
        /// <param name="htmClassifier">The HTM classifier used for reconstruction</param>
        /// <param name="datasetType">The type of dataset ("Train" or "Test")</param>
        private void ReconstructInput(List<double> dataset, EncoderBase encoder, SpatialPooler sp,
            KNeighborsClassifier<string, string> knnClassifier, HtmClassifier<string, string> htmClassifier,
            double max, string datasetType)
        {
            
            Console.WriteLine($"\n----- Start of {datasetType} data reconstruction -----");
            UpdateOutputFile($"\n----- Start of {datasetType} data reconstruction -----");
            
            Results.Clear();
            
            foreach (double data in dataset)
            {
                Console.WriteLine($"\nInput: {data.ToString("F", CultureInfo.InvariantCulture)}");
                UpdateOutputFile($"\nInput: {data.ToString("F", CultureInfo.InvariantCulture)}");
                
                // Generate SDR using the trained SP
                int[] sdr = encoder.Encode(data);
                int[] actCols = sp.Compute(sdr, false);

                // Converting the int[] to Cell[] because we need Cell[] format for reconstruction
                Cell[] cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();

                // Get predictions using the test SDR
                ClassifierResult<string> knnPrediction = knnClassifier.GetPredictedInputValues(cells)[0];
                ClassifierResult<string> htmPrediction = htmClassifier.GetPredictedInputValues(cells)[0];

                // This is done because HTM provides Similarity value between 0 - 100, but we want between 0 - 1
                double htmNormalizedSimilarity = htmPrediction.Similarity / 100;

                double knnPercentageSimilarity = this.CalculatePercentageSimilarity(data,
                    double.Parse(knnPrediction.PredictedInput, CultureInfo.InvariantCulture));
                double htmPercentageSimilarity = this.CalculatePercentageSimilarity(data,
                    double.Parse(htmPrediction.PredictedInput, CultureInfo.InvariantCulture));

                Console.WriteLine($"KNN - Reconstructed Input: {knnPrediction.PredictedInput}");
                Console.WriteLine(
                    $"KNN - Internal Similarity: {knnPrediction.Similarity.ToString("P", CultureInfo.InvariantCulture)}");
                Console.WriteLine(
                    $"KNN - Percentage Similarity: {knnPercentageSimilarity.ToString("P", CultureInfo.InvariantCulture)}");
                Console.WriteLine($"HTM - Reconstructed Input: {htmPrediction.PredictedInput}");
                Console.WriteLine(
                    $"HTM - Internal Similarity: {htmNormalizedSimilarity.ToString("P", CultureInfo.InvariantCulture)}");
                Console.WriteLine(
                    $"HTM - Percentage Similarity: {htmPercentageSimilarity.ToString("P", CultureInfo.InvariantCulture)}");
                
                // Write to file
                UpdateOutputFile($"KNN - Reconstructed Input: {knnPrediction.PredictedInput}");
                UpdateOutputFile($"KNN - Internal Similarity: {knnPrediction.Similarity.ToString("P", CultureInfo.InvariantCulture)}");
                UpdateOutputFile($"KNN - Percentage Similarity: {knnPercentageSimilarity.ToString("P", CultureInfo.InvariantCulture)}");
                UpdateOutputFile($"HTM - Reconstructed Input: {htmPrediction.PredictedInput}");
                UpdateOutputFile($"HTM - Internal Similarity: {htmNormalizedSimilarity.ToString("P", CultureInfo.InvariantCulture)}");
                UpdateOutputFile($"HTM - Percentage Similarity: {htmPercentageSimilarity.ToString("P", CultureInfo.InvariantCulture)}");


                // Add per-input comparison
                if (htmPercentageSimilarity > knnPercentageSimilarity)
                {
                    Console.WriteLine("Based on PercentageSimilarity - HTM performed better for this input");
                    UpdateOutputFile("Based on PercentageSimilarity - HTM performed better for this input");
                }
                else if (htmPercentageSimilarity < knnPercentageSimilarity)
                {
                    Console.WriteLine("Based on PercentageSimilarity - KNN performed better for this input");
                    UpdateOutputFile("Based on PercentageSimilarity - KNN performed better for this input");
                }
                else
                {
                    Console.WriteLine("Based on PercentageSimilarity - Both performed similar for this input");
                    UpdateOutputFile("Based on PercentageSimilarity - Both performed similar for this input");
                }

                // Store results for visualisations
                Results[data] = (
                    double.Parse(knnPrediction.PredictedInput, CultureInfo.InvariantCulture),
                    double.Parse(htmPrediction.PredictedInput, CultureInfo.InvariantCulture),
                    knnPrediction.Similarity,
                    htmNormalizedSimilarity,
                    knnPercentageSimilarity,
                    htmPercentageSimilarity
                );

            }
            
            Console.WriteLine($"\n----- End of {datasetType} data reconstruction -----");
            UpdateOutputFile($"\n----- End of {datasetType} data reconstruction -----");

            // Plot results
            String path = this.PathToSave();
            this.PlotReconstructionResults(Results, max, datasetType, path);
            this.PlotSimilarityResults(Results, max, datasetType, path);
            
            Console.WriteLine($"\nOutput file saved at: {outputFilePath}");
        }
        
        /// <summary>
        ///     Initializes the output file with experiment header
        /// </summary>
        private void InitializeOutputFile()
        {
            string path = PathToSave();
            
            // Get the project root path dynamically
            string saveDir = Path.Combine(path, "Generated_Output");
    
            // Create directory if it doesn't exist
            Directory.CreateDirectory(saveDir);
            
            outputFilePath = Path.Combine(saveDir, "Output.txt");
            
            // Write initial experiment info
            File.WriteAllText(outputFilePath, $"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstructionExperiment)}");
        }
        
        /// <summary>
        ///     Updates the output file with required information
        /// </summary>
        private void UpdateOutputFile(string content)
        {
            File.AppendAllText(outputFilePath, "\n"+content);
        }
        
        /// <summary>
        ///     Calculates the Absolute Percentage Similarity between two given values
        /// </summary>
        /// <param name="value1">First value</param>
        /// <param name="value2">Second value</param>
        /// <returns>Calculates the Percentage Similarity between the given vales and returns the result between 0 -1</returns>
        private double CalculatePercentageSimilarity(double value1, double value2)
        {
            double difference = Math.Abs(value1 - value2);
            double similarity = 1 - (difference / Math.Max(value1, value2));

            return Math.Round(similarity, 2);
        }

        /// <summary>
        ///     Plots the reconstruction results by creating a scatter plot comparing the original input values
        ///     with the reconstructed predictions from both KNN and HTM classifiers.
        /// </summary>
        private void PlotReconstructionResults(Dictionary<double, (double KnnReconstructedInput,
            double HtmReconstructedInput,
            double KnnInternalSimilarity,
            double HtmInternalSimilarity,
            double KnnPercentageSimilarity,
            double HtmPercentageSimilarity)> results, double max, string datasetType, string path)
        {
            Plot knnPlot = new();
            
            var knnScatter = knnPlot.Add.Scatter(results.Keys.ToArray(),
                results.Values.Select(result => result.KnnReconstructedInput).ToArray());

            knnScatter.LegendText = "KNN Predictions";
            knnScatter.LineWidth = 0;
            knnScatter.MarkerSize = 10;
            knnScatter.MarkerColor = Colors.Blue;
            knnScatter.MarkerShape = MarkerShape.FilledCircle;
            
            knnPlot.Axes.SetLimits(0, max+1, 0, max+1);
            knnPlot.Title(datasetType + " - KNN - Reconstruction Predictions");
            knnPlot.XLabel("Input Values");
            knnPlot.YLabel("Predictions");
            this.SavePlot(knnPlot, datasetType + "_KNN_ReconstructionPlot.png", max, path);
            
            Plot htmPlot = new();
            
            var htmScatter = htmPlot.Add.Scatter(results.Keys.ToArray(),
                results.Values.Select(result => result.HtmReconstructedInput).ToArray());

            htmScatter.LegendText = "HTM Predictions";
            htmScatter.LineWidth = 0;
            htmScatter.MarkerSize = 10;
            htmScatter.MarkerColor = Colors.Red;
            htmScatter.MarkerShape = MarkerShape.FilledCircle;
            
            htmPlot.Axes.SetLimits(0, max+1, 0, max+1);
            htmPlot.Title(datasetType + " - HTM - Reconstruction Predictions");
            htmPlot.XLabel("Input Values");
            htmPlot.YLabel("Predictions");
            this.SavePlot(htmPlot, datasetType + "_HTM_ReconstructionPlot.png", max, path);
        }

        /// <summary>
        ///     Plots the similarity results by creating Line plots comparing similarities
        ///     of reconstructed inputs with original inputs from both KNN and HTM classifiers.
        /// </summary>
        private void PlotSimilarityResults(Dictionary<double, (double KnnReconstructedInput,
            double HtmReconstructedInput,
            double KnnInternalSimilarity,
            double HtmInternalSimilarity,
            double KnnPercentageSimilarity,
            double HtmPercentageSimilarity)> results, double max, string datasetType, string path)
        {
            Plot plot = new();

            // Plotting the KNN Similarities
            results.ToList().ForEach(kvp =>
            {
                LinePlot line = plot.Add.Line(
                    kvp.Key - 0.05,
                    x2: kvp.Key - 0.05,
                    y1: 0,
                    y2: kvp.Value.KnnPercentageSimilarity * 100
                );
                line.Color = Colors.Blue;
                line.LineWidth = 3;
            });

            // Plotting the HTM Similarities
            results.ToList().ForEach(kvp =>
            {
                LinePlot line = plot.Add.Line(
                    kvp.Key + 0.05,
                    x2: kvp.Key + 0.05,
                    y1: 0,
                    y2: kvp.Value.HtmPercentageSimilarity * 100
                );
                line.Color = Colors.Red;
                line.LineWidth = 3;
            });

            // Dummy Line to add Legend
            LinePlot knnLegend = plot.Add.Line(0, 0, 0, 0);
            knnLegend.Color = Colors.Blue;
            knnLegend.LegendText = "KNN Similarities";

            // Dummy Line to add Legend
            LinePlot htmLegend = plot.Add.Line(0, 0, 0, 0);
            htmLegend.Color = Colors.Red;
            htmLegend.LegendText = "HTM Similarities";

            plot.Legend.Alignment = Alignment.UpperLeft;
            plot.Title(datasetType + " - Similarity Comparison");
            plot.XLabel("Input Values");
            plot.YLabel("Similarity (%)");
            plot.Axes.SetLimits(0, max+1, 0, 105);

            this.SavePlot(plot, datasetType + "_SimilarityPlot.png", max, path);
        }

        /// <summary>
        ///     Saves the generated plot to the desktop in a cross-platform compatible way.
        ///     The plot is saved as "ScalarInputReconstructionPlot.png" with specified dimensions.
        /// </summary>
        /// <param name="plot"></param>
        /// <param name="fileName"></param>
        private void SavePlot(Plot plot, string fileName, double max, string path)
        {
            const int baseHeight = 600;
            const int minWidth = 600;
            const int maxWidth = 1200;
    
            // Calculate proportional width between 600-1200 based on max input value
            double widthFactor = Math.Clamp((max - 10) / (100 - 10), 0, 1);
            int dynamicWidth = (int)(minWidth + (maxWidth - minWidth) * widthFactor);

            // Get the project root path dynamically
            string saveDir = Path.Combine(path, "Generated_Plots");
    
            // Create directory if it doesn't exist
            Directory.CreateDirectory(saveDir);

            string savePath = Path.Combine(saveDir, fileName);
    
            plot.Save(savePath, dynamicWidth, baseHeight);
            Console.WriteLine($"\nPlot saved at: {savePath}");
            UpdateOutputFile($"Plot saved at: {savePath}");
        }
        
        /// <summary>
        ///     Finds the project root by searching upward for the "neocortexapi-team-untitled" folder
        ///     and then saves the plot.
        /// </summary>
        private string PathToSave()
        {
            DirectoryInfo dir = new (Directory.GetCurrentDirectory());
            while (dir != null)
            {
                if (dir.GetDirectories("neocortexapi-team-untitled").Any())
                {
                    return Path.Combine(dir.FullName, "neocortexapi-team-untitled", 
                        "source", 
                        "Team_Untitled_Files");
                }
                dir = dir.Parent;
            }
    
            // Fallback if not found
            Console.WriteLine("Warning: Project root not found. Using desktop instead.");
            return Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
        }
    }
}