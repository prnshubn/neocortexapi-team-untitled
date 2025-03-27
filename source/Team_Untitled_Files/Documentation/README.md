# ML 24/25-02 
# Investigate Input Reconstruction by using Classifiers
#### Through this project we contribute to implement the input reconstruction using [KNN Classifier](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs) & [HTM Classifier](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi/Classifiers/HtmClassifier.cs) to regenerate scalar inputs back from SDRs.

[![N|Logo](https://ddobric.github.io/neocortexapi/images/logo-NeoCortexAPI.svg )](https://ddobric.github.io/neocortexapi/)

Here we will describe our contribution to this project.

#### Instruction for running the experiment
- Clone the Repository
- Build the Project
- Run the ExperimentRunner inside [SpatialPoolerInputReconstructionExperiment](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs)

#### Our Code Contributions
- [SpatialPoolerInputReconstructionExperiment](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs): The implementation of the Spatial Pooler Input Reconstruction Experiment.

- [SpatialPoolerInputReconstructionExperimentTests](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs): The Unit Tests for the experiment.

- [Documentation](https://github.com/prnshubn/neocortexapi-team-untitled/tree/master/source/Team_Untitled_Files/Documentation)

### Project Folder Structure
```bash
neocortexapi-team-untitled
│
├── source
│   │
│   ├── NeoCortexApi.Experiments/
│   │   ├── SpatialPoolerInputReconstructionExperiment.cs
│   │
│   ├── UnitTestsProject/
│   │   │── SpatialPoolerInputReconstructionExperimentTests.cs
│   │   
│   ├── Team_Untitled_Files/
│   │   │── Media_Release_Forms/
│   │   │   ├── # Contains the media release forms of the three contributors
│   │   │
│   │   ├── Documentation/
│   │   │   ├── README.md
│   │   │   ├── ML24-25_02_Investigate_Input_Reconstruction_by_using_Classifiers_Team_Untitled-Presentation.pptx
│   │   │   ├── ML24-25_02_Investigate_Input_Reconstruction_by_using_Classifiers_Team_Untitled-Paper.pdf
│   │   │   ├── ML24-25_02_Investigate_Input_Reconstruction_by_using_Classifiers_Team_Untitled-Video.mp4
│   │   │   ├── ML24-25_02_Investigate_Input_Reconstruction_by_using_Classifiers_Team_Untitled-Paper.docx
│   │   │   ├── Flowchart_Investigate_Input_Reconstruction_by_using_Classifiers_Team_Untitled.png
│   │   │
│   │   ├── Generated_Output/
│   │   │   ├── # This folder stores the generated Output.txt file everytime when the experiment runs
│   │   │
│   │   ├── Generated_Plots/
│   │   │   ├── # This folder stores the generated plots everytime when the experiment runs 
│   │   │
│   │   ├── Result_Case_1/
│   │   │   ├── # Contains generated plots and Output.txt file for inputs 1-20
│   │   │
│   │   ├── Result_Case_2/
│   │   │   ├── # Contains generated plots and Output.txt file for inputs 1-50
│   │   │
│   │   ├── Result_Case_3/
│   │   │   ├── # Contains generated plots and Output.txt file for inputs 1-100
│   │   │

```


## Introduction
This experiment investigates the concept of input reconstruction using classifiers - HTM & KNN. The goal is to analyze how well HTM and KNN can reconstruct the original input based on Sparse Distributed Representations (SDRs) stabilized by Spatial Pooler (SP). The Spatial Learning experiment inspires this investigation and extends it by incorporating input reconstruction.

## Methodology Flowchart

<p align="left">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Team_Untitled_Files/Documentation/Flowchart_Investigate_Input_Reconstruction_by_using_Classifiers_Team_Untitled.png"> </p>

# Methodology
The experiment follows a structured pipeline starting with data encoding, SDR generation, classifier training, and input reconstruction. First, numerical input values are encoded using a Scalar Encoder, which transforms continuous values into binary representations. These encoded values are then passed through the Spatial Pooler (SP), which learns stable patterns and generates stable and robust SDRs. The Spatial Pooler applies synaptic learning rules to form a structured representation of input data, which serves as the basis for reconstruction.

Once the SDRs are generated from the Spatial Pooler, both the classifiers — HTM and KNN are trained to associate SDRs with their corresponding input values. The HTM Classifier learns temporal sequences, meaning it adapts over time to improve predictions, whereas the KNN Classifier memorizes SDRs and reconstructs inputs based on similarity to previously seen patterns. The classifiers are trained only on 80% of the input values, and the remaining 20% was used for testing. Before splitting, we randomized the input list so that there is no bias in the classifiers towards the lower values. During inference, classifiers attempt to predict the original input from the test SDRs. The reconstructed values are then compared with actual inputs using similarity metrics.

To evaluate classifier performance, the results are visualized through similarity graphs. These visualizations show the accuracy of HTM and KNN predictions in reconstructing inputs from SDRs. By comparing their performance, we gain insights into which classifier is more effective for input reconstruction. The findings contribute to a better understanding of classification-based reconstruction techniques and their potential for enhancing Sparse Distributed Representations in machine learning applications.

## Difference between the classifiers

- **HTM Classifier (Hierarchical Temporal Memory):** This classifier learns temporal patterns over time. It associates SDRs with input values and refines its predictions as more data is observed.

- **KNN Classifier (K-Nearest Neighbors):** This classifier memorizes SDRs and predicts new inputs by comparing them with previously stored representations, selecting the closest match.

## Training & Reconstruction Process:
* The input dataset is first randomized.
* The randomized dataset is then split into training (80%) and testing (20%) subsets.
* For each training value, an SDR is generated using the trained Spatial Pooler.
* During training, classifiers store SDR-input mappings to be used later during reconstruction.
* The trained classifiers are tested on unseen data, where they take an SDR as input and attempt to reconstruct the original input value.

* For example we have 20 scalar inputs (1 to 20).
  * After the SP has been trained, we will first randomize the ordering of the input list. That means the list will still contain 1 to 20 but in no specific order.
  * Now we split the input into two subsets. So the training subset will contain 80% of the data that is 16 scalar values and the rest 20% which is 4 scalar values will be used to test the reconstruction.
  * We now encode the training subset to generate SDRs and then use the already trained Spatial Pooler to train the classifiers.
  * Now, we encode the testing subset to generate SDRs and from these SDRs, the classifier will try to reconstruct the original input.

## Code Explanation:
### Method [ReconstructionExperiment](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs#L82-L135) - Lines (82 to 135)
The ReconstructionExperiment() method runs the input reconstruction experiment by setting up the required components, and configurations, training the Spatial Pooler (SP), and performing input reconstruction using HTM and KNN classifiers. It starts by defining HTM configurations and Scalar Encoder settings to convert input values into Sparse Distributed Representations (SDRs). The method then calls another method with the defined configurations to train the Spatial Pooler to generate stable SDRs and passes them to the classifiers for learning. Once trained, the classifiers attempt to reconstruct the original input values from SDRs. Finally, reconstruction similarities and visualization results are compared.
``` csharp
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
```

### Method [TrainSpatialPooler](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs#L146-L234) - Lines (146 to 234)
Trains the Spatial Pooler to generate stable Sparse Distributed Representations (SDRs) for inputs.
```csharp
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
```

### Method [CompareClassifier](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs#L247-L326) - Lines (247 to 326)
Trains HTM and KNN classifiers using SDRs, then reconstructs input values for evaluation.

```csharp
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
```

### Method [ReconstructInput](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs#L339-L431) - Lines (339 to 431)
This part reconstructs original data from SDRs using KNN and HTM classifiers, measures their accuracy by comparing reconstructed values to the original inputs, and visualizes the results. It highlights which classifier (KNN or HTM) performs better for each input based on similarity scores.

```csharp
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
```

### Method [CalculatePercentageSimilarity](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs#L446-L472) - Lines (446 to 472)
Calculates the Absolute Percentage Similarity between two values, returning a result between 0 and 1, where 1 means identical values and 0 means maximum difference.

```csharp
        private double CalculatePercentageSimilarity(double value1, double value2)
        {
            double difference = Math.Abs(value1 - value2);
            double similarity = 1 - (difference / Math.Max(value1, value2));

            return Math.Round(similarity, 2);
        }
```
## Result Analysis

### [Case 1: Inputs 1 - 20](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_1/Results_Case_1.xlsx)

As we discussed before, we divide the input set into two subsets with 80:20 split. As the number of inputs is 20, the 80:20 split makes number of training data 16 and testing data 4. The below set of graphs represent the 20% data which was purely used for testing and was unseen to the classifiers. Hence the reconstructions are not perfect. We can also notice that the HTM Classifier performed better than the KNN Classifier.

<p align="center">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_1/Test_HTM_ReconstructionPlot.png" width="33%">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_1/Test_SimilarityPlot.png" width="33%">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_1/Test_KNN_ReconstructionPlot.png" width="33%">
</p>

The below set of data is the rest 80% which was used to train the classifiers. Here we can notice that the reconstruction is perfect with no deviation and both the classifiers performed with 100% similarity.

<p align="center">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_1/Train_HTM_ReconstructionPlot.png" width="33%">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_1/Train_SimilarityPlot.png" width="33%">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_1/Train_KNN_ReconstructionPlot.png" width="33%">
</p>

### [Case 2: Inputs 1 - 50](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_2/Results_Case_2.xlsx)

For the second case, we took 50 inputs. With the 80:20 split, the number of training data is 40 and testing data is 10. The below set of graphs represent the 20% data which was purely used for testing and was unseen to the classifiers. Hence the reconstructions are not perfect. Here also the HTM Classifier performed much better than the KNN Classifier.

<p align="center">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_2/Test_HTM_ReconstructionPlot.png" width="33%">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_2/Test_SimilarityPlot.png" width="33%">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_2/Test_KNN_ReconstructionPlot.png" width="33%">
</p>

The below set of data is the rest 80% which was used to train the classifiers. In this case also we notice that the reconstruction is perfect with no deviation and both the classifiers performed perfectly.

<p align="center">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_2/Train_HTM_ReconstructionPlot.png" width="33%">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_2/Train_SimilarityPlot.png" width="33%">
  <img src="https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/Documentation_Team_Untitled/Result_Case_2/Train_KNN_ReconstructionPlot.png" width="33%">
</p>

## Unit Tests - [SpatialPoolerInputReconstructionExperimentTests](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs)
We tested with 5 test cases, and all passed successfully. These tests validate the correct execution of the Spatial Pooler training, input reconstruction, and classifier accuracy.

### [Test_Experiment_Completes_Without_Exception](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs#L40-L44)
- **Test Category:** SpatialPoolerReconstruction
- **Description:** Verifies whether the ReconstructionExperiment() method runs without throwing errors. Ensures that all components (Scalar Encoder, Spatial Pooler, HTM & KNN Classifiers) initialize and execute correctly.

### [Test_Experiment_With_Improper_Max_Value](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs#L54-L61)
- **Test Category:** ReconstructionExceptionHandling
- **Description:** Ensures that the ReconstructionExperiment() method throws an ArgumentException when an invalid max value (less than 10) is provided.

### [Test_SpatialPoolerTraining_ReachesStableState](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs#L71-L89)
- **Test Category:** SpatialPoolerStability
- **Description:** Checks whether the Spatial Pooler reaches a stable state during training. Captures the console output and verifies that "STABLE STATE REACHED" appears, confirming that the Spatial Pooler has learned stable SDRs.

### [Test_Reconstruction_ProducesPredictions](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs#L99-L119)
- **Test Category:** ClassifierPrediction
- **Description:** Verifies that the HTM and KNN classifiers successfully predict reconstructed inputs. Ensures that the console output contains predictions from both classifiers and similarity percentages.

### [Test_ReconstructionPart_Results_Have_Valid_Similarity](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs#L128-L138)
- **Test Category:** ReconstructionAccuracy
- **Description:** Validates that similarity scores between reconstructed and actual inputs fall within the valid range (0% - 100%). Ensures that both HTM and KNN classifiers return meaningful predictions.



