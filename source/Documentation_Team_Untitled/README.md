# ML 24/25-02 # ML Investigate Input reconstruction by using Classifiers
###### _Through out this project we contribute to  implement the Spatial Pooler SDR Reconstruction in NeoCortexAPI_This project implements input reconstruction using  classifiers (KNN and HtmClassifier) to regenerate scalar inputs from Spatial Pooler SDRs

[![N|Logo](https://ddobric.github.io/neocortexapi/images/logo-NeoCortexAPI.svg )](https://ddobric.github.io/neocortexapi/)
In this Documentation we will describe our contribution in this project.

#### Instruction for Running the Project
- Clone the Repository and Run
- You will get the project here
[NeoCortexApi-Team](https://github.com/prnshubn/neocortexapi-team-untitled/tree/master/source/NeoCortexApi.Experiments)

#### Two Existing Classifiers
- **`HtmClassifier.cs`**: Numerical Inputs 
[HtmClassifier.cs](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi/Classifiers/HtmClassifier.cs)
- **`KnnClassifier.cs`**: Image Inputs 
[KnnClassifier.cs](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs)

#### Experiment Code
- **`SpatialPoolerInputReconstructionExperiment.cs`**: The implementation of the Spatial Pooler Input Reconstruction Experiment can be found here: 
[SpatialPoolerInputReconstructionExperiment.cs](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs)
- **`SpatialPoolerInputReconstructionExperimentTests.cs`**: The Unit Test for the experiment can be found here:
[SpatialPoolerInputReconstructionExperimentTests.cs](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs)

###### Image Input sets are already uploaded here
- [Documentation](https://github.com/prnshubn/neocortexapi-team-untitled/tree/master/source/Documentation_Team_Untitled)

###### All the output will be saved here
- [neocortexapi_team.yet to make it *******]

## Introduction
This project explores the concept of input reconstruction using classifiers within HTM (Hierarchical Temporal Memory). The goal is to analyze how well HTM and KNN classifiers can reconstruct the original input based on Sparse Distributed Representations (SDRs). This investigation is inspired by the SpatialLearning experiment and extends it by incorporating input reconstruction.

# Methodology
The Spatial Pooler Input Reconstruction Experiment investigates how well classifiers can reconstruct original input values from Sparse Distributed Representations (SDRs). The experiment follows a structured pipeline starting with data encoding, SDR generation, classifier training, and input reconstruction. First, numerical input values are encoded using a Scalar Encoder, which transforms continuous values into binary representations. These encoded values are then passed through the Spatial Pooler (SP), which learns stable patterns and generates SDRs. The Spatial Pooler applies synaptic learning rules to form a structured representation of input data, which serves as the basis for reconstruction.

Once the SDRs are generated, two classifiers—HTM Classifier and KNN Classifier—are trained to associate SDRs with their corresponding input values. The HTM Classifier learns temporal sequences, meaning it adapts over time to improve predictions, whereas the KNN Classifier memorizes SDRs and reconstructs inputs based on similarity to previously seen patterns. The classifiers are trained using an 80-20 split on the dataset, with 80% of the input values used for learning and the remaining 20% used for testing. During inference, classifiers attempt to predict the original input from unseen SDRs. The reconstructed values are then compared with actual inputs using similarity metrics.

To evaluate classifier performance, the results are visualized through similarity graphs and heatmaps. These visualizations show the accuracy of HTM and KNN predictions in reconstructing inputs from SDRs. By comparing their performance, we gain insights into which classifier is more effective for input reconstruction within HTM-based systems. The findings contribute to a better understanding of classification-based reconstruction techniques and their potential for enhancing Sparse Distributed Representations in machine learning applications.

**Fig: Methodology Flowchart**
![Methodology Flowchart](******** yet to add it ********)

## Training the HTM and KNN Classifiers

After SDRs are generated, the next step is training two different classifiers to learn and predict input values.

- **HTM Classifier:** This classifier learns temporal patterns over time. It associates SDRs with input values and refines its predictions as more data is observed.

 - **KNN Classifier (K-Nearest Neighbors):** This classifier memorizes SDRs and predicts new inputs by comparing them with previously stored representations, selecting the closest match.

- **Training Process:** 
* The dataset is split into training (80%) and testing (20%) subsets.
* For each training value, an SDR is generated using the trained Spatial Pooler.
* These SDRs are passed to both classifiers using the Learn() method:
 - **cls.Learn(key, actCells.ToArray());** 
* During training, classifiers store SDR-input mappings to be used later during reconstruction.
* The trained classifiers are tested on unseen data, where they take an SDR as input and attempt to reconstruct the original input value.


## Reconstruct() Method:

The ReconstructionExperiment method runs the input reconstruction experiment by setting up the required components, training the Spatial Pooler (SP), and performing input reconstruction using HTM and KNN classifiers. It starts by defining HTM configurations and Scalar Encoder settings to convert input values into Sparse Distributed Representations (SDRs). The method then trains the Spatial Pooler to generate stable SDRs and passes them to the classifiers for learning. Once trained, the classifiers attempt to reconstruct the original input values from SDRs. Finally, the method evaluates reconstruction accuracy and visualizes results for comparison.
``` csharp
 public Dictionary<int, double> Reconstruct(int[] activeMiniColumns)
 {
     if (activeMiniColumns == null)
     {
         throw new ArgumentNullException(nameof(activeMiniColumns));
     }

     var cols = connections.GetColumnList(activeMiniColumns);

     Dictionary<int, double> permancences = new Dictionary<int, double>();

    
     foreach (var col in cols)
     {
         col.ProximalDendrite.Synapses.ForEach(s =>
         {
             double currPerm = 0.0;

             
             if (permancences.TryGetValue(s.InputIndex, out currPerm))
             {
               
                 permancences[s.InputIndex] = s.Permanence + currPerm;
             }
             else
             {
              
                 permancences[s.InputIndex] = s.Permanence;
             }
         });
     }

     return permancences;
 }
```
[Reconstruction in SP](https://github.com/BidhanPaul/neocortexapi_team.bji/blob/master/source/NeoCortexApi/SpatialPooler.cs#L1442) - Lines (1442 to 1482)

#### Reconstruct() Workflow:
- **Input Validation:** Thorough validation checks, throwing an `ArgumentNullException` if the input array of active mini-columns is null.
   
- **Column Retrieval:** Retrieve the list of columns associated with the active mini-columns from the connections.
   
- **Reconstruction Process:** Iterate through each column, accessing the synapses in its proximal dendrite.
   
- **Permanence Accumulation:** For each synapse, accumulate the permanence values for each input index in the reconstructed input dictionary.
   
- **Dictionary Update:** Update the reconstructed input dictionary, considering whether the input index already exists or needs to be added as a new key-value pair.
   
- **Result Return:** The method concludes by returning the reconstructed input as a dictionary, mapping input indices to their associated permanences.

# Running Reconstruct Method for Numerical Inputs
```csharp
     public void ReconstructionExperiment(double max, int seedValue=0)
        {
            if (max < 10) throw new ArgumentException("max must be 10 or greater", nameof(max));
            
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstructionExperiment)}");

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
            List<double> inputValues = Enumerable.Range(1, (int)max).Select(i => (double)i).ToList();

            // Train the Spatial Pooler
            SpatialPooler sp = TrainSpatialPooler(cfg, encoder, inputValues);

            // Perform Reconstruction Experiment
            ClassifierPart(sp, encoder, inputValues, seedValue);
        }

```
[Running Reconstruct Method ](git link with line number to be based ) - Lines (243 to 329)
# The ReconstructionPart method generates SDRs for input data, predicts reconstructed values using HTM and KNN classifiers, compares them with original inputs using similarity metrics, and visualizes the results.
```csharp
    private void ReconstructionPart(List<double> dataset, EncoderBase encoder, SpatialPooler sp,
            KNeighborsClassifier<string, string> knnClassifier, HtmClassifier<string, string> htmClassifier,
            double max, string datasetType)
        {
            // Initialize output file
            InitializeOutputFile();
            Results.Clear();
            foreach (double data in dataset)
            {
                Console.WriteLine($"\nInput: {data.ToString("F", CultureInfo.InvariantCulture)}");
                // Write to file
                File.AppendAllText(outputFilePath, $"\nInput: {data.ToString("F", CultureInfo.InvariantCulture)}\n");
                
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
                File.AppendAllText(outputFilePath, $"KNN - Reconstructed Input: {knnPrediction.PredictedInput}\n");
                File.AppendAllText(outputFilePath, $"KNN - Internal Similarity: {knnPrediction.Similarity.ToString("P", CultureInfo.InvariantCulture)}\n");
                File.AppendAllText(outputFilePath, $"KNN - Percentage Similarity: {knnPercentageSimilarity.ToString("P", CultureInfo.InvariantCulture)}\n");
                File.AppendAllText(outputFilePath, $"HTM - Reconstructed Input: {htmPrediction.PredictedInput}\n");
                File.AppendAllText(outputFilePath, $"HTM - Internal Similarity: {htmNormalizedSimilarity.ToString("P", CultureInfo.InvariantCulture)}\n");
                File.AppendAllText(outputFilePath, $"HTM - Percentage Similarity: {htmPercentageSimilarity.ToString("P", CultureInfo.InvariantCulture)}\n");


                // Add per-input comparison
                if (htmPercentageSimilarity > knnPercentageSimilarity)
                {
                    Console.WriteLine("Based on PercentageSimilarity - HTM performed better for this input");
                }
                else if (htmPercentageSimilarity < knnPercentageSimilarity)
                {
                    Console.WriteLine("Based on PercentageSimilarity - KNN performed better for this input");
                }
                else
                {
                    Console.WriteLine("Based on PercentageSimilarity - Both performed similar for this input");
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

            // Plot results
            String path = PathToSavePlots();
            this.PlotReconstructionResults(Results, max, datasetType, path);
            this.PlotSimilarityResults(Results, max, datasetType, path);
        }
```
[Not yet done]
[Running Reconstruct Method for Image Data](https://github.com/BidhanPaul/neocortexapi_team.bji/blob/master/source/Samples/NeoCortexApiSample/ImageBinarizerSpatialPattern.cs#L157-L251) - Lines (157 to 251)
### Implementation Details for both inputs Type():
###### Reconstruct permanence values from active columns using the Spatial Pooler
reconstructedPermanence = sp.Reconstruct(actCols)

###### Set the maximum input index
maxInput = lengthofinputvectors
###### Note: According to the size of Encoded Inputs (200 bits for numerical inputs)
###### Note: According to the size of Encoded Inputs (for image  inputs the output of encoded bits depends on the multiplication of height and width of the image )

###### Initialize a dictionary to store all input indices and their associated permanence probabilities
allPermanenceDictionary = new Dictionary<int, double>()

###### Storing Permanence in the dictionary with reconstructed permanence values
for each key-value pair (inputIndex, probability) in reconstructedPermanence
    allPermanenceDictionary[inputIndex] = probability

###### Handling Inactive Columns Permanence by assigning a default permanence value of 0.0
for inputIndex from 0 to maxInput
    if inputIndex not in reconstructedPermanence
        allPermanenceDictionary[inputIndex] = 0.0

###### Note: reconstructedPermanence is a subset contributing to the construction of allPermanenceDictionary

## Getting Data For Visualizing Results
```csharp
    //Getting The Heatmap data from Reconstructed Permanence as Double
     List<List<double>> heatmapData = new List<List<double>>();
     //Getting The encoded bits data
     List<int[]> encodedInputs = new List<int[]>();
    //Getting The Nomalize Permanence as int
     List<int[]> normalizedPermanence = new List<int[]>();
```
## Normalizing the Permanence Values for Numeric Input Data
```csharp
   //We used the Threshold values 8.3 to normalize the permanence
   var ThresholdValue = 8.3;
   //calling the function ThresholdingProbabilities from Helpers.cs
List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue);
  //Converting normalizedPermanence into Array
normalizedPermanence.Add(normalizePermanenceList.ToArray());
```

###### Note: The Threshold Value 8.3 has the ability to Normalize The permanence with the most similiraty with Encoded Inputs. We tried multiple Threshold values and Debugged the output and compared with encoded inputs.
## Normalizing the Permanence Values for Image Input Data
```csharp
   //Normalizing Permanence Threshold
var ThresholdValue = 30.5;

// Normalize permanences (0 and 1) based on the threshold value and convert them to a list of integers.
List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue);

//Collecting Normalized Permanence List for Visualizing
normalizedPermanence.Add(normalizePermanenceList.ToArray());
```
###### Note: The Threshold Value 30.5 has the ability to Normalize The permanence with the most similiraty with Encoded Inputs. We tried multiple Threshold values and Debugged the output and compared with encoded inputs.
## Normalizing Function (ThresholdingProbabilities)
```csharp
  public static List<int> ThresholdingProbabilities(IEnumerable<double> values, double threshold)
{
    if (values == null)
    {
        return null;
    }

    List<int> resultList = new List<int>();

    foreach (var numericValue in values)
    {
        int thresholdedValue = (numericValue >= threshold) ? 1 : 0;

        resultList.Add(thresholdedValue);
    }

    return resultList;
}
```
Here is the Function
[Helpers.cs
](https://github.com/BidhanPaul/neocortexapi_team.bji/blob/master/source/NeoCortexApi/Helpers.cs#L620-L637) - Lines (620 to 637)
## Generate1DHeatmaps Function for both Input types
```csharp
   private void Generate1DHeatmaps(List<List<double>> heatmapData, List<int[]> encodedData, List<int[]> normalizedPermanence)
{
    int i = 1;

    foreach (var values in heatmapData)
    {
       
        string folderPath = Path.Combine(Environment.CurrentDirectory, "1DHeatMap");

        if (!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
        }

        string filePath = Path.Combine(folderPath, $"heatmap_{i}.png");
        Debug.WriteLine($"FilePath: {filePath}");
      
        double[] array1D = values.ToArray();
       
        NeoCortexUtils.Draw1DHeatmap(new List<double[]>() { array1D }, new List<int[]>() { normalizedPermanence[i - 1] }, new List<int[]>() { normalizedPermanence[i - 1] }, filePath, 200, 8, 9, 4, 0, 30);

        Debug.WriteLine("Heatmap generated and saved successfully.");
        i++;
    }
}
```
[GenarateHeatmap Function](https://github.com/BidhanPaul/neocortexapi_team.bji/blob/master/source/Samples/NeoCortexApiSample/SpatialPatternLearning.cs#L311) - Lines (311 to 341)
###### Parameters
- `heatmapData`: A list of lists containing probability data for heatmap generation.
- `EncodedData`: A list of lists containing Encoded input Data.
- `normalizedPermanence`: A list of arrays containing normalized permanence values corresponding to the heatmap data.

###### Implementation
- The function iterates through each set of probabilities in `heatmapData`.

###### Folder and File Management:
- A folder path is defined based on the current environment, specifically within the "1DHeatMap" directory.
- If the folder does not exist, it is created to ensure proper organization.
- The file path for each heatmap is constructed dynamically using the folder path and an index (`i`).

###### 1D Array Conversion:
- The probabilities list is converted into a 1D array (`array1D`) using the `ToArray` method for compatibility with the subsequent heatmap generation process.

###### Heatmap Generation:
- The function calls a modified version of the `Draw1DHeatmapWithSeparatedValues` function from the `NeoCortexUtils` class.
- This function handles the visualization process, considering the 1D array of probabilities (`array1D`) and the corresponding normalized permanence values.
- Key parameters, such as file path, dimensions, and visualization settings, are dynamically adjusted for each iteration.
###### Note: Heatmap Generation Parameters

- **`filePath`**: File path where the heatmap image will be saved.
- **`width`**: 200 (pixels) - Width of the heatmap image.
- **`height`**: 8 (pixels) - Height of the heatmap image.
- **`mostHeatedColor`**: 9 - Value for the most heated color (Red represents 1).
- **`medianValue`**: 4 - Median value for color interpolation.
  - Example: Greater than 4 represents orange to red, less than 4 represents green to yellow.
- **`coldestColor`**: 0 - Coldest color representing 0 bits.
- **`enlargementFactor`**: 30 - Enlargement factor used to magnify the image for better visualization.


###### Debugging Information:
- Debugging information, including file paths and successful heatmap generation confirmation, is output using `Debug.WriteLine`.
## Calling HeatMap Function
```csharp
//Calling the HeatMap Function in RunRestructuringExperiment with two Perameters
Generate1DHeatmaps(heatmapData, normalizedPermanence);
```

## Combined Visualization: Heatmaps and int[] Sequences
We Applied this Function to Draw1DHeatmap
Click Below for More Details 
[Draw1dHeatmap](https://github.com/BidhanPaul/neocortexapi_team.bji/blob/master/source/NeoCortexUtils/NeoCortexUtils.cs#L222-L351) - Lines (222 to 351)
**Outcomes:**
- HeatMap Image for all inputs as Image Visualization.
- Encoded Inputs as int []
- Reconstruced Input as int [] (Normalized Permanence)
- Combined Image.


**Results Example:**
**Fig: Final Outcome**
![Final Outcome](https://raw.githubusercontent.com/BidhanPaul/neocortexapi_team.bji/master/source/Docomentation%20neocortexapi_Team.bji/Final_Outcome_Example_heatmap_1.png)
## Similarity Calculation Using Jaccard Similarity Coefficient
```csharp
   public static double JaccardSimilarityofBinaryArrays(int[] arr1, int[] arr2)
{
    if (arr1.Length != arr2.Length)
    {
        throw new ArgumentException("Arrays must have the same length.");
    }

    int intersectionCount = 0;
    int unionCount = 0;

    for (int i = 0; i < arr1.Length; i++)
    {
        if (arr1[i] == 1 && arr2[i] == 1)
        {
            intersectionCount++;
        }
        if (arr1[i] == 1 || arr2[i] == 1)
        {
            unionCount++;
        }
    }

    return (double)intersectionCount / unionCount;
}
```
Here is the Function
[MathHelpers.cs](https://github.com/BidhanPaul/neocortexapi_team.bji/blob/master/source/NeoCortexApi/Utility/MathHelpers.cs#L182-L205) - Lines (182 to 205)

## Genarate Similarity Graph
- Note The calling Similarity Function is same like Drwaing Heatmap

We Applied this Function to DrawCombinedSimilarityplot
Click Below for More Details 
[DrawCombinedSimilarityPlot](https://github.com/BidhanPaul/neocortexapi_team.bji/blob/master/source/NeoCortexUtils/NeoCortexUtils.cs#L428-L536) - Lines (428 to 536)
**Outcomes:**
- Bar graphs of similarity for each inputs


**Results Example:**
**Fig: Final Outcome for Image  input**
![Final Outcome](https://raw.githubusercontent.com/BidhanPaul/neocortexapi_team.bji/master/source/Docomentation%20neocortexapi_Team.bji/FinalOutcomeExamplecombined_similarity_plot_Image_Inputs.png)
## Spatial Pooler Reconstruction Tests
## UnitTest of SdrReconstructionTests
We Tested the SdrReconstruction.cs with 9 Test cases and all Passed
This document provides an overview of the unit tests present in the project.
[SdrReconstructionTests](https://github.com/BidhanPaul/neocortexapi_team.bji/blob/master/source/UnitTestsProject/SdrReconstructionTests.cs)
### Reconstruct_ValidInput_ReturnsResult
- **Test Category:** SpatialPoolerReconstruction
- **Description:** Verifies whether the `Reconstruct` method in the `SPSdrReconstructor` class behaves correctly under valid input conditions. It ensures that the method returns a dictionary containing keys for all provided active mini-columns, with corresponding permanence values. Additionally, it confirms that the method properly handles the case where a key is not present in the dictionary.

### Reconstruct_NullInput_ThrowsArgumentNullException
- **Test Category:** ReconstructionExceptionHandling
- **Description:** Verifies that the `Reconstruct` method in the `SPSdrReconstructor` class throws an `ArgumentNullException` when invoked with a null input parameter.

### Reconstruct_EmptyInput_ReturnsEmptyResult
- **Test Category:** ReconstructionEdgeCases
- **Description:** Tests whether the `Reconstruct` method returns an empty dictionary when provided with an empty input.

## Reconstruction Tests for Various Scenarios

### Reconstruct_AllPositivePermanences_ReturnsExpectedValues
- **Test Category:** ReconstructionAllPositiveValues
- **Description:** Checks if the `Reconstruct` method in the `SPSdrReconstructor` class handles a scenario where all mini-column indices provided as input are positive integers and returns permanence values that are non-negative.

### Reconstruct_AddsKeyIfNotExists
- **Test Category:** ReconstructionAddingKeyIfNotExist
- **Description:** Ensures that the `Reconstruct` method adds a key to the dictionary if it doesn't already exist.

### Reconstruct_ReturnsValidDictionary
- **Test Category:** ReconstructionReturnsKvP
- **Description:** Validates whether the `Reconstruct` method returns a valid dictionary containing integer keys and double values.

### Reconstruct_NegativePermanences_ReturnsFalse
- **Test Category:** ReconstructedNegativePermanenceRetunsFalse
- **Description:** Tests the behavior of the `Reconstruct` method when encountering negative permanences and asserts that no negative permanences should be present in the reconstructed values.

### Reconstruct_AtLeastOneNegativePermanence_ReturnsFalse
- **Test Category:** ReconstructedNegativePermanenceRetunsFalse
- **Description:** Validates the behavior of the `Reconstruct` method when at least one permanence value is negative.

### Reconstruct_InvalidDictionary_ReturnsFalse
- **Test Category:** DataIntegrityValidation
- **Description:** Verifies if the `Reconstruct` method returns a valid dictionary by checking specific criteria such as NaN values and keys less than 0.

### IsDictionaryInvalid with Not a Number
- **Test Category:** DictionaryValidityTests
- **Description:** Determines whether a dictionary is considered invalid based on specific criteria like null reference, NaN values, and keys less than 0.
