using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Experiments;
using System;
using System.IO;

namespace UnitTestsProject
{
    /// <summary>
    /// <see href="https://github.com/prnshubn/neocortexapi-team-untitled">Project Link</see><br/>
    /// <see href="https://github.com/prnshubn/neocortexapi-team-untitled/tree/master/source/Documentation_Team_Untitled">Documentation</see><br/>
    /// Unit tests for the SpatialPoolerInputReconstructionExperiment class.
    /// These tests validate the correct execution of the experiment, the training of the Spatial Pooler,
    /// and the input reconstruction process using KNN and HTM classifiers.
    /// </summary><br/>
    /// <para>
    /// University: Frankfurt University of Applied Sciences<br/>
    /// Degree: Master's in Information Technology<br/>
    /// Year: 2024-2025<br/>
    /// Team: Untitled<br/>
    /// Contributors:
    /// <see href="https://github.com/prnshubn">Priyanshu Bandyopadhyay</see>,  
    /// <see href="https://github.com/Hanumanthumanoj01">Manoj Hanumanthu</see>,  
    /// <see href="https://github.com/Akshay-Gudekar">Akshay Gudekar</see>
    /// </para>
    [TestClass]
    public class SpatialPoolerInputReconstructionExperimentTests
    {
        private const string RECONSTRUCTION_PLOT_FILE_NAME = "ReconstructionPlot.png";
        private const string SIMILARITY_PLOT_FILE_NAME = "SimilarityPlot.png";

        /// <summary>
        /// Tests that the RunExperiment method executes without throwing any exceptions.
        /// This is a basic smoke test to ensure the experiment runs to completion.
        /// </summary>
        [TestMethod]
        [Priority(1)]
        [TestCategory("Experiment")]
        public void Test_RunExperiment_CompletesWithoutException()
        {
            SpatialPoolerInputReconstructionExperiment experiment = new();
            experiment.RunExperiment(10, 0);
        }
        
        /// <summary>
        /// Tests that the reconstruction and percentage similarities are according to the predefined results.
        /// </summary>
        [TestMethod]
        [Priority(2)]
        [TestCategory("Experiment")]
        public void Test_PercentageSimilarity_For_Known_values()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();

            // Act
            experiment.RunExperiment(20, 42);

            // Assert
            // TODO: Add assertions of known values
        }

        /// <summary>
        /// Tests that the Spatial Pooler reaches a stable state during training.
        /// This test captures the console output and checks for the "STABLE STATE REACHED" message.
        /// </summary>
        [TestMethod]
        [TestCategory("Experiment")]
        public void Test_SpatialPoolerTraining_ReachesStableState()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();
            var originalConsoleOut = Console.Out;
            var consoleOutput = new StringWriter();

            Console.SetOut(consoleOutput);

            // Act
            experiment.RunExperiment(10, 0);

            // Reset console output
            Console.SetOut(originalConsoleOut);
            string output = consoleOutput.ToString();

            // Assert
            Assert.IsTrue(output.Contains("STABLE STATE REACHED"), "Spatial Pooler did not reach stable state.");
        }

        /// <summary>
        /// Tests that the input reconstruction phase produces valid predictions and similarity metrics.
        /// This test captures the console output and checks for reconstruction results.
        /// </summary>
        [TestMethod]
        [TestCategory("Experiment")]
        public void Test_Reconstruction_ProducesPredictions()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();
            var originalConsoleOut = Console.Out;
            var consoleOutput = new StringWriter();

            Console.SetOut(consoleOutput);

            // Act
            experiment.RunExperiment(10, 0);

            // Reset console output
            Console.SetOut(originalConsoleOut);
            string output = consoleOutput.ToString();

            // Assert
            Assert.IsTrue(output.Contains("KNN - Reconstructed Input"), "KNN predictions not found in output.");
            Assert.IsTrue(output.Contains("HTM - Reconstructed Input"), "HTM predictions not found in output.");
            Assert.IsTrue(output.Contains("Percentage Similarity"), "Similarity metrics not found in output.");
        }

        /// <summary>
        /// Tests that the reconstruction and similarity plots are generated and saved to the desktop.
        /// This test checks for the existence of the output files (environment-dependent).
        /// </summary>
        [TestMethod]
        [TestCategory("Experiment")]
        public void Test_PlotsGenerated()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();
            string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string reconstructionPlotPath = Path.Combine(desktopPath, RECONSTRUCTION_PLOT_FILE_NAME);
            string similarityPlotPath = Path.Combine(desktopPath, SIMILARITY_PLOT_FILE_NAME);

            // Ensure any existing files are deleted before test
            if (File.Exists(reconstructionPlotPath)) File.Delete(reconstructionPlotPath);
            if (File.Exists(similarityPlotPath)) File.Delete(similarityPlotPath);

            // Act
            experiment.RunExperiment(10, 0);

            // Assert
            Assert.IsTrue(File.Exists(reconstructionPlotPath), "Reconstruction plot file not found.");
            Assert.IsTrue(File.Exists(similarityPlotPath), "Similarity plot file not found.");
        }
    }
}