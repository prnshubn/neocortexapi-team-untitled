using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Experiments;
using System;
using System.IO;

namespace UnitTestsProject
{
    /// <summary>
    /// Unit tests for the SpatialPoolerInputReconstructionExperiment class.
    /// These tests validate the correct execution of the experiment, the training of the Spatial Pooler,
    /// and the input reconstruction process using KNN and HTM classifiers.
    /// </summary>
    [TestClass]
    public class SpatialPoolerInputReconstructionExperimentTests
    {
        private const string ReconstructionPlotFileName = "ReconstructionPlot.png";
        private const string SimilarityPlotFileName = "SimilarityPlot.png";

        /// <summary>
        /// Tests that the RunExperiment method executes without throwing any exceptions.
        /// This is a basic smoke test to ensure the experiment runs to completion.
        /// </summary>
        [TestMethod]
        [TestCategory("Experiment")]
        public void Test_RunExperiment_CompletesWithoutException()
        {
            // Arrange
            var experiment = new SpatialPoolerInputReconstructionExperiment();

            // Act & Assert
            experiment.RunExperiment();
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
            var experiment = new SpatialPoolerInputReconstructionExperiment();
            var originalConsoleOut = Console.Out;
            var consoleOutput = new StringWriter();

            Console.SetOut(consoleOutput);

            // Act
            experiment.RunExperiment();

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
            var experiment = new SpatialPoolerInputReconstructionExperiment();
            var originalConsoleOut = Console.Out;
            var consoleOutput = new StringWriter();

            Console.SetOut(consoleOutput);

            // Act
            experiment.RunExperiment();

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
            var experiment = new SpatialPoolerInputReconstructionExperiment();
            string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string reconstructionPlotPath = Path.Combine(desktopPath, ReconstructionPlotFileName);
            string similarityPlotPath = Path.Combine(desktopPath, SimilarityPlotFileName);

            // Ensure any existing files are deleted before test
            if (File.Exists(reconstructionPlotPath)) File.Delete(reconstructionPlotPath);
            if (File.Exists(similarityPlotPath)) File.Delete(similarityPlotPath);

            // Act
            experiment.RunExperiment();

            // Assert
            Assert.IsTrue(File.Exists(reconstructionPlotPath), "Reconstruction plot file not found.");
            Assert.IsTrue(File.Exists(similarityPlotPath), "Similarity plot file not found.");
        }
    }
}