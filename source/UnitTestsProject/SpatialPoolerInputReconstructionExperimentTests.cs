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
        }/// <summary>
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
        
    }
}