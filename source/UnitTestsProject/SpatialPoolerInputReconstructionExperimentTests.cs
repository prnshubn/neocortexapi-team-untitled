using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Experiments;

namespace UnitTestsProject
{
    /// <summary>
    ///     <see href="https://github.com/prnshubn/neocortexapi-team-untitled">Project Link</see><br />
    ///     <see href="https://github.com/prnshubn/neocortexapi-team-untitled/tree/master/source/Documentation_Team_Untitled">Documentation</see>
    ///     <br />
    ///     Unit tests for the <see cref="SpatialPoolerInputReconstructionExperiment" /> class.
    ///     These tests validate the correct execution of the experiment, the training of the Spatial Pooler,
    ///     and the input reconstruction process using KNN and HTM classifiers.
    /// </summary>
    /// <br />
    /// <para>
    ///     University: Frankfurt University of Applied Sciences<br />
    ///     Degree: Master's in Information Technology<br />
    ///     Year: 2024-2025<br />
    ///     Team: Untitled<br />
    ///     Contributors:
    ///     <see href="https://github.com/prnshubn">Priyanshu Bandyopadhyay</see>,
    ///     <see href="https://github.com/Hanumanthumanoj01">Manoj Hanumanthu</see>,
    ///     <see href="https://github.com/Akshay-Gudekar">Akshay Gudekar</see>
    /// </para>
    [TestClass]
    public class SpatialPoolerInputReconstructionExperimentTests
    {
        /// <summary>
        ///     Tests that the RunExperiment method executes without throwing any exceptions.
        ///     This is a basic smoke test to ensure the experiment runs to completion.
        /// </summary>
        [TestMethod]
        [Priority(1)]
        [TestCategory("Experiment")]
        public void Test_Experiment_Completes_Without_Exception()
        {
            SpatialPoolerInputReconstructionExperiment experiment = new();
            experiment.RunExperiment(20, 0);
        }

        /// <summary>
        ///     Tests that the reconstruction and percentage similarities are according to the predefined results.
        /// </summary>
        [TestMethod]
        [Priority(2)]
        [TestCategory("Experiment")]
        public void Test_Experiment_With_Random_Seed_Value()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();

            // Act
            experiment.RunExperiment(20, 42);

            // Assert
            List<double> testData = experiment.Results.Keys.ToList();

            // The below values are valid only for input of 20 and seedValue of 42
            double[] expectedData = { 1.00, 7.00, 10.00, 19.00 };

            Assert.IsTrue(expectedData.All(v => testData.Contains(v)));
        }

        /// <summary>
        ///     Tests that the Spatial Pooler reaches a stable state during training.
        ///     This test captures the console output and checks for the "STABLE STATE REACHED" message.
        /// </summary>
        [TestMethod]
        [Priority(3)]
        [TestCategory("Experiment")]
        public void Test_SpatialPoolerTraining_ReachesStableState()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();
            TextWriter originalConsoleOut = Console.Out;
            StringWriter consoleOutput = new();

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
        ///     Tests that the input reconstruction phase produces valid predictions and similarity metrics.
        ///     This test captures the console output and checks for reconstruction results.
        /// </summary>
        [TestMethod]
        [Priority(4)]
        [TestCategory("Experiment")]
        public void Test_Reconstruction_ProducesPredictions()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();
            TextWriter originalConsoleOut = Console.Out;
            StringWriter consoleOutput = new();

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
    }
}