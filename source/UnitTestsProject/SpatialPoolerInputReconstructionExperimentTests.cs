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
    ///     Unit tests for the <see cref="SpatialPoolerInputReconstructionExperiment" />.
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
        [TestCategory("SmokeTest")]
        public void Test_Experiment_Completes_Without_Exception()
        {
            SpatialPoolerInputReconstructionExperiment experiment = new();
            experiment.ReconstructionExperiment(20);
        }
        
        /// <summary>
        ///     Tests that the Experiment terminates with improper value of max.
        ///     This test is expected to throw an ArgumentException.
        /// </summary>
        [TestMethod]
        [Priority(2)]
        [TestCategory("Experiment")]
        [TestCategory("ExceptionHandling")]
        public void Test_Experiment_With_Improper_Max_Value() 
        {
            var experiment = new SpatialPoolerInputReconstructionExperiment();
            var ex = Assert.ThrowsException<ArgumentException>(() => experiment.ReconstructionExperiment(8));
    
            // Explicitly check error message
            Assert.IsTrue(ex.Message.Contains("max must be 10 or greater"), "Error message should specify min max value.");
        }

        /// <summary>
        ///     Tests that the Spatial Pooler reaches a stable state during training.
        ///     This test captures the console output and checks for the "STABLE STATE REACHED" message.
        /// </summary>
        [TestMethod]
        [Priority(3)]
        [TestCategory("Experiment")]
        [TestCategory("SpatialPooler")]
        public void Test_SpatialPoolerTraining_ReachesStableState()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();
            TextWriter originalConsoleOut = Console.Out;
            StringWriter consoleOutput = new();

            Console.SetOut(consoleOutput);

            // Act
            experiment.ReconstructionExperiment(10);

            // Reset console output
            Console.SetOut(originalConsoleOut);
            string output = consoleOutput.ToString();

            // Assert
            Assert.IsTrue(output.Contains("STABLE STATE REACHED"), "Spatial Pooler did not reach stable state.");
        }

        /// <summary>
        ///     Tests that the input reconstruction phase produces valid predictions and similarity metrics.
        ///     This test captures the console output and checks for reconstruction results in the console.
        /// </summary>
        [TestMethod]
        [Priority(4)]
        [TestCategory("Experiment")]
        [TestCategory("Reconstruction")]
        public void Test_Reconstruction_ProducesPredictions()
        {
            // Arrange
            SpatialPoolerInputReconstructionExperiment experiment = new();
            TextWriter originalConsoleOut = Console.Out;
            StringWriter consoleOutput = new();

            Console.SetOut(consoleOutput);

            // Act
            experiment.ReconstructionExperiment(10);

            // Reset console output
            Console.SetOut(originalConsoleOut);
            string output = consoleOutput.ToString();

            // Assert
            Assert.IsTrue(output.Contains("KNN - Reconstructed Input"), "KNN predictions not found in output.");
            Assert.IsTrue(output.Contains("HTM - Reconstructed Input"), "HTM predictions not found in output.");
            Assert.IsTrue(output.Contains("Percentage Similarity"), "Similarity metrics not found in output.");
        }
        
        /// <summary>
        ///     Checks if reconstructed inputs from KNN/HTM have valid similarity scores (0-100%).
        /// </summary>
        [TestMethod]
        [Priority(5)]
        [TestCategory("Experiment")]
        [TestCategory("ClassifierAccuracy")]
        public void Test_ReconstructionPart_Results_Have_Valid_Similarity()
        {
            var experiment = new SpatialPoolerInputReconstructionExperiment();
            experiment.ReconstructionExperiment(10);
            
            foreach (var result in experiment.Results.Values)
            {
                Assert.IsTrue(result.KnnPercentageSimilarity is >= 0 and <= 1, "Invalid KNN similarity.");
                Assert.IsTrue(result.HtmPercentageSimilarity is >= 0 and <= 1, "Invalid HTM similarity.");
            }
        }
    }
}