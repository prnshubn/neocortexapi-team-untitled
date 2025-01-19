using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;

namespace NeoCortexApi.Experiments.SpatialPoolerInputReconstruction
{
    /// <summary>
    /// Input -> Scalar Encoder -> Spatial Pooler -> Classifier -> Output
    /// </summary>
    [TestClass]
    public class SpatialPoolerInputReconstructionExperiment
    {
        [TestMethod]
        [TestCategory("Experiment")]
        public void Setup()
        {
            int[] scalarInputs = { 1, 2, 3, 4, 5 };
            // Find the maximum and minimum values
            int maxValue = scalarInputs.Max();
            int minValue = scalarInputs.Min();
        } 
        
        static void ProcessTestCase(int[] scalarInputs, InputParams inputs, double minValue, double maxValue)
        {
            ScalarEncoder encoder = new (new Dictionary<string, object>()
            {
               // { "W", inputs.width},
                { "N", 1024},
               // { "Radius", inputs.radius},
                { "MinVal", minValue},
                { "MaxVal", maxValue},
                { "Periodic", false},
                { "Name", "scalar"},
                { "ClipInput", false},
            });

            }
            
        }


    }
public class InputParams
{
    
        
}