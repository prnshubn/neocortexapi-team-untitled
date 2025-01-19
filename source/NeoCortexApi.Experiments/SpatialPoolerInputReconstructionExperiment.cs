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
            List<int[]> sdrList = new ();
            
            foreach (var i in scalarInputs)
            {
                var result = encoder.Encode(i);
                sdrList.Add(result);
                Console.WriteLine($"Input: {i} -> SDR: {string.Join("", result)}");
            }

            // Initialize the SpatialPooler and its connections
            // SpatialPooler sp = new ();
            // Connections connections = new ();
            // sp.Init(connections);
            //
            // int[] sdr2 = sp.Compute(sdrList[0], false);
            // Console.WriteLine($"Input: {1} -> SDR: {string.Join("", sdr2)}");

            }
            
        }


    }
public class InputParams
{
    
        
}