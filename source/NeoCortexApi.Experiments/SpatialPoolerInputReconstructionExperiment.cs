using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

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
        }


    }
}