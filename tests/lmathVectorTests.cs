using System.Collections;
using System.Collections.Generic;
using lmath;
using statistics;
using NUnit.Framework;

[TestFixture]
public class lmathVectorTests {
    Vector a;
    Vector b;
    Vector c;
    Vector zeros;
    Vector ones;

    [SetUp]
    public void Setup() {
        a = new Vector(new float[] { 1f, 3f, -5f });
        b = new Vector(new float[] { 4f, -2f, -1f });
        c = new Vector();
        zeros = Vector.Zeros(3);
        ones = Vector.Ones(3);
    }

    // CONSTRUCTORS
    //------------------------------------------------------------------------------
    [Test, Category("Constructor")]
    public void Constructor_Empty() {
        Assert.NotNull(c.AccessData(), "Empty vector internal data array is not initialized.");
        Assert.AreEqual(1, c.rank, "Empty vector does not have rank 1.");
        Assert.AreEqual(0, c.length, "Empty vector does not have length 0.");
    }

    [Test, Category("Constructor")]
    public void Constructor_FloatArray() {
        float[] elements = new float[] { 1f, 3f, -5f };
        c = new Vector(elements);
        Assert.NotNull(c.AccessData(), "Vector internal data array is not initialized.");
        Assert.AreEqual(1, c.rank, "Vector does not have rank 1.");
        Assert.AreEqual(elements.Length, c.length, $"Vector length {c.length} is not expected length {elements.Length}");

        for(int i = 0; i < elements.Length; i++) {
            Assert.AreEqual(elements[i], c[i], $"Element {c[i]} at index {i} does not equal expected element {elements[i]}");
        }
    }

    [Test, Category("Constructor")]
    public void Constructor_Vector() {
        c = new Vector(a);
        Assert.NotNull(c.AccessData(), "Vector internal data array is not initialized.");
        Assert.AreNotSame(a.AccessData(), c.AccessData(), "Copy constructor did not create a deep copy of internal data.");
        Assert.AreEqual(1, c.rank, "Vector does not have rank 1.");
        Assert.AreEqual(a.length, c.length, $"Vector length {c.length} is not expected length {a.length}");

        for (int i = 0; i < a.length; i++) {
            Assert.AreEqual(a[i], c[i], $"Element {c[i]} at index {i} does not equal expected element {a[i]}");
        }
    }

    // Data Management
    //------------------------------------------------------------------------------
    [Test, Category("Data Management")]
    public void DataManagement_Reshape_Vector_GreaterSize() {
        int oldLength = a.length;
        b.Copy(a);
        a.Reshape(a.length + 3);

        Assert.AreEqual(oldLength + 3, a.length, $"Vector length {a.length} is not expected length {oldLength + 3}");

        for (int i = 0; i < oldLength; i++) {
            Assert.AreEqual(b[i], a[i], $"Element {a[i]} at index {i} does not equal expected element {b[i]}");
        }

        for (int i = oldLength; i < a.length; i++) {
            Assert.AreEqual(a[i], 0, $"Element {a[i]} at index {i} did not fill 0 element in indexes beyond the initial length {oldLength}");
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_Reshape_Vector_SameSize() {
        int oldLength = a.length;
        b.Copy(a);
        a.Reshape(a.length);

        Assert.AreEqual(oldLength, a.length, $"Vector length {a.length} is not expected length {oldLength}");

        for (int i = 0; i < oldLength; i++) {
            Assert.AreEqual(b[i], a[i], $"Element {a[i]} at index {i} does not equal expected element {b[i]}");
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_Reshape_Vector_SmallerSize() {
        int oldLength = a.length;
        b.Copy(a);
        a.Reshape(a.length - 2);

        Assert.AreEqual(oldLength - 2, a.length, $"Vector length {a.length} is not expected length {oldLength - 2}");

        for (int i = 0; i < a.length; i++) {
            Assert.AreEqual(b[i], a[i], $"Element {a[i]} at index {i} does not equal expected element {b[i]}");
        }
    }

    // DEFAULT OBJECTS
    //------------------------------------------------------------------------------
    [Test, Category("Default Objects")]
    public void DefaultObject_Zeros() {
        Assert.AreEqual(3, zeros.length, $"Vector length {zeros.length} is not expected length {3}");

        for (int i = 0; i < zeros.length; i++) {
            Assert.AreEqual(0f, zeros[i], $"Element {zeros[i]} at index {i} does not equal expected element {0f}");
        }
    }

    [Test, Category("Default Objects")]
    public void DefaultObject_Ones() {
        Assert.AreEqual(3, ones.length, $"Vector length {ones.length} is not expected length {3}");

        for(int i = 0; i < ones.length; i++) {
            Assert.AreEqual(1f, ones[i], $"Element {ones[i]} at index {i} does not equal expected element {1f}");
        }
    }

    // OPERATIONS
    //------------------------------------------------------------------------------
    [Test, Category("Operations")]
    public void Operations_Scale() {
        ones.Scale(2f);

        for (int i = 0; i < ones.length; i++) {
            Assert.That(ones[i], Is.EqualTo(2f).Within(1e-5f), $"Element {ones[i]} at index {i} does not equal expected element {2f}");
        }
    }

    [Test, Category("Operations")]
    public void Operations_Negate() {
        c = Vector.Negate(a);

        for (int i = 0; i < c.length; i++) {
            Assert.That(c[i], Is.EqualTo(-a[i]).Within(1e-5f), $"Element {c[i]} at index {i} does not equal expected element {-a[i]}");
        }
    }

    [TestCase("Add"), Category("Operations")]
    [TestCase("Subtract"), Category("Operations")]
    [TestCase("Hadamard"), Category("Operations")]
    public void Operations_TwoVectors_ElementWise(string operation) {
        switch(operation) {
            case "Add": c = a + b; break;
            case "Subtract": c = a - b; break;
            case "Hadamard": c = Vector.HadamardProduct(a, b); break;
            default: throw new System.ArgumentException();
        }

        for (int i = 0; i < c.length; i++) {
            float expected = operation switch {
                "Add" => a[i] + b[i],
                "Subtract" => a[i] - b[i],
                "Hadamard" => a[i] * b[i],
                _ => 0
            };

            Assert.That(c[i], Is.EqualTo(expected).Within(1e-5f), $"Element {c[i]} at index {i} does not equal expected element {expected}");
        }
    }

    [Test, Category("Operations")]
    public void Operations_DotProduct_TwoVectors_CorrectResult() {
        Assert.That(a.Dot(b), Is.EqualTo(3f).Within(1e-5f));
    }

    [Test, Category("Operations")]
    public void Operations_CrossProduct() {
        c = a.Cross(b);
        Vector expected = new Vector(new float[] {-13, -19, -14});
        for (int i = 0; i < 3; i++) {
            Assert.That(c[i], Is.EqualTo(expected[i]).Within(1e-5f), $"Element {c[i]} at index {i} does not equal expected element {expected[i]}");
        }
    }

    [Test, Category("Operations")]
    public void Operations_Norm_L1() {
        float norm = a.Norm(1);
        Assert.That(norm, Is.EqualTo(9f).Within(1e-5f));
    }

    [Test, Category("Operations")]
    public void Operations_Norm_L2() {
        float norm = a.EuclidNorm();
        Assert.That(norm, Is.EqualTo(System.MathF.Sqrt(35)).Within(1e-5f));
    }

    [Test, Category("Operations")]
    public void Operations_Norm_L3() {
        float norm = a.Norm(3);
        Assert.That(norm, Is.EqualTo(System.MathF.Pow(153, 1f/3f)).Within(1e-5f));
    }

    [Test, Category("Operations")]
    public void Operations_MaxNorm() {
        float norm = a.MaxNorm();
        Assert.That(norm, Is.EqualTo(5f).Within(1e-5f));
    }

    [Test, Category("Operations")]
    public void Operations_UnitVector() {
        c = a.Unit();
        for(int i = 0; i < a.length; i++) {
            Assert.That(c[i], Is.EqualTo(a[i]/a.EuclidNorm()).Within(1e-5f));
        }
    }

    // RANDOM
    //------------------------------------------------------------------------------
    [Test, Category("Random")]
    public void Random_Uniform_VectorFromFloats() {
        int bcount = 100; //Bucket Count
        int scount = 100000; //Sample Count

        Vector sum = Vector.Zeros(3);
        Vector[] buckets = new Vector[3] {Vector.Zeros(bcount), Vector.Zeros(bcount), Vector.Zeros(bcount) };
        for(int i = 0; i < scount; i++) {
            c = Vector.Random(0, 1, 3);
            for(int j = 0; j < 3; j++) {
                Assert.That(c[j], Is.InRange(0, 1));
                int bucket = (int)System.Math.Min(bcount-1, System.Math.Floor(c[j] * bcount));
                buckets[j][bucket]++;
            }
            sum += c;
        }

        Vector mean = sum * (1f / scount);
        float expectedMean = (1f + 0f) / 2f;
        for (int i = 0; i < 3; i++) {
            Assert.That(mean[i], Is.EqualTo(expectedMean).Within(1e-2f));
        }

        int expectedCount = scount / bcount;
        double p = 1.0 / bcount;
        double sigma = System.Math.Sqrt(scount * p * (1-p));
        int expectedMin = expectedCount - (int)(4 * sigma);
        int expectedMax = expectedCount + (int)(4 * sigma);
        for (int i = 0; i < bcount; i++) {
            for(int j = 0; j < 3; j++) {
                Assert.That(buckets[j][i], Is.InRange(expectedMin, expectedMax));
            }
        }
    }

    [Test, Category("Random")]
    public void Random_Normal_VectorFromFloats() {
        const int bcount = 3; //Bucket Count
        int scount = 100000; //Sample Count
        float expectedMean = 1f;
        float expectedSTDDev = 1f;
        Vector bvalues = Vector.Zeros(bcount);
        for(int i = 0; i < bcount; i++) {
            double cdf = Statistics.NormalCDF(0.0, 1.0, i+1);
            double probability = 2 * cdf - 1;
            bvalues[i] = (float) probability;

        }

        for(int i = bcount - 1; i > 0; i--) {
            bvalues[i] -= bvalues[i - 1];
        }

        Vector sum = Vector.Zeros(3);
        Vector sumSQ = Vector.Zeros(3);
        Vector[] buckets = new Vector[3] {Vector.Zeros(bcount), Vector.Zeros(bcount), Vector.Zeros(bcount) };
        for(int i = 0; i < scount; i++) {
            c = Vector.RandomN(expectedMean, expectedSTDDev, 3);
            sum += c;
            for(int j = 0; j < 3; j++) {
                sumSQ[j] += c[j] * c[j];
            }

            for(int j = 0; j < 3; j++) {
                bool bucketFound = false;
                float value = System.MathF.Abs(c[j] - expectedMean);
                for(int k = 0; k < bcount-1; k++) {
                    if(value <= (k+1) * expectedSTDDev) {
                        buckets[j][k]++;
                        bucketFound = true;
                        break;
                    }
                }

                if(!bucketFound) {
                    buckets[j][bcount - 1]++;
                }
            }
        }

        Vector mean = sum * (1f / scount);
        for(int i = 0; i < 3; i++) {
            Assert.That(mean[i], Is.EqualTo(expectedMean).Within(1e-2f));
        }

        Vector variance = (sumSQ * (1f / scount)) - Vector.HadamardProduct(mean, mean);
        Vector stdDev = new Vector(variance);
        for(int i = 0; i < 3; i++) {
            stdDev[i] = System.MathF.Sqrt(stdDev[i]);
            Assert.That(stdDev[i], Is.EqualTo(expectedSTDDev).Within(1e-2f));
        }
        
        for(int i = 0; i < bcount; i++) {
            for (int j = 0; j < 3; j++) {
                float percent = (float)buckets[j][i]/scount;
                Assert.That(percent, Is.EqualTo(bvalues[i]).Within(1e-2f));
            }
        }
    }

    // CONDITIONS
    //------------------------------------------------------------------------------
    [Test, Category("Conditions")]
    public void Condition_IsUnit() {
        Assert.IsTrue(new Vector(new float[] { 1f, 0f, 0f }).IsUnit());
        Assert.IsFalse(a.IsUnit());
        Assert.IsTrue(a.Unit().IsUnit());
    }

    [Test, Category("Conditions")]
    public void Condition_IsOrthogonal() {
        Assert.IsFalse(a.IsOrthogonal(b));
        a = new Vector(new float[] { 1f, 0f, 0f });
        b = new Vector(new float[] {0f, 1f, 0f });
        Assert.IsTrue(a.IsOrthogonal(b));
        a *= 2;
        b *= 5;
        Assert.IsTrue(a.IsOrthogonal(b));
    }

    [Test, Category("Conditions")]
    public void Condition_IsOrthonormal() {
        Assert.IsFalse(a.IsOrthonormal(b));
        a = new Vector(new float[] { 1f, 0f, 0f });
        b = new Vector(new float[] { 0f, 1f, 0f });
        Assert.IsTrue(a.IsOrthonormal(b));
        a *= 2;
        b *= 5;
        Assert.IsFalse(a.IsOrthonormal(b));
    }
}
