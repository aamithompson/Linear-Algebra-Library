using System.Collections;
using System.Collections.Generic;
using lmath;
using NUnit.Framework;

[TestFixture]
public class lmathTensorTests {
    Tensor A_2D;
    Tensor A_3D;

    [SetUp]
    public void SetUp() {
        A_2D = new Tensor(new float[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        A_3D = new Tensor(new float[,,] {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{11, 12, 13}, {14, 15, 16}, {17, 18, 19}}, {{21, 22, 23}, {24, 25, 26}, {27, 28, 29}}});
    }

// DATA MANAGEMENT
//------------------------------------------------------------------------------
    [Test, Category("Data Management")]
    public void DataManagement_GetSlice_Scalar() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                float[] slice = A_2D.GetSlice(new int[,] {{i, i}, {j, j}});
                Assert.AreEqual(1, slice.Length, $"Scalar slice should only contain 1 element but contains {slice.Length} elements.");

                float value = slice[0];
                float expected = A_2D.GetElement(new int[] {i, j});
                Assert.AreEqual(expected, value, $"Element {value} at index {j} does not equal expected value {expected} from index [{i}, {j}] in original array.");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_GetSlice_Row() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        for(int i = 0; i < n; i++) {
            float[] slice = A_2D.GetSlice(new int[,] {{i, i}, {0, m-1}});
            Assert.AreEqual(m, slice.Length, $"Row slice should contain {m} elements but contains {slice.Length} elements.");

            for(int j = 0; j < m; j++) {
                float value = slice[j];
                float expected = A_2D.GetElement(new int[] { i, j });
                Assert.AreEqual(expected, value, $"Element {value} at index {j} does not equal expected value {expected} from index [{i}, {j}] in original array.");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_GetSlice_Column() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        for(int j = 0; j < m; j++) {
            float[] slice = A_2D.GetSlice(new int[,] {{0, n-1}, {j, j}});
            Assert.AreEqual(n, slice.Length, $"Column slice should contain {n} elements but contains {slice.Length} elements.");

            for(int i = 0; i < n; i++) {
                float value = slice[i];
                float expected = A_2D.GetElement(new int[] { i, j });
                Assert.AreEqual(expected, value, $"Element {value} at index {i} does not equal expected value {expected} from index [{i}, {j}] in original array.");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_GetSlice_Block() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        int xa = 1;
        int xb = n-1;
        int xsize = xb - xa + 1;
        int ya = 1;
        int yb = m-1;
        int ysize = yb - ya + 1;

        float[] slice = A_2D.GetSlice(new int[,] {{xa,xb}, {ya,yb}});
        Tensor result = new Tensor(slice, new int[] {xsize, ysize});
        Assert.AreEqual(xsize * ysize, slice.Length, $"Total slice length {slice.Length} does not equal expected length {xsize * ysize}.");
        Assert.AreEqual(xsize, result.GetShape()[0], $"Shape {0} length {result.GetShape()[0]} does not equal expected length {xsize}.");
        Assert.AreEqual(ysize, result.GetShape()[1], $"Shape {1} length {result.GetShape()[1]} does not equal expected length {ysize}.");
        for(int i = xa; i <= xb; i++) {
            for(int j = ya; j <= yb; j++) {
                float value = result.GetElement(new int[] { i - xa, j - ya});
                float expected = A_2D.GetElement(new int[] { i, j });
                Assert.AreEqual(expected, value, $"Element {value} at index [{i-xa}, {j-ya}] does not equal expected value {expected} from index [{i}, {j}] in original array.");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_GetSlice_FullBlock() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];

        float[] slice = A_2D.GetSlice(new int[,] {{0, n-1}, {0, m-1}});
        Tensor result = new Tensor(slice, new int[] {n, m});
        Assert.AreEqual(n*m, slice.Length, $"Total slice length {slice.Length} does not equal expected length {n * m}.");
        Assert.AreEqual(n, result.GetShape()[0], $"Shape {0} length {result.GetShape()[0]} does not equal expected length {n}.");
        Assert.AreEqual(m, result.GetShape()[1], $"Shape {1} length {result.GetShape()[1]} does not equal expected length {m}.");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                float value = result.GetElement(new int[] { i, j });
                float expected = A_2D.GetElement(new int[] { i, j });
                Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {j}] does not equal expected value {expected} from index [{i}, {j}] in original array.");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_GetSlice_3D_Block() {
        int n = A_3D.GetShape()[0];
        int m = A_3D.GetShape()[1];
        int p = A_3D.GetShape()[2];
        int xa = 1;
        int xb = n-1;
        int xsize = xb - xa + 1;
        int ya = 1;
        int yb = m-1;
        int ysize = yb - ya + 1;
        int za = 1;
        int zb = p-1;
        int zsize = zb - za + 1;

        float[] slice = A_3D.GetSlice(new int[,] {{xa,xb}, {ya,yb}, {za, zb}});
        Tensor result = new Tensor(slice, new int[] {xsize, ysize, zsize});
        Assert.AreEqual(xsize * ysize * zsize, slice.Length, $"Total slice length {slice.Length} does not equal expected length {xsize * ysize * zsize}.");
        Assert.AreEqual(xsize, result.GetShape()[0], $"Shape {0} length {result.GetShape()[0]} does not equal expected length {xsize}.");
        Assert.AreEqual(ysize, result.GetShape()[1], $"Shape {1} length {result.GetShape()[1]} does not equal expected length {ysize}.");
        Assert.AreEqual(zsize, result.GetShape()[2], $"Shape {2} length {result.GetShape()[2]} does not equal expected length {zsize}.");
        for (int i = xa; i <= xb; i++) {
            for(int j = ya; j <= yb; j++) {
                for(int k = za; k <= zb; k++) {
                    float value = result.GetElement(new int[] { i - xa, j - ya, k - za});
                    float expected = A_3D.GetElement(new int[] { i, j, k });
                    Assert.AreEqual(expected, value, $"Element {value} at index [{i-xa}, {j-ya}, {k-za}] does not equal expected value {expected} from index [{i}, {j}, {k}] in original array.");
                }
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_SetSlice_Scalar() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        Tensor expected = new Tensor(A_2D);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                int[] indices = {i, j};
                int[,] indicesRange = {{i, i}, {j, j}};
                A_2D.SetSlice(new float[] {-1}, indicesRange);
                for(int i2 = 0; i2 < n; i2++) {
                    for(int j2 = 0; j2 < m; j2++) {
                        int[] indices2 = {i2, j2};
                        float value = A_2D.GetElement(indices2);
                        if(i2 == i && j2 == j) {
                            Assert.AreEqual(-1f, value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {-1f}.");
                        } else {
                            Assert.AreEqual(expected.GetElement(indices2), value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {expected.GetElement(indices2)}.");
                        }
                    }
                }

                A_2D.SetElement(expected.GetElement(indices), indices);
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_SetSlice_Row() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        float[] slice = new float[m];
        System.Array.Fill(slice, -1f);
        Tensor expected = new Tensor(A_2D);
        for(int i = 0; i < n; i++) {
            int[,] indicesRange = {{i, i}, {0, m-1}};
            A_2D.SetSlice(slice, indicesRange);
            for(int i2 = 0; i2 < n; i2++) {
                for(int j2 = 0; j2 < m; j2++) {
                    int[] indices = {i2, j2};
                    float value = A_2D.GetElement(indices);
                    if(i2 == i) {
                        Assert.AreEqual(-1f, value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {-1f}.");
                    } else {
                        Assert.AreEqual(expected.GetElement(indices), value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {expected.GetElement(indices)}.");
                    }
                }
            }

            A_2D.SetSlice(expected.GetSlice(indicesRange), indicesRange);
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_SetSlice_Column() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        float[] slice = new float[n];
        System.Array.Fill(slice, -1f);
        Tensor expected = new Tensor(A_2D);
        for(int j = 0; j < n; j++) {
            int[,] indicesRange = {{0, n-1}, {j, j}};
            A_2D.SetSlice(slice, indicesRange);
            for(int i2 = 0; i2 < n; i2++) {
                for(int j2 = 0; j2 < m; j2++) {
                    int[] indices = {i2, j2};
                    float value = A_2D.GetElement(indices);
                    if(j2 == j) {
                        Assert.AreEqual(-1f, value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {-1f}.");
                    } else {
                        Assert.AreEqual(expected.GetElement(indices), value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {expected.GetElement(indices)}.");
                    }
                }
            }

            A_2D.SetSlice(expected.GetSlice(indicesRange), indicesRange);
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_SetSlice_Block() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        float[] slice = new float[(n - 1) * (m - 1)];
        System.Array.Fill(slice, -1f);
        Tensor expected = new Tensor(A_2D);
        for(int i = 0; i <= 1; i++) {
            int xa = i;
            int xb = n - (2 - i);
            for (int j = 0; j <= 1; j++) {
                int ya = j;
                int yb = m - (2 - j);
                int[,] indicesRange = {{xa, xb}, {ya, yb}};
                A_2D.SetSlice(slice, indicesRange);
                for(int i2 = 0; i2 < n; i2++) {
                    for(int j2 = 0; j2 < m; j2++) {
                        int[] indices = {i2, j2};
                        float value = A_2D.GetElement(indices);
                        if((i2 >= xa && i2 <= xb) && (j2 >= ya && j2 <= yb)) {
                            Assert.AreEqual(-1f, value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {-1f}.");
                        } else {
                            Assert.AreEqual(expected.GetElement(indices), value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {expected.GetElement(indices)}.");
                        }
                    }
                }

                A_2D.SetSlice(expected.GetSlice(indicesRange), indicesRange);
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_SetSlice_FullBlock() {
        int n = A_2D.GetShape()[0];
        int m = A_2D.GetShape()[1];
        float[] slice = new float[n * m];
        System.Array.Fill(slice, -1f);
        int[,] indicesRange = {{0, n - 1}, {0, m-1}};
        A_2D.SetSlice(slice, indicesRange);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                int[] indices = {i, j};
                float value = A_2D.GetElement(indices);
                Assert.AreEqual(-1f, value, $"Element {value} at index [{i}, {j}] does not equal expected value {-1}.");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_SetSlice_3D_Block() {
        int n = A_3D.GetShape()[0];
        int m = A_3D.GetShape()[1];
        int p = A_3D.GetShape()[2];
        float[] slice = new float[(n-1) * (m-1) * (p-1)];
        System.Array.Fill(slice, -1f);
        Tensor expected = new Tensor(A_3D);
        for (int i = 0; i <= 1; i++) {
            int xa = i;
            int xb = n - (2 - i);
            for(int j = 0; j <= 1; j++) {
                int ya = j;
                int yb = m - (2 - j);
                for(int k = 0; k <= 1; k++) {
                    int za = k;
                    int zb = p - (2 - k);
                    int[,] indicesRange = {{xa, xb}, {ya, yb}, {za, zb}};
                    A_3D.SetSlice(slice, indicesRange);
                    for(int i2 = 0; i2 < n; i2++) {
                        for(int j2 = 0; j2 < m; j2++) {
                            for (int k2 = 0; k2 < p; k2++) {
                                int[] indices = {i2, j2, k2};
                                float value = A_3D.GetElement(indices);
                                if((i2 >= xa && i2 <= xb) && (j2 >= ya && j2 <= yb) && (k2 >= za && k2 <= zb)) {
                                    Assert.AreEqual(-1f, value, $"Element {value} at index [{i2}, {j2}, {k2}] does not equal expected value {-1f}.");
                                } else {
                                    Assert.AreEqual(expected.GetElement(indices), value, $"Element {value} at index [{i2}, {j2}, {k2}] does not equal expected value {expected.GetElement(indices)}.");
                                }
                            }
                        }
                    }

                    A_3D.SetSlice(expected.GetSlice(indicesRange), indicesRange);
                }
            }
        }
    }
}

