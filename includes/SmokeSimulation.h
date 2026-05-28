#ifndef SMOKE_SIMULATION_H
#define SMOKE_SIMULATION_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>

class SmokeSimulation {
public:
    int nx, ny, nz; 
    float dx;       
    
    // Staggered grid fields
    std::vector<float> u, u_prev; 
    std::vector<float> v, v_prev; 
    std::vector<float> w, w_prev; 
    std::vector<float> density, density_prev; 
    std::vector<float> pressure;  

    SmokeSimulation(int resX, int resY, int resZ, float cellSize) 
        : nx(resX), ny(resY), nz(resZ), dx(cellSize) {
        int numCells = nx * ny * nz;
        u.resize((nx + 1) * ny * nz, 0.0f); u_prev = u;
        v.resize(nx * (ny + 1) * nz, 0.0f); v_prev = v;
        w.resize(nx * ny * (nz + 1), 0.0f); w_prev = w;
        density.resize(numCells, 0.0f); density_prev = density;
        pressure.resize(numCells, 0.0f);
    }

    inline int dIdx(int i, int j, int k) { return i + j * nx + k * nx * ny; }
    inline int uIdx(int i, int j, int k) { return i + j * (nx + 1) + k * (nx + 1) * ny; }
    inline int vIdx(int i, int j, int k) { return i + j * nx + k * nx * (ny + 1); }
    inline int wIdx(int i, int j, int k) { return i + j * nx + k * nx * ny; }

    void EmitSmoke(int i, int j, int k, float amount, glm::vec3 initialVel) {
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1) {
            density[dIdx(i, j, k)] = std::min(density[dIdx(i, j, k)] + amount, 1.0f);
            u[uIdx(i, j, k)] = initialVel.x;
            v[vIdx(i, j, k)] = initialVel.y;
            w[wIdx(i, j, k)] = initialVel.z;
        }
    }

    // Step 1: Linear Backwards Tracing (Advection)
    void Advect(float dt) {
        // CRITICAL FIX 1: Save current states to previous arrays before tracking backward paths
        density_prev = density;
        u_prev = u;
        v_prev = v;
        w_prev = w;

        // --- Advect Density Array ---
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    float avgU = 0.5f * (u_prev[uIdx(i, j, k)] + u_prev[uIdx(i + 1, j, k)]);
                    float avgV = 0.5f * (v_prev[vIdx(i, j, k)] + v_prev[vIdx(i, j + 1, k)]);
                    float avgW = 0.5f * (w_prev[wIdx(i, j, k)] + w_prev[wIdx(i, j, k + 1)]);

                    float oldX = (float)i + 0.5f - (dt / dx) * avgU;
                    float oldY = (float)j + 0.5f - (dt / dx) * avgV;
                    float oldZ = (float)k + 0.5f - (dt / dx) * avgW;

                    density[dIdx(i, j, k)] = SampleDensityLinear(oldX, oldY, oldZ);
                }
            }
        }

        // CRITICAL FIX 2: Advect Velocities so motion flows across cells smoothly
        // Advect U field
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx; ++i) {
                    float avgV = 0.25f * (v_prev[vIdx(i-1, j, k)] + v_prev[vIdx(i, j, k)] + v_prev[vIdx(i-1, j+1, k)] + v_prev[vIdx(i, j+1, k)]);
                    float avgW = 0.25f * (w_prev[wIdx(i-1, j, k)] + w_prev[wIdx(i, j, k)] + w_prev[wIdx(i-1, j, k+1)] + w_prev[wIdx(i, j, k+1)]);
                    
                    float oldX = (float)i - (dt / dx) * u_prev[uIdx(i, j, k)];
                    float oldY = (float)j + 0.5f - (dt / dx) * avgV;
                    float oldZ = (float)k + 0.5f - (dt / dx) * avgW;
                    
                    u[uIdx(i, j, k)] = SampleFieldLinear(oldX, oldY, oldZ, u_prev, nx + 1, ny, nz, true);
                }
            }
        }

        // Advect V field
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    float avgU = 0.25f * (u_prev[uIdx(i, j-1, k)] + u_prev[uIdx(i, j, k)] + u_prev[uIdx(i+1, j-1, k)] + u_prev[uIdx(i+1, j, k)]);
                    float avgW = 0.25f * (w_prev[wIdx(i, j-1, k)] + w_prev[wIdx(i, j, k)] + w_prev[wIdx(i, j-1, k+1)] + w_prev[wIdx(i, j, k+1)]);
                    
                    float oldX = (float)i + 0.5f - (dt / dx) * avgU;
                    float oldY = (float)j - (dt / dx) * v_prev[vIdx(i, j, k)];
                    float oldZ = (float)k + 0.5f - (dt / dx) * avgW;
                    
                    v[vIdx(i, j, k)] = SampleFieldLinear(oldX, oldY, oldZ, v_prev, nx, ny + 1, nz, false);
                }
            }
        }

        // Advect W field
        for (int k = 1; k < nz; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    float avgU = 0.25f * (u_prev[uIdx(i, j, k-1)] + u_prev[uIdx(i, j, k)] + u_prev[uIdx(i+1, j, k-1)] + u_prev[uIdx(i+1, j, k)]);
                    float avgV = 0.25f * (v_prev[vIdx(i, j, k-1)] + v_prev[vIdx(i, j, k)] + v_prev[vIdx(i, j+1, k-1)] + v_prev[vIdx(i, j+1, k)]);
                    
                    float oldX = (float)i + 0.5f - (dt / dx) * avgU;
                    float oldY = (float)j + 0.5f - (dt / dx) * avgV;
                    float oldZ = (float)k - (dt / dx) * w_prev[wIdx(i, j, k)];
                    
                    w[wIdx(i, j, k)] = SampleFieldLinear(oldX, oldY, oldZ, w_prev, nx, ny, nz + 1, false);
                }
            }
        }

        // Fade smoke gently down
        for (auto& d : density) d *= 0.992f; 
    }

    void Project(int iterations) {
        std::fill(pressure.begin(), pressure.end(), 0.0f);

        for (int iter = 0; iter < iterations; ++iter) {
            for (int k = 1; k < nz - 1; ++k) {
                for (int j = 1; j < ny - 1; ++j) {
                    for (int i = 1; i < nx - 1; ++i) {
                        float divergence = (u[uIdx(i + 1, j, k)] - u[uIdx(i, j, k)] +
                                            v[vIdx(i, j + 1, k)] - v[vIdx(i, j, k)] +
                                            w[wIdx(i, j, k + 1)] - w[wIdx(i, j, k)]) / dx;

                        float p = -divergence / (6.0f / (dx * dx));
                        pressure[dIdx(i, j, k)] += p;

                        u[uIdx(i, j, k)]         -= p / dx;
                        u[uIdx(i + 1, j, k)]     += p / dx;
                        v[vIdx(i, j, k)]         -= p / dx;
                        v[vIdx(i, j + 1, k)]     += p / dx;
                        w[wIdx(i, j, k)]         -= p / dx;
                        w[wIdx(i, j, k + 1)]     += p / dx;
                    }
                }
            }
        }
    }

    void UpdateBuoyancy(float dt) {
        float buoyancyForce = 3.5f; // Boosted force slightly for snappier vertical movement
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    float d = density[dIdx(i, j, k)];
                    if (d > 0.01f) {
                        v[vIdx(i, j, k)] += d * buoyancyForce * dt;
                    }
                }
            }
        }
    }

    void Step(float dt) {
        UpdateBuoyancy(dt);
        Project(5); 
        Advect(dt);
    }

private:
    float SampleDensityLinear(float x, float y, float z) {
        x = std::clamp(x, 0.5f, (float)nx - 1.5f);
        y = std::clamp(y, 0.5f, (float)ny - 1.5f);
        z = std::clamp(z, 0.5f, (float)nz - 1.5f);

        int i0 = (int)(x - 0.5f); float tx = (x - 0.5f) - (float)i0;
        int j0 = (int)(y - 0.5f); float ty = (y - 0.5f) - (float)j0;
        int k0 = (int)(z - 0.5f); float tz = (z - 0.5f) - (float)k0;

        int i1 = i0 + 1; int j1 = j0 + 1; int k1 = k0 + 1;

        float d000 = density_prev[dIdx(i0, j0, k0)];
        float d100 = density_prev[dIdx(i1, j0, k0)];
        float d010 = density_prev[dIdx(i0, j1, k0)];
        float d110 = density_prev[dIdx(i1, j1, k0)];
        float d001 = density_prev[dIdx(i0, j0, k1)];
        float d101 = density_prev[dIdx(i1, j0, k1)];
        float d011 = density_prev[dIdx(i0, j1, k1)];
        float d111 = density_prev[dIdx(i1, j1, k1)];

        return glm::mix(glm::mix(glm::mix(d000, d100, tx), glm::mix(d010, d110, tx), ty),
                        glm::mix(glm::mix(d001, d101, tx), glm::mix(d011, d111, tx), ty), tz);
    }

    // Generic staggered velocity vector look-up interpolation helper
    float SampleFieldLinear(float x, float y, float z, const std::vector<float>& field, int fNx, int fNy, int fNz, bool isU) {
        float xOff = isU ? 0.0f : 0.5f;
        float yOff = !isU && (fNy > ny) ? 0.0f : 0.5f; // Check for V orientation mismatch handles
        
        x = std::clamp(x, xOff, (float)fNx - 1.5f);
        y = std::clamp(y, yOff, (float)fNy - 1.5f);
        z = std::clamp(z, 0.5f, (float)fNz - 1.5f);

        int i0 = (int)(x - xOff); float tx = (x - xOff) - (float)i0;
        int j0 = (int)(y - yOff); float ty = (y - yOff) - (float)j0;
        int k0 = (int)(z - 0.5f); float tz = (z - 0.5f) - (float)k0;

        int i1 = std::min(i0 + 1, fNx - 1);
        int j1 = std::min(j0 + 1, fNy - 1);
        int k1 = std::min(k0 + 1, fNz - 1);

        auto idx = [&](int i, int j, int k) { return i + j * fNx + k * fNx * fNy; };

        float v000 = field[idx(i0, j0, k0)]; float v100 = field[idx(i1, j0, k0)];
        float v010 = field[idx(i0, j1, k0)]; float v110 = field[idx(i1, j1, k0)];
        float v001 = field[idx(i0, j0, k1)]; float v101 = field[idx(i1, j0, k1)];
        float v011 = field[idx(i0, j1, k1)]; float v111 = field[idx(i1, j1, k1)];

        return glm::mix(glm::mix(glm::mix(v000, v100, tx), glm::mix(v010, v110, tx), ty),
                        glm::mix(glm::mix(v001, v101, tx), glm::mix(v011, v111, tx), ty), tz);
    }
};

#endif