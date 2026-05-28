#ifndef SOFT_BODY_H
#define SOFT_BODY_H

#include <vector>
#include <unordered_map>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Forward declaration of your engine's Shader class
class Shader;

struct SoftBodyParticle {
    glm::vec3 pos;
    glm::vec3 prevPos;
    glm::vec3 vel;
    float invMass;
};

struct EdgeConstraint {
    int id0, id1;
    float restLength;
    float compliance;
    float lambda; // Tracks the accumulated Lagrange multiplier for XPBD accuracy
};

struct SoftBodyVertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

// Helper to hash glm::vec3 for quick vertex welding lookups
struct Vec3Hash {
    std::size_t operator()(const glm::vec3& v) const {
        // Simple hash combining with a small epsilon tolerance
        auto h1 = std::hash<float>()(std::round(v.x * 1000.0f));
        auto h2 = std::hash<float>()(std::round(v.y * 1000.0f));
        auto h3 = std::hash<float>()(std::round(v.z * 1000.0f));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class SoftBody {
private:
    float savedInvMass = 1.0f;
public:
    std::vector<SoftBodyParticle> particles;
    std::vector<EdgeConstraint> constraints;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> surfaceIndices;

    unsigned int VAO, VBO, EBO;

    SoftBody() : VAO(0), VBO(0), EBO(0) {}

    // Destructor to clean up OpenGL memory resource leaks
    ~SoftBody() {
        cleanupBuffers();
    }

    void SetGrabbedParticle(int index, const glm::vec3& targetPosition) {
        if (index < 0 || index >= static_cast<int>(particles.size())) return;

        // If this is the initial frame of the grab operation, back up the weight scalar
        if (particles[index].invMass > 0.0f) {
            savedInvMass = particles[index].invMass;
        }

        // Overwrite kinematics directly
        particles[index].invMass = 0.0f; // Locks it in place for XPBD solver
        particles[index].pos = targetPosition;
        particles[index].vel = glm::vec3(0.0f);
    }

    void ReleaseGrabbedParticle(int index) {
        if (index < 0 || index >= static_cast<int>(particles.size())) return;

        // Restore physical behavior tracking variables
        particles[index].invMass = savedInvMass;
    }


    void InitializeFromModelTetrahedral(const Model& model, const glm::vec3& spawnPos, const glm::vec3& scale, float compliance)
    {
        this->particles.clear();
        this->constraints.clear();
        this->indices.clear();         // Will store our raw tetrahedron indices (4 per cell)
        this->surfaceIndices.clear();   // Will store clean, outward-facing rendering triangles (3 per face)
        this->particleTexCoords.clear();
        this->cleanupBuffers();

        // HashMap to weld identical positions into unique physical particles
        std::unordered_map<glm::vec3, unsigned int, Vec3Hash> vertexWeldMap;

        // 1. Process meshes and parse true 4-vertex tetrahedral elements
        for (const Mesh& mesh : model.meshes)
        {
            // First, build a local-to-global mapping for the welded particles
            std::vector<unsigned int> meshToGlobalIndexMap(mesh.vertices.size());

            for (size_t i = 0; i < mesh.vertices.size(); ++i)
            {
                glm::vec3 worldSpacePos = (mesh.vertices[i].Position * scale) + spawnPos;

                auto it = vertexWeldMap.find(worldSpacePos);
                if (it != vertexWeldMap.end())
                {
                    meshToGlobalIndexMap[i] = it->second;
                }
                else
                {
                    SoftBodyParticle p;
                    p.pos = worldSpacePos;
                    p.prevPos = p.pos;
                    p.vel = glm::vec3(0.0f);
                    p.invMass = 1.0f;

                    unsigned int newGlobalIndex = static_cast<unsigned int>(this->particles.size());
                    this->particles.push_back(p);
                    this->particleTexCoords.push_back(mesh.vertices[i].TexCoords);

                    vertexWeldMap[worldSpacePos] = newGlobalIndex;
                    meshToGlobalIndexMap[i] = newGlobalIndex;
                }
            }

            // We assume your model loader outputs indices in sets of 4 for tets, 
            // or sets of 3 that we map. If it's a true tetrahedral mesh parsed as triangles,
            // we process them in chunks. Let's process the indices array into our internal topology:
            for (unsigned int localIndex : mesh.indices)
            {
                this->indices.push_back(meshToGlobalIndexMap[localIndex]);
            }
        }

        // Verify we actually have 4-node volumetric cells to process
        size_t numberOfTetrahedrons = this->indices.size() / 4;
        if (numberOfTetrahedrons == 0) {
            std::cout << "[XPBD ERROR] Input mesh does not contain 4-node tetrahedral elements!" << std::endl;
            return;
        }

        // 2. Identify external surface faces using the hardcoded outward volume tracking matrix
        unsigned int volumeIdxOrder[] = {
            1, 3, 2,  // Face 0
            0, 2, 3,  // Face 1
            0, 3, 1,  // Face 2
            0, 1, 2   // Face 3
        };

        struct SortedFace {
            unsigned int v0, v1, v2;
            bool operator==(const SortedFace& o) const {
                return v0 == o.v0 && v1 == o.v1 && v2 == o.v2;
            }
        };

        auto faceHash = [](const SortedFace& f) {
            return std::hash<unsigned int>()(f.v0) ^ std::hash<unsigned int>()(f.v1) ^ std::hash<unsigned int>()(f.v2);
            };

        std::unordered_map<SortedFace, int, decltype(faceHash)> faceCounts(1024, faceHash);

        // Phase A: Count face occurrences using a sorted key configuration
        for (size_t i = 0; i < numberOfTetrahedrons; i++) {
            unsigned int tet[4] = {
                this->indices[4 * i + 0],
                this->indices[4 * i + 1],
                this->indices[4 * i + 2],
                this->indices[4 * i + 3]
            };

            for (int f = 0; f < 4; f++) {
                unsigned int arr[3] = {
                    tet[volumeIdxOrder[3 * f + 0]],
                    tet[volumeIdxOrder[3 * f + 1]],
                    tet[volumeIdxOrder[3 * f + 2]]
                };
                std::sort(arr, arr + 3);
                faceCounts[SortedFace{ arr[0], arr[1], arr[2] }]++;
            }
        }

        // Phase B: Collect surface triangles using the mathematically guaranteed outward layout
        for (size_t i = 0; i < numberOfTetrahedrons; i++) {
            unsigned int tet[4] = {
                this->indices[4 * i + 0],
                this->indices[4 * i + 1],
                this->indices[4 * i + 2],
                this->indices[4 * i + 3]
            };

            for (int f = 0; f < 4; f++) {
                unsigned int idx0 = tet[volumeIdxOrder[3 * f + 0]];
                unsigned int idx1 = tet[volumeIdxOrder[3 * f + 1]];
                unsigned int idx2 = tet[volumeIdxOrder[3 * f + 2]];

                unsigned int arr[3] = { idx0, idx1, idx2 };
                std::sort(arr, arr + 3);
                SortedFace key{ arr[0], arr[1], arr[2] };

                // If it's only referenced once, it's on the exterior boundary hull
                if (faceCounts[key] == 1) {
                    this->surfaceIndices.push_back(idx0);
                    this->surfaceIndices.push_back(idx1);
                    this->surfaceIndices.push_back(idx2);
                }
            }
        }

        // 3. Generate Edge Constraints from the 6 structural edges of each tetrahedron
        unsigned int edgeIdxOrder[] = {
            0, 1,  0, 2,  0, 3,
            1, 2,  1, 3,  2, 3
        };

        std::unordered_set<unsigned long long> uniqueEdges;
        for (size_t i = 0; i < numberOfTetrahedrons; i++) {
            unsigned int tet[4] = {
                this->indices[4 * i + 0],
                this->indices[4 * i + 1],
                this->indices[4 * i + 2],
                this->indices[4 * i + 3]
            };

            auto registerEdge = [&](unsigned int id0, unsigned int id1) {
                if (id0 == id1) return;
                unsigned int low = std::min(id0, id1);
                unsigned int high = std::max(id0, id1);
                unsigned long long edgeKey = (static_cast<unsigned long long>(low) << 32) | high;

                if (uniqueEdges.insert(edgeKey).second) {
                    this->addConstraint(low, high, compliance);
                }
                };

            // Create the 6 edge constraints that prevent the volume from collapsing
            for (int e = 0; e < 6; e++) {
                registerEdge(tet[edgeIdxOrder[2 * e + 0]], tet[edgeIdxOrder[2 * e + 1]]);
            }
        }

        this->setupBuffers();

        std::cout << "[XPBD TET LOG] Volumetric mesh initialized: " << particles.size()
            << " particles, " << uniqueEdges.size() << " constraints, and "
            << (surfaceIndices.size() / 3) << " outward-facing surface faces extracted." << std::endl;
    }

    void InitializeFromModel(const Model& model, const glm::vec3& spawnPos, const glm::vec3& scale, float compliance)
    {
        this->particles.clear();
        this->constraints.clear();
        this->indices.clear();
        this->particleTexCoords.clear(); // Clear old UV data
        this->cleanupBuffers();

        std::unordered_map<glm::vec3, unsigned int, Vec3Hash> vertexWeldMap;

        for (const Mesh& mesh : model.meshes)
        {
            std::vector<unsigned int> meshToGlobalIndexMap(mesh.vertices.size());

            for (size_t i = 0; i < mesh.vertices.size(); ++i)
            {
                glm::vec3 worldSpacePos = (mesh.vertices[i].Position * scale) + spawnPos;

                auto it = vertexWeldMap.find(worldSpacePos);
                if (it != vertexWeldMap.end())
                {
                    meshToGlobalIndexMap[i] = it->second;
                }
                else
                {
                    SoftBodyParticle p;
                    p.pos = worldSpacePos;
                    p.prevPos = p.pos;
                    p.vel = glm::vec3(0.0f);
                    p.invMass = 1.0f;

                    unsigned int newGlobalIndex = static_cast<unsigned int>(this->particles.size());
                    this->particles.push_back(p);

                    // Keep this line to save the original UV coordinates!
                    this->particleTexCoords.push_back(mesh.vertices[i].TexCoords);

                    vertexWeldMap[worldSpacePos] = newGlobalIndex;
                    meshToGlobalIndexMap[i] = newGlobalIndex;
                }
            }

            for (unsigned int localIndex : mesh.indices)
            {
                this->indices.push_back(meshToGlobalIndexMap[localIndex]);
            }
        }

        // 2. Generate Edge Constraints from Unique Mesh Edges
        std::unordered_set<unsigned long long> uniqueEdges;
        for (size_t i = 0; i < this->indices.size(); i += 3)
        {
            unsigned int i0 = this->indices[i];
            unsigned int i1 = this->indices[i + 1];
            unsigned int i2 = this->indices[i + 2];

            auto registerEdge = [&](unsigned int id0, unsigned int id1) {
                if (id0 == id1) return;
                unsigned int low = std::min(id0, id1);
                unsigned int high = std::max(id0, id1);
                unsigned long long edgeKey = (static_cast<unsigned long long>(low) << 32) | high;

                if (uniqueEdges.insert(edgeKey).second) {
                    this->addConstraint(low, high, compliance);
                }
                };

            registerEdge(i0, i1);
            registerEdge(i1, i2);
            registerEdge(i2, i0);
        }

        this->setupBuffers();

        std::cout << "[XPBD WELD LOG] Welded structural mesh down to " << particles.size()
            << " unique physical particles and " << uniqueEdges.size() << " edge constraints." << std::endl;
    }


    void Step(float dt, int subSteps, glm::vec3 gravity) {
        if (dt <= 0.0f || subSteps <= 0) return;
        float sdt = dt / subSteps;

        for (int step = 0; step < subSteps; ++step) {
            // 1. Predict positions 
            for (auto& p : particles) {
                if (p.invMass == 0.0f) continue;
                p.vel += gravity * sdt;
                p.prevPos = p.pos;
                p.pos += p.vel * sdt;
            }

            // Reset accumulated multipliers at the beginning of each substep sub-iteration loop
            for (auto& c : constraints) {
                c.lambda = 0.0f;
            }

            // 2. Solve Internal Distance Constraints (True XPBD)
            for (auto& c : constraints) {
                SoftBodyParticle& p0 = particles[c.id0];
                SoftBodyParticle& p1 = particles[c.id1];

                float w0 = p0.invMass;
                float w1 = p1.invMass;
                float wSum = w0 + w1;
                if (wSum == 0.0f) continue;

                glm::vec3 dir = p0.pos - p1.pos;
                float len = glm::length(dir);
                if (len < 0.0001f) continue;

                dir /= len; // Faster normalization
                float C = len - c.restLength;

                // XPBD Compliance Evaluation
                float tildeCompliance = c.compliance / (sdt * sdt);
                float denominator = wSum + tildeCompliance;
                if (denominator < 0.0001f) continue;

                // Calculate the true change in lambda, factoring in the constraint history
                float dLambda = (-C - tildeCompliance * c.lambda) / denominator;
                c.lambda += dLambda;

                if (w0 > 0.0f) p0.pos += w0 * dLambda * dir;
                if (w1 > 0.0f) p1.pos -= w1 * dLambda * dir;
            }

            // 3. Resolve Environment Collisions with Stable Friction Dampening
            for (auto& p : particles) {
                if (p.pos.y < 0.0f) {
                    p.pos.y = 0.0f;
                    // Dynamic Slide-Friction instead of completely freezing spatial axes
                    p.pos.x = glm::mix(p.pos.x, p.prevPos.x, 0.2f);
                    p.pos.z = glm::mix(p.pos.z, p.prevPos.z, 0.2f);
                }
            }

            // 4. Velocity Update (Verlet Update rule)
            for (auto& p : particles) {
                if (p.invMass == 0.0f) continue;
                p.vel = (p.pos - p.prevPos) / sdt;
            }
        }

        updateBuffers();
    }

    void Draw(Shader& shader) {
        if (VAO == 0) return;
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

private:

    std::vector<glm::vec2> particleTexCoords;

    void addConstraint(int id0, int id1, float compliance) {
        EdgeConstraint c;
        c.id0 = id0;
        c.id1 = id1;
        c.restLength = glm::distance(particles[id0].pos, particles[id1].pos);
        c.compliance = compliance;
        c.lambda = 0.0f;
        constraints.push_back(c);
    }

    void cleanupBuffers() {
        if (VAO != 0) { glDeleteVertexArrays(1, &VAO); VAO = 0; }
        if (VBO != 0) { glDeleteBuffers(1, &VBO); VBO = 0; }
        if (EBO != 0) { glDeleteBuffers(1, &EBO); EBO = 0; }
    }
    void computeSmoothNormals(std::vector<SoftBodyVertex>& outVertices) {
        // Initialize/reset position properties and clear accumulated normals
        for (size_t i = 0; i < particles.size(); ++i) {
            outVertices[i].Position = particles[i].pos;
            outVertices[i].Normal = glm::vec3(0.0f);
            // Apply mapped texture tracking values if populated
            if (i < particleTexCoords.size()) {
                outVertices[i].TexCoords = particleTexCoords[i];
            }
            else {
                outVertices[i].TexCoords = glm::vec2(0.0f);
            }
        }

        // Accumulate face plane cross-products for perfect smooth deforming normals
        for (size_t i = 0; i < indices.size(); i += 3) {
            unsigned int idx0 = indices[i];
            unsigned int idx1 = indices[i + 1];
            unsigned int idx2 = indices[i + 2];

            glm::vec3 v0 = particles[idx0].pos;
            glm::vec3 v1 = particles[idx1].pos;
            glm::vec3 v2 = particles[idx2].pos;

            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 faceNormal = glm::cross(edge1, edge2);

            outVertices[idx0].Normal += faceNormal;
            outVertices[idx1].Normal += faceNormal;
            outVertices[idx2].Normal += faceNormal;
        }

        // Normalize vectors to make them unit length
        for (size_t i = 0; i < outVertices.size(); ++i) {
            if (glm::length(outVertices[i].Normal) > 0.0001f) {
                outVertices[i].Normal = glm::normalize(outVertices[i].Normal);
            }
        }
    }

    void setupBuffers() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        // Pre-allocate structural vertex layout data array
        std::vector<SoftBodyVertex> vertices(particles.size());
        // Initial normal generation step
        computeSmoothNormals(vertices);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(SoftBodyVertex), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        // Attribute 0: Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SoftBodyVertex), (void*)offsetof(SoftBodyVertex, Position));

        // Attribute 1: Normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SoftBodyVertex), (void*)offsetof(SoftBodyVertex, Normal));

        // Attribute 2: UV Texture Coordinates
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(SoftBodyVertex), (void*)offsetof(SoftBodyVertex, TexCoords));

        glBindVertexArray(0);
    }

    void updateBuffers() {
        if (VBO == 0) return;

        std::vector<SoftBodyVertex> vertices(particles.size());
        computeSmoothNormals(vertices);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(SoftBodyVertex), vertices.data());
    }


};

#endif