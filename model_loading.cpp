#define GLM_ENABLE_EXPERIMENTAL

#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp> // Required for rotation math

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iostream>
#include <vector>
#include <limits>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <iostream>

#define USE_26_DOP true 

#if USE_26_DOP
const int K_NUM_AXES = 13;
const glm::vec3 KDOP_NORMALS[13] = {
    // 3 Principal Face Axes (AABB faces)
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},

    // 6 Edge Diagonal Axes
    {1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f},
    {1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, -1.0f},
    {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, -1.0f},

    // 4 Corner Diagonal Axes (This matches the 8-DOP axes!)
    {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, -1.0f},
    {1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, -1.0f}
};
#else
const int K_NUM_AXES = 4;
const glm::vec3 KDOP_NORMALS[4] = {
    { 1.0f,  1.0f,  1.0f},
    { 1.0f,  1.0f, -1.0f},
    { 1.0f, -1.0f,  1.0f},
    { 1.0f, -1.0f, -1.0f}
};
#endif

// Global normalized axes
glm::vec3 G_Normals[K_NUM_AXES];
void initializeNormals() {
    for (int i = 0; i < K_NUM_AXES; ++i) {
        G_Normals[i] = glm::normalize(KDOP_NORMALS[i]);
    }
}

// --- k-DOP Structure ---
struct KDOP {
    float min_intervals[K_NUM_AXES];
    float max_intervals[K_NUM_AXES];
};

struct Entity {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 scale;
    KDOP localDOP;   // Cached baseline relative to object center
    KDOP worldDOP;   // Updated dynamically based on translation

    void calculateLocalDOP(const std::vector<glm::vec3>& vertices) {
        for (int i = 0; i < K_NUM_AXES; ++i) {
            localDOP.min_intervals[i] = std::numeric_limits<float>::infinity();
            localDOP.max_intervals[i] = -std::numeric_limits<float>::infinity();

            for (const auto& v : vertices) {
                // Apply object scale to vertices during extraction
                float proj = glm::dot(v * scale, G_Normals[i]);
                if (proj < localDOP.min_intervals[i]) localDOP.min_intervals[i] = proj;
                if (proj > localDOP.max_intervals[i]) localDOP.max_intervals[i] = proj;
            }
        }
    }

    void updateWorldDOP() {
        for (int i = 0; i < K_NUM_AXES; ++i) {
            // Fast translation shifting: projection of position onto the plane axis
            float offset = glm::dot(position, G_Normals[i]);
            worldDOP.min_intervals[i] = localDOP.min_intervals[i] + offset;
            worldDOP.max_intervals[i] = localDOP.max_intervals[i] + offset;
        }
    }
};

// --- Helper Functions to extract data from LearnOpenGL Model class ---
void extractVertices(const Mesh& mesh, std::vector<glm::vec3>& allVertices) {
    for (unsigned int i = 0; i < mesh.vertices.size(); i++) {
        allVertices.push_back(mesh.vertices[i].Position);
    }
}
std::vector<glm::vec3> getModelVertices(Model& model) {
    std::vector<glm::vec3> allVertices;
    for (unsigned int i = 0; i < model.meshes.size(); i++) {
        extractVertices(model.meshes[i], allVertices);
    }
    return allVertices;
}

// --- Collision Logic ---
bool checkCollision(const KDOP& a, const KDOP& b) {
    for (int i = 0; i < K_NUM_AXES; ++i) {
        if (a.max_intervals[i] < b.min_intervals[i] || b.max_intervals[i] < a.min_intervals[i]) {
            return false; // Found a separating axis
        }
    }
    return true; // Intersects on all axes
}

// Callbacks & Camera Configuration
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
void UpdateImGui();

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

Camera camera(glm::vec3(0.0f, 2.0f, 15.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool isHoldingAlt = false;
bool isPause = false;
bool isShowingBounds = false;

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "k-DOP Collision Scene", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    initializeNormals();
    stbi_set_flip_vertically_on_load(false);
    glEnable(GL_DEPTH_TEST);


    Shader ourShader("1.model_loading.vs", "1.model_loading.fs");
    Shader kDopShader("kdop.vs", "kdop.fs");

    Model ourModel(FileSystem::getPath("resources/objects/backpack/backpack.obj"));
    std::vector<glm::vec3> modelVertices = getModelVertices(ourModel);

    // --- Instantiate Dynamic Translating Entities ---
    std::vector<Entity> entities;
    const int NUM_OBJECTS = 100;

    for (int i = 0; i < NUM_OBJECTS; ++i) {
        Entity bp;

        // Disperse backpacks cleanly throughout an expanded world space volume
        float posX = -8.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 16.0f));
        float posY = -8.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 16.0f));
        float posZ = -8.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 16.0f));
        bp.position = glm::vec3(posX, posY, posZ);

        // Assign randomized velocities so they drift in different directions
        float velX = -3.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 6.0f));
        float velY = -3.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 6.0f));
        float velZ = -3.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 6.0f));
        bp.velocity = glm::vec3(velX, velY, velZ);

        // Scale them down a bit so 50 objects can breathe in the viewport
        bp.scale = glm::vec3(0.5f);
        bp.calculateLocalDOP(modelVertices);

        entities.push_back(bp);
    }

    // Track intersection flags independently per object for rendering
    std::vector<bool> objectCollidingFlags(NUM_OBJECTS, false);

    // --- Generate Plane Mesh for Visualizing Bounds ---
    // A single simple generic plane primitive centered horizontally
    float planeSize = 0.5f;
    float planeVertices[] = {
        -planeSize, 0.0f, -planeSize,
         planeSize, 0.0f, -planeSize,
         planeSize, 0.0f,  planeSize,
        -planeSize, 0.0f,  planeSize
    };
    unsigned int planeIndices[] = { 0, 1, 2, 0, 2, 3 };

    unsigned int planeVAO, planeVBO, planeEBO;
    glGenVertexArrays(1, &planeVAO);
    glGenBuffers(1, &planeVBO);
    glGenBuffers(1, &planeEBO);

    glBindVertexArray(planeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(planeIndices), planeIndices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // --- Core Engine Render Loop ---
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        if (!isPause)
        {
            // 1. Physics Phase: Move and constrain inside expanded boundary limits
            for (auto& entity : entities) {
                entity.position += entity.velocity * deltaTime;

                // Expanded bounding arena so 50 entities don't choke instantly
                if (abs(entity.position.x) > 10.0f)  entity.velocity.x *= -1.0f;
                if (abs(entity.position.y) > 10.0f)  entity.velocity.y *= -1.0f;
                if (abs(entity.position.z) > 10.0f)  entity.velocity.z *= -1.0f;

                entity.updateWorldDOP();
            }
        }
        

        // Reset intersection flags every frame
        std::fill(objectCollidingFlags.begin(), objectCollidingFlags.end(), false);

        // 2. Pair-Wise O(N^2) Collision Resolution Check
        for (size_t i = 0; i < entities.size(); ++i) {
            for (size_t j = i + 1; j < entities.size(); ++j) {
                if (checkCollision(entities[i].worldDOP, entities[j].worldDOP)) {
                    objectCollidingFlags[i] = true;
                    objectCollidingFlags[j] = true;

                    //// Elastic momentum bounce response
                    //glm::vec3 temp = entities[i].velocity;
                    //entities[i].velocity = entities[j].velocity;
                    //entities[j].velocity = temp;

                    //// Prevent interlocking sticky states by instantly stepping position values
                    //entities[i].position += entities[i].velocity * deltaTime;
                    //entities[j].position += entities[j].velocity * deltaTime;
                }
            }
        }

        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();

        // Pass 1: Render Opaque Backpacks
        ourShader.use();
        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);

        for (const auto& entity : entities) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, entity.position);
            model = glm::scale(model, entity.scale);
            ourShader.setMat4("model", model);
            ourModel.Draw(ourShader);
        }

        if (isShowingBounds)
        {
            // Pass 2: Render Transparent Volumetric Planes
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_FALSE);

            kDopShader.use();
            kDopShader.setMat4("projection", projection);
            kDopShader.setMat4("view", view);

            glBindVertexArray(planeVAO);
            for (size_t idx = 0; idx < entities.size(); ++idx) {
                const auto& entity = entities[idx];

                // Assign custom color uniform dependent on the specific entity's state
                glm::vec4 volumeColor = objectCollidingFlags[idx] ? glm::vec4(1.0f, 0.2f, 0.2f, 0.25f) : glm::vec4(0.2f, 0.7f, 1.0f, 0.10f);
                kDopShader.setVec4("planeColor", volumeColor);

                for (int i = 0; i < K_NUM_AXES; ++i) {
                    glm::vec3 norm = G_Normals[i];
                    glm::mat4 rotationMatrix = glm::toMat4(glm::rotation(glm::vec3(0.0f, 1.0f, 0.0f), norm));

                    // Max Bound Plane
                    glm::mat4 modelMax = glm::mat4(1.0f);
                    modelMax = glm::translate(modelMax, entity.position);
                    modelMax = glm::translate(modelMax, norm * entity.localDOP.max_intervals[i]);
                    modelMax = modelMax * rotationMatrix;
                    kDopShader.setMat4("model", modelMax);
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                    // Min Bound Plane
                    glm::mat4 modelMin = glm::mat4(1.0f);
                    modelMin = glm::translate(modelMin, entity.position);
                    modelMin = glm::translate(modelMin, norm * entity.localDOP.min_intervals[i]);
                    modelMin = modelMin * rotationMatrix;
                    kDopShader.setMat4("model", modelMin);
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
                }
            }

            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
        }
        
        UpdateImGui();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &planeVAO);
    glDeleteBuffers(1, &planeVBO);
    glDeleteBuffers(1, &planeEBO);

    glfwTerminate();
    return 0;
}

void UpdateImGui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(550, 0), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_Once);




    ImGui::Begin("Shader Setting");

    ImGui::Checkbox("isShowingBounds", &isShowingBounds);
    ImGui::Checkbox("isPause", &isPause);

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


// Input processing and window adjustment callbacks unchanged below
void processInput(GLFWwindow* window) {

    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) {
        if (!isHoldingAlt) { // Only call GLFW state changes once per toggle change
            isHoldingAlt = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
    else {
        if (isHoldingAlt) {
            isHoldingAlt = false;
            firstMouse = true; // Crucial: resets mouse drag jumps when re-entering window
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_RELEASE)
    {
        isHoldingAlt = false;
    }

    if (!isHoldingAlt) {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.ProcessKeyboard(RIGHT, deltaTime);
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    if (isHoldingAlt) return;

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) {
        lastX = xpos; lastY = ypos; firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (!isHoldingAlt)
        camera.ProcessMouseScroll(static_cast<float>(yoffset));
}