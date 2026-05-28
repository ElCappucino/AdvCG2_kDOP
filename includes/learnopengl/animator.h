#pragma once

#include <glm/glm.hpp>
#include <map>
#include <vector>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <learnopengl/animation.h>
#include <learnopengl/bone.h>
#include <glm/gtx/quaternion.hpp> // Ensure glm quaternion helpers are accessible
#include <glm/gtx/matrix_decompose.hpp>

class Animator
{
private:
    bool m_AnimationFinished = false;
public:
    

    Animator(Animation* animation)
    {
        m_CurrentTime = 0.0f;
        m_CurrentTime2 = 0.0f;
        m_CurrentAnimation = animation;
        m_CurrentAnimation2 = nullptr;
        m_blendAmount = 0.0f;
        m_isBlending = false;
        m_blendSpeed = 1.0f;

        m_FinalBoneMatrices.reserve(100);
        for (int i = 0; i < 100; i++)
            m_FinalBoneMatrices.push_back(glm::mat4(1.0f));
    }

    // Call this once in main.cpp when shifting states to initiate a clean crossfade transition
    void CrossFade(Animation* targetAnimation, float durationInSeconds)
    {
        // If we don't have an active animation, just switch instantly
        if (!m_CurrentAnimation)
        {
            m_CurrentAnimation = targetAnimation;
            m_CurrentTime = 0.0f;
            return;
        }

        // If trying to transition to the same active clip, ignore to prevent visual snapping
        if (m_CurrentAnimation == targetAnimation) return;

        m_CurrentAnimation2 = targetAnimation;
        m_CurrentTime2 = 0.0f;
        m_blendAmount = 0.0f;
        m_isBlending = true;
        // speed = 1 / duration (e.g., 0.2 second blend means speed = 5.0f)
        m_blendSpeed = (durationInSeconds > 0.0f) ? (1.0f / durationInSeconds) : 100.0f;
    }

    void UpdateAnimation(float dt)
    {
        m_DeltaTime = dt;

        m_AnimationFinished = false;

        if (m_CurrentAnimation)
        {
            // Advance Clip 1
            m_CurrentTime += m_CurrentAnimation->GetTicksPerSecond() * dt;

            if (m_CurrentTime >= m_CurrentAnimation->GetDuration())
            {
                m_AnimationFinished = true;
                m_CurrentTime = fmod(m_CurrentTime, m_CurrentAnimation->GetDuration());
            }

            // Handle active crossfade ticking
            if (m_isBlending && m_CurrentAnimation2)
            {
                // Advance Clip 2
                m_CurrentTime2 += m_CurrentAnimation2->GetTicksPerSecond() * dt;
                m_CurrentTime2 = fmod(m_CurrentTime2, m_CurrentAnimation2->GetDuration());

                // Update blend factor using deltaTime (independent of framerate)
                m_blendAmount += dt * m_blendSpeed;
                if (m_blendAmount >= 1.0f)
                {
                    m_isBlending = false;
                    m_CurrentAnimation = m_CurrentAnimation2;
                    m_CurrentTime = m_CurrentTime2;
                    m_CurrentAnimation2 = nullptr;
                    m_blendAmount = 0.0f;
                }
            }

            CalculateBoneTransform(&m_CurrentAnimation->GetRootNode(), glm::mat4(1.0f));
        }
    }

    // Fallback-safe TRS generator matrix calculation
    glm::mat4 UpdateBlend(Bone* Bone1, Bone* Bone2, const glm::mat4& defaultTransform)
    {
        glm::vec3 bonePos1, bonePos2, finalPos;
        glm::vec3 boneScale1, boneScale2, finalScale;
        glm::quat boneRot1, boneRot2, finalRot;

        // Extract clip 1 positions or extract from default bind pose if bone track missing
        if (Bone1) {
            Bone1->InterpolatePosition(m_CurrentTime, bonePos1);
            Bone1->InterpolateRotation(m_CurrentTime, boneRot1);
            Bone1->InterpolateScaling(m_CurrentTime, boneScale1);
        }
        else {
            // Deconstruct default matrix elements as fallback
            float dummy;
            glm::vec3 dummySkew;
            glm::vec4 dummyPerspective;
            glm::decompose(defaultTransform, boneScale1, boneRot1, bonePos1, dummySkew, dummyPerspective);
        }

        // Extract clip 2 positions or extract from default bind pose if bone track missing
        if (Bone2) {
            Bone2->InterpolatePosition(m_CurrentTime2, bonePos2);
            Bone2->InterpolateRotation(m_CurrentTime2, boneRot2);
            Bone2->InterpolateScaling(m_CurrentTime2, boneScale2);
        }
        else {
            float dummy;
            glm::vec3 dummySkew;
            glm::vec4 dummyPerspective;
            glm::decompose(defaultTransform, boneScale2, boneRot2, bonePos2, dummySkew, dummyPerspective);
        }

        // Blend structural components smoothly
        finalPos = glm::mix(bonePos1, bonePos2, m_blendAmount);
        finalRot = glm::slerp(boneRot1, boneRot2, m_blendAmount);
        finalRot = glm::normalize(finalRot);
        finalScale = glm::mix(boneScale1, boneScale2, m_blendAmount);

        glm::mat4 translation = glm::translate(glm::mat4(1.0f), finalPos);
        glm::mat4 rotation = glm::toMat4(finalRot);
        glm::mat4 scale = glm::scale(glm::mat4(1.0f), finalScale);

        return translation * rotation * scale;
    }

    void CalculateBoneTransform(const AssimpNodeData* node, glm::mat4 parentTransform)
    {
        std::string nodeName = node->name;
        glm::mat4 nodeTransform = node->transformation;

        Bone* Bone1 = m_CurrentAnimation->FindBone(nodeName);
        Bone* Bone2 = m_CurrentAnimation2 ? m_CurrentAnimation2->FindBone(nodeName) : nullptr;

        if (m_isBlending)
        {
            // --- FIX: Force underlying bones to update their internal structures before blending ---
            if (Bone1) Bone1->Update(m_CurrentTime);
            if (Bone2) Bone2->Update(m_CurrentTime2);

            if (Bone1 || Bone2)
            {
                nodeTransform = UpdateBlend(Bone1, Bone2, node->transformation);
            }
        }
        else if (Bone1)
        {
            Bone1->Update(m_CurrentTime);
            nodeTransform = Bone1->GetLocalTransform();
        }

        glm::mat4 globalTransformation = parentTransform * nodeTransform;
        m_BoneGlobals[nodeName] = globalTransformation;

        auto boneInfoMap = m_CurrentAnimation->GetBoneIDMap();
        if (boneInfoMap.find(nodeName) != boneInfoMap.end())
        {
            int index = boneInfoMap[nodeName].id;
            glm::mat4 offset = boneInfoMap[nodeName].offset;
            m_FinalBoneMatrices[index] = globalTransformation * offset;
        }

        for (int i = 0; i < node->childrenCount; i++)
            CalculateBoneTransform(&node->children[i], globalTransformation);
    }

    std::vector<glm::mat4> GetFinalBoneMatrices() { return m_FinalBoneMatrices; }

    bool IsBlending() const { return m_isBlending; }
    float GetBlendAmount() const { return m_blendAmount; }
    float GetCurrentTime() const { return m_CurrentTime; }
    bool AnimationFinished() const { return m_AnimationFinished; }

    // Legacy method signature maintained so it won't break your build immediately
    void PlayAnimation(Animation* pAnimation, Animation* pAnimation2, float time1, float time2, float blend)
    {
        m_CurrentAnimation = pAnimation;
        m_CurrentTime = time1;
        m_CurrentAnimation2 = pAnimation2;
        m_CurrentTime2 = time2;
        m_blendAmount = blend;
        m_isBlending = (pAnimation2 != nullptr);
    }

    // Public variable hooks accessed by your main loops
    float m_CurrentTime;
    float m_CurrentTime2;

private:
    std::vector<glm::mat4> m_FinalBoneMatrices;
    Animation* m_CurrentAnimation;
    Animation* m_CurrentAnimation2;
    float m_DeltaTime;
    float m_blendAmount;
    float m_blendSpeed;
    bool m_isBlending;
    std::map<std::string, glm::mat4> m_BoneGlobals;
};