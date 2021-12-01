/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xondri07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
        : BaseMeshBuilder(gridEdgeSize, "Octree") {

    // Initialize mGridSizes
    for (unsigned box_width = mGridSize; box_width > 0; box_width /= 2) {
        box_widths.push_back(box_width);
        box_widths_float.push_back(static_cast<float>(box_width));

        float limit = mIsoLevel + sqrt(3.0) / 2.0 * box_width;
        limits.push_back(limit);

        cubeCounts.push_back(box_width * box_width * box_width);
    }

    unsigned box_width = box_widths[cutoff_depth];
    for (unsigned i = 0; i < cubeCounts[cutoff_depth]; ++i) {
        Vec3_t<float> vec(i % box_width,
                          (i / box_width) % box_width,
                          i / (box_width * box_width));
        cutoff_box_indices.push_back(vec);
    }
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field) {
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    auto max_threads = omp_get_max_threads();
    std::vector<std::vector<Triangle_t>> triangles(max_threads);
    threadsTriangles = triangles;

    for (auto &depth : box_widths_float) {
        box_widths_physical.push_back(depth * mGridResolution);
    }

#pragma omp parallel default(none) firstprivate(field)
#pragma omp single
    {
        unsigned depth = 0;
        Vec3_t<float> centerPoint((0 + box_widths[depth + 1]) * mGridResolution);
        marchCubesRecurse(Vec3_t<float>(0), centerPoint, depth, field);
#pragma omp taskwait
    };

    for (auto const &triangleVector: threadsTriangles) {
        mTriangles.insert(std::end(mTriangles), std::begin(triangleVector), std::end(triangleVector));
    }

    return mTriangles.size();
}

void
TreeMeshBuilder::marchCubesRecurse(const Vec3_t<float> &pos, const Vec3_t<float> &centerPoint,
                                   unsigned depth, const ParametricScalarField &field) {
    float a_div2 = box_widths_float[depth + 1];

    // If we can't go any deeper.
    if (depth == max_depth) {
        buildCube(pos, field);
    } else if (depth == cutoff_depth) {
        unsigned cubesCount = cubeCounts[depth];
        for (unsigned i = 0; i < cubesCount; ++i) {
            Vec3_t<float> offset_index = cutoff_box_indices[i];
            Vec3_t<float> cubeOffset(pos.x + offset_index.x,
                                     pos.y + offset_index.y,
                                     pos.z + offset_index.z);
            buildCube(cubeOffset, field);
        }
    } else {
        Vec3_t<float> vecs[8]{
                Vec3_t<float>(pos.x, pos.y, pos.z),
                Vec3_t<float>(pos.x + a_div2, pos.y, pos.z),
                Vec3_t<float>(pos.x, pos.y + a_div2, pos.z),
                Vec3_t<float>(pos.x + a_div2, pos.y + a_div2, pos.z),
                Vec3_t<float>(pos.x, pos.y, pos.z + a_div2),
                Vec3_t<float>(pos.x + a_div2, pos.y, pos.z + a_div2),
                Vec3_t<float>(pos.x, pos.y + a_div2, pos.z + a_div2),
                Vec3_t<float>(pos.x + a_div2, pos.y + a_div2, pos.z + a_div2)
        };
        float a_div4 = box_widths_physical[depth + 2];
        for (auto &vec : vecs) {
            Vec3_t<float> nextCenterPoint(vec.x * mGridResolution + a_div4,
                                          vec.y * mGridResolution + a_div4,
                                          vec.z * mGridResolution + a_div4);

            float F_p = evaluateFieldAt(centerPoint, field);
            if (F_p > limits[depth + 1]) {
                continue;
            }

#pragma omp task default(none) firstprivate(vec, nextCenterPoint, depth, field)
            {
                marchCubesRecurse(vec, nextCenterPoint, depth + 1, field);
            }
        }
    }
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos,
                                       const ParametricScalarField &field) {// NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
#pragma omp simd reduction(min:value)
    for (unsigned i = 0; i < count; ++i) {
        float distanceSquared = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
    int threadNum = omp_get_thread_num();
    threadsTriangles[threadNum].push_back(triangle);
}
