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

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field) {
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    auto max_threads = omp_get_max_threads();
    std::vector<std::vector<Triangle_t>> triangles(max_threads);
    threadsTriangles = triangles;

    unsigned totalTriangles;
#pragma omp parallel default(none) shared(field, totalTriangles)
#pragma omp single
    {
        totalTriangles = marchCubesRecurse(Vec3_t<float>(0), mGridSize, field);
    };

    for(auto const& triangleVector: threadsTriangles) {
        mTriangles.insert(std::end(mTriangles), std::begin(triangleVector), std::end(triangleVector));
    }

    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubesRecurse(const Vec3_t<float> &pos, int a_int, const ParametricScalarField &field) {
    auto a = static_cast<float>(a_int);
    int a_div2_int = a_int / 2;
    float a_div2 = static_cast<float>(a_div2_int);

    float limit = mIsoLevel + sqrt(3.0) / 2.0 * a;
    Vec3_t<float> centerPoint((pos.x + a_div2) * mGridResolution,
                              (pos.y + a_div2) * mGridResolution,
                              (pos.z + a_div2) * mGridResolution);
    float F_p = evaluateFieldAt(centerPoint, field);
    if (F_p > limit) {
        return 0;
    } else {
        // If we can't go any deeper.
        if (a_int == 1) {
            return buildCube(pos, field);

        } else if (a_int == cutoff) {
            int cubesCount = a_int * a_int * a_int;
            unsigned totalTriangles = 0;
            for (int i = 0; i < cubesCount; ++i) {
                Vec3_t<float> cubeOffset(pos.x + i % a_int,
                                         pos.y + (i / a_int) % a_int,
                                         pos.z + i / (a_int * a_int));
                totalTriangles += buildCube(cubeOffset, field);
            }
            return totalTriangles;
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

            unsigned totalTriangles = 0;
            for (auto &vec : vecs) {
#pragma omp task default(none) shared(totalTriangles, field) firstprivate(pos, a_div2_int, vec)
                {
#pragma omp atomic update
                    totalTriangles += marchCubesRecurse(vec, a_div2_int, field);
                }
            }
#pragma omp taskwait
            return totalTriangles;
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
//#pragma omp critical
//    mTriangles.push_back(triangle);

    int threadNum = omp_get_thread_num();
    threadsTriangles[threadNum].push_back(triangle);
}
