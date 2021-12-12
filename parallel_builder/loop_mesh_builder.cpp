/**
 * @file    loop_mesh_builder.cpp
 *
 * @author  FULL NAME <xondri07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "loop_mesh_builder.h"

LoopMeshBuilder::LoopMeshBuilder(unsigned gridEdgeSize)
        : BaseMeshBuilder(gridEdgeSize, "OpenMP Loop") {

}

unsigned LoopMeshBuilder::marchCubes(const ParametricScalarField &field) {
    // 1. Compute total number of cubes in the grid.
    size_t totalCubesCount = mGridSize * mGridSize * mGridSize;

    unsigned totalTriangles = 0;
    auto gridSize = mGridSize;

    auto max_threads = omp_get_max_threads();
    std::vector<std::vector<Triangle_t>> triangles(max_threads);
    threadsTriangles = triangles;

    for (auto &point : field.getPoints()) {
        pPointsX.push_back(point.x);
        pPointsY.push_back(point.y);
        pPointsZ.push_back(point.z);
    }

    // 2. Loop over each coordinate in the 3D grid.
    #pragma omp parallel for schedule(runtime) default(none) firstprivate(totalCubesCount, gridSize) shared(field) reduction(+:totalTriangles)
    for (size_t i = 0; i < totalCubesCount; ++i) {
        // 3. Compute 3D position in the grid.
        Vec3_t<float> cubeOffset(i % gridSize,
                                 (i / gridSize) % gridSize,
                                 i / (gridSize * gridSize));

        // 4. Evaluate "Marching Cube" at given position in the grid and
        //    store the number of triangles generated.
        totalTriangles += buildCube(cubeOffset, field);
    }

    for(auto const& triangleVector: threadsTriangles) {
        mTriangles.insert(std::end(mTriangles), std::begin(triangleVector), std::end(triangleVector));
    }

    // 5. Return total number of triangles generated.
    return totalTriangles;
}

float LoopMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) {
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    const float *pointsX = pPointsX.data();
    const float *pointsY = pPointsY.data();
    const float *pointsZ = pPointsZ.data();

    #pragma omp simd reduction(min:value)
    for (unsigned i = 0; i < count; ++i) {
        float distanceSquared = (pos.x - pointsX[i]) * (pos.x - pointsX[i]);
        distanceSquared += (pos.y - pointsY[i]) * (pos.y - pointsY[i]);
        distanceSquared += (pos.z - pointsZ[i]) * (pos.z - pointsZ[i]);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.

    //#pragma omp critical
    //mTriangles.push_back(triangle);
    int threadNum = omp_get_thread_num();
    threadsTriangles[threadNum].push_back(triangle);

}
