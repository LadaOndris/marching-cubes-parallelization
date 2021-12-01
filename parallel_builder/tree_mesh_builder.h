/**
 * @file    tree_mesh_builder.h
 *
 * @author  FULL NAME <xondri07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"
#include <cmath>

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    explicit TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    void marchCubesRecurse(const Vec3_t<float> &pos, const Vec3_t<float> &centerPoint, unsigned depth, const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }
    std::vector<Triangle_t> mTriangles; ///< Temporary array of triangles
    std::vector<std::vector<Triangle_t>> threadsTriangles;
    std::vector<unsigned> box_widths;
    std::vector<float> box_widths_float;
    std::vector<float> limits;
    std::vector<unsigned> cubeCounts;
    std::vector<Vec3_t<float>> cutoff_box_indices;
    std::vector<float> box_widths_physical;
    std::vector<unsigned> totalThreadTriangles;
private:
    unsigned max_depth = static_cast<unsigned>(sqrt(mGridSize));
    unsigned cutoff_depth = 4;
};

#endif // TREE_MESH_BUILDER_H
