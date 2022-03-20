//
//  Shaders.metal
//  LearnMetal
//
//  Created by  kevintsuixu on 2022/3/20.
//

#include <metal_stdlib>
using namespace metal;

// 创建一个顶点结构体，包含顶点位置和颜色
struct Vertex{
    float4 position [[position]];
    float4 color;
};

/** 创建VertexShader*/
vertex Vertex basic_vertex(constant Vertex *vertices [[buffer(0)]],uint vid [[vertex_id]]){
    return vertices[vid];
}

/** 创建FragmentShader*/
fragment float4 basic_fragment(Vertex vert [[stage_in]]){
    return vert.color;
}
