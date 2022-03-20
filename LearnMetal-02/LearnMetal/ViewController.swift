//
//  ViewController.swift
//  LearnMetal
//
//  Created by  kevintsuixu on 2022/3/20.
//

import UIKit
import Metal
import simd

class ViewController: UIViewController {
    
    // 创建顶点结构体，包含顶点位置和顶点色
    struct Vertex {
        var position: vector_float4
        var color: vector_float4
    }
    
    // 创建顶点数据
    let vertexData = [Vertex(position: [0.0, 1.0, 0.0, 1.0], color: [1,0,0,1]),Vertex(position: [-1.0, -1.0, 0.0, 1.0], color: [0,1,0,1]),Vertex(position: [ 1.0, -1.0, 0.0, 1.0], color: [0,0,1,1])]

    var device: MTLDevice! // 指定硬件
    var metalLayer: CAMetalLayer! // 绘制图层
    var vertexBuffer: MTLBuffer! // VertexBuffer
    var pipelineState: MTLRenderPipelineState! // 渲染管线
    var commandQueue: MTLCommandQueue! // 指令队列
    
    var timer: CADisplayLink! // 显示器刷新率同步计时器
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.

        /**-------------------------------------------------*/
        /** 指定默认硬件设备**/
        device = MTLCreateSystemDefaultDevice()
        
        /**-------------------------------------------------*/
        /** 创建CAMetalLayer**/
        metalLayer = CAMetalLayer()          // 创建一个新的CAMetalLayer
        metalLayer.device = device           // 指定MTLDevice
        metalLayer.pixelFormat = .bgra8Unorm // 设置像素格式 8bit的RGBA
        metalLayer.framebufferOnly = true    // 除非需要采样本层生成的贴图或开启新的绘制线程，设为true
        metalLayer.frame = view.layer.frame  // 匹配view的layer帧数
        view.layer.addSublayer(metalLayer)   // 加入主层，新创建的CAMetalLayer为子层
        
        /**-------------------------------------------------*/
        /** 创建VertexBuffer*/
        // 计算VertexData的存储大小
        let dataSize = vertexData.count * MemoryLayout.size(ofValue: vertexData[0])
        // 在指定硬件的GPU上创建一个新的Buffer，从CPU往GPU的Buffer上发送数据。这里创建了一个空白的Buffer
        vertexBuffer = device.makeBuffer(bytes: vertexData, length: dataSize, options: [])

        /**-------------------------------------------------*/
        /** 创建RenderPipeline渲染管线*/
        // 通过调用makeDefaultLibrary可以访问所有预编译的Shader
        let defaultLibrary = device.makeDefaultLibrary()!
        let fragmentProgram = defaultLibrary.makeFunction(name: "basic_fragment")
        let vertexProgram = defaultLibrary.makeFunction(name: "basic_vertex")
        // 配置RenderPipeline参数，渲染输出的层就是CAMetalLayer本身
        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.vertexFunction = vertexProgram
        pipelineStateDescriptor.fragmentFunction = fragmentProgram
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        // 编译RenderPipeline为PipelineState，调用更加高效
        pipelineState = try! device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
        
        /**-------------------------------------------------*/
        /** 创建指令队列*/
        commandQueue = device.makeCommandQueue()
                
        /**-------------------------------------------------*/
        /** 初始化计时器*/
        timer = CADisplayLink(target: self, selector: #selector(gameloop))
        timer.add(to: RunLoop.main, forMode: .default)

    } // viewDidLoad

    func render() {
        /**-------------------------------------------------*/
        /** 创建Render Pass Descriptor, 指定贴图渲染到什么位置，ClearColor等参数*/
        guard let drawable = metalLayer?.nextDrawable() else { return } // 拿到之前上一帧的绘制
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture // 拿到绘制的贴图
        renderPassDescriptor.colorAttachments[0].loadAction = .clear // 贴图绘制之前先Clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 104.0/255.0, blue: 55.0/255.0, alpha: 1.0)

        /**-------------------------------------------------*/
        /** 创建指令队列的Buffer*/
        let commandBuffer = commandQueue.makeCommandBuffer()!
            
        /**-------------------------------------------------*/
        /** 创建指令队列的编码器*/
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0) // 指定VertexBuffer
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3, instanceCount: 1)
        renderEncoder.endEncoding()

        /**-------------------------------------------------*/
        /** 发送渲染指令*/
        commandBuffer.present(drawable) // 确保GPU持有现在正在绘制的贴图
        commandBuffer.commit() // 发送指令
    } // render
    
    @objc func gameloop() {
      autoreleasepool {
        self.render()
      }
    } // gameloop

  } // class UIViewController

